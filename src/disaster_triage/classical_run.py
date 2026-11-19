"""Private, resumable CPU campaigns for the classical model-selection runs."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import fcntl
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import signal
import tarfile
import tempfile
import time
import uuid

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from threadpoolctl import threadpool_limits

from .accounting import Attempt, guard_budget, identity, utc_now
from .bundles import BundleExpectation, read_bundle, write_bundle
from .classical import (
    CANDIDATES, TIE_TOLERANCE, VECTORIZER, ClassicalError, checked_iterations,
    decision_scores, effective_weights, fit_features, fit_lr, load_inputs,
    majority_predictions, rank_lr, training_counts,
)
from .constants import CUTOFF, LABELS
from .evaluation import BundleInput, score_bundle
from .profiles import PROFILES
from .provenance import json_bytes, save_json, sha256_bytes, sha256_file, source_snapshot

RUNS = ("majority", *CANDIDATES)
HEARTBEAT_SECONDS = 15.0
FEATURE_FILES = ("vectorizer.joblib", "train.npz", "selection.npz")
PREDICTION_FILES = ("selection/manifest.json", "selection/predictions.npz")


class DeadlineReached(KeyboardInterrupt):
    """The supervised research stage reached the shared experiment limit."""


def _read_json(path):
    return json.loads(path.read_bytes())


def _runtime():
    dependencies = ("numpy", "scipy", "scikit-learn", "datasketch", "joblib", "threadpoolctl")
    installed = {package: version(package) for package in dependencies}
    python = platform.python_version()
    compatible = any(
        profile["python"] == python
        and all(installed[key] == profile["packages"][key] for key in dependencies[:4])
        for profile in PROFILES.values()
    )
    if not compatible:
        raise ClassicalError("Use a pinned project Python, NumPy, SciPy, scikit-learn, and datasketch profile.")
    return {
        "python": python, **installed, "system": platform.system(),
        "machine": platform.machine(), "numeric_threads": 1, "device": "cpu",
    }


@contextmanager
def _campaign_lock(root):
    for directory in (root, root / "classical"):
        if directory.is_symlink():
            raise ClassicalError("Classical artifact paths must be local directories.")
        directory.mkdir(exist_ok=True)
    lock = root / "classical" / ".lock"
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "r+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ClassicalError("Another classical campaign is already active.") from None
        yield


def _check_source(path):
    """Verify the imported package against the archived producer bytes."""
    package = Path(__file__).resolve().parent
    with tarfile.open(path, "r:") as archive:
        sources = {file.relative_to(package).as_posix() for file in package.rglob("*.py")}
        prefix = "src/disaster_triage/"
        archived = {
            member[len(prefix):] for member in archive.getnames() if member.startswith(prefix)
        }
        if sources != archived or any(
            (package / name).read_bytes() != archive.extractfile(prefix + name).read()
            for name in sources
        ):
            raise ClassicalError("The imported package differs from the producer source snapshot.")


def _worker(target, arguments, staging, reply, deadline):
    # A default-action alarm also stops a native solver if the supervisor exits.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            with threadpool_limits(limits=1):
                target(staging, *arguments)
        reply.send(("ok", None))
    except ClassicalError as error:
        reply.send(("failed", str(error)))
    except Exception as error:
        reply.send(("failed", f"Classical worker failed ({type(error).__name__})."))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        reply.close()


def _supervise(target, arguments, staging, attempt, deadline):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    worker = context.Process(target=_worker, args=(target, arguments, staging, send, deadline))
    started = False
    try:
        worker.start()
        started = True
        send.close()
        while worker.is_alive():
            remaining = min(deadline - time.monotonic(),
                            (datetime.fromisoformat(CUTOFF) - utc_now()).total_seconds())
            if remaining <= 0:
                raise DeadlineReached()
            worker.join(min(HEARTBEAT_SECONDS, remaining))
            if worker.is_alive():
                attempt.update()
        if worker.exitcode == -signal.SIGALRM or time.monotonic() >= deadline:
            raise DeadlineReached()
        if worker.exitcode != 0 or not receive.poll():
            raise ClassicalError("The classical worker exited before producing verified output.")
        try:
            status, detail = receive.recv()
        except EOFError:
            raise ClassicalError("The classical worker stopped before returning its status.") from None
        if status != "ok":
            raise ClassicalError(detail)
    finally:
        send.close()
        receive.close()
        if started:
            if worker.is_alive():
                worker.kill()
            worker.join()
            worker.close()


def _hash_artifacts(directory, filenames):
    if directory.is_symlink() or not directory.is_dir():
        raise ClassicalError("Invalid classical stage directory.")
    expected = set(filenames)
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ClassicalError("Classical artifacts cannot be symbolic links.")
        if path.is_file() and path != directory / "stage.json":
            actual.add(path.relative_to(directory).as_posix())
    if actual != expected:
        raise ClassicalError("Unexpected or missing classical stage files.")
    return {name: sha256_file(directory / name) for name in filenames}


def _fit_features(staging, inputs):
    vectorizer, train, selection = fit_features(inputs.train_text, inputs.selection_text)
    joblib.dump(vectorizer, staging / "vectorizer.joblib")
    save_npz(staging / "train.npz", train)
    save_npz(staging / "selection.npz", selection)


def _fit_and_score(staging, inputs, campaign, key, config):
    family = config["model_family"]
    converged = {"n_iter": None, "majority_label": None}
    if key == "majority":
        counts = training_counts(inputs.train_labels)
        predictions = {"y_pred": majority_predictions(counts, len(inputs.selection_rows))}
        converged["majority_label"] = LABELS[counts.argmax()]
    else:
        model, iterations = fit_lr(
            CANDIDATES[key], config["runtime"]["scikit-learn"],
            load_npz(campaign / "features/train.npz"), inputs.train_labels,
        )
        logits = decision_scores(model, load_npz(campaign / "features/selection.npz"))
        predictions = {"logits": logits}
        converged["n_iter"] = iterations
        joblib.dump(model, staging / "classifier.joblib")

    config_sha = save_json(staging / "resolved-config.json", config)
    bundle = _bundle(staging, inputs, config, config_sha)
    rows = inputs.selection_rows
    arrays = {
        "row_id": np.array([row.row_id for row in rows]),
        "event_id": np.array([row.event_id for row in rows]),
        "y_true": np.array([row.y_true for row in rows], dtype=np.int64),
        **predictions,
    }
    write_bundle(bundle.directory, expected=bundle.expected, run_id=config["run_id"],
                 model_family=family, seed=None, arrays=arrays,
                 code_revision=config["source"]["code_revision"])
    save_json(staging / "convergence.json", converged)
    save_json(staging / "report.json", score_bundle(bundle))


def _bundle(directory, inputs, config, config_sha):
    lineage = inputs.lineage
    expected = BundleExpectation(inputs.mode, "model_selection", inputs.selection_rows, {
        "archive_sha256": lineage["archive_sha256"],
        "eligible_manifest_sha256": lineage["eligible_manifest_sha256"],
        "role_manifest_sha256": lineage["role_manifest_sha256"]["model_selection"],
        "config_sha256": config_sha, "code_sha256": config["source"]["code_sha256"],
    })
    return BundleInput(directory / "selection", expected, config["run_id"], config["model_family"], None)


def _replay(staging, inputs, campaign, configs, selection):
    vectorizer = joblib.load(campaign / "features/vectorizer.joblib")
    features = vectorizer.transform(inputs.selection_text)
    scores = {}
    for key, config in configs.items():
        directory = campaign / key
        stored_config = directory / "resolved-config.json"
        if stored_config.read_bytes() != json_bytes(config):
            raise ClassicalError("A replay candidate has incompatible resolved settings.")
        bundle = _bundle(directory, inputs, config, sha256_file(stored_config))
        _, arrays = read_bundle(bundle.directory, expected=bundle.expected)
        convergence = _read_json(directory / "convergence.json")
        if key == "majority":
            expected = majority_predictions(training_counts(inputs.train_labels), len(inputs.selection_rows))
            actual = arrays["y_pred"]
        else:
            fitted = joblib.load(directory / "classifier.joblib")
            if checked_iterations(fitted) != convergence["n_iter"]:
                raise ClassicalError("Reloaded LR convergence metadata disagrees.")
            expected = decision_scores(fitted, features)
            actual = arrays["logits"]
        if not np.array_equal(expected, actual):
            raise ClassicalError("Replayed predictions differ from the v1 bundle.")
        report = score_bundle(bundle)
        if json_bytes(report) != (directory / "report.json").read_bytes():
            raise ClassicalError("Replayed scores differ from the saved aggregate report.")
        scores[key] = report["pooled"]["macro_f1"]
    best, tied, winner = rank_lr({key: scores[key] for key in CANDIDATES})
    if (best, tied, winner) != (selection["best_lr_macro_f1"], selection["tied_lr"], selection["winner"]):
        raise ClassicalError("The replayed scores do not recover the saved LR winner.")
    save_json(staging / "replay.json", {
        "status": "verified", "predictions_replayed": len(scores), "winner": winner,
        "selection_sha256": sha256_file(campaign / "selection.json"),
        "macro_f1": scores,
    })


def _run_files(key):
    common = (*PREDICTION_FILES, "resolved-config.json", "convergence.json", "report.json")
    return common if key == "majority" else (*common, "classifier.joblib")


class Campaign:
    """Execute or verify the fixed grid under one budget and provenance record."""

    def __init__(self, artifact_root, campaign_id, inputs, runtime):
        identity(campaign_id)
        if len(campaign_id) > 70:
            raise ClassicalError("Campaign IDs are limited to 70 characters.")
        self.artifact_root = artifact_root
        self.campaign_id = campaign_id
        self.directory = artifact_root / "classical" / campaign_id
        if self.directory.is_symlink():
            raise ClassicalError("Classical campaign directories cannot be symbolic links.")
        self.inputs = inputs
        self.runtime = runtime
        self.counts = training_counts(inputs.train_labels)
        self.design = {
            "schema_version": 1, "campaign_id": campaign_id, "mode": inputs.mode,
            "lineage": inputs.lineage, "runtime": runtime, "labels": list(LABELS),
            "training_counts": self.counts.tolist(),
            "rows": {"train": len(inputs.train_labels), "model_selection": len(inputs.selection_rows)},
            "vectorizer": {**VECTORIZER, "dtype": "float64"},
            "grid": {key: candidate.parameters(runtime["scikit-learn"])
                     for key, candidate in CANDIDATES.items()},
        }

    def open(self, source_root):
        if source_root is not None:
            with tempfile.TemporaryDirectory(prefix=".source-", dir=self.directory.parent) as temporary:
                candidate = Path(temporary) / "campaign"
                candidate.mkdir(mode=0o700)
                source = source_snapshot(source_root, candidate / "source.tar")
                _check_source(candidate / "source.tar")
                if not self.directory.exists():
                    save_json(candidate / "campaign.json", {**self.design, "source": source})
                    candidate.rename(self.directory)
                saved = _read_json(self.directory / "campaign.json")["source"]
                if source["code_sha256"] != saved["code_sha256"]:
                    raise ClassicalError("Implementation source changed. Use a new campaign ID.")
        manifest = _read_json(self.directory / "campaign.json")
        self.source = manifest.pop("source")
        if json_bytes(manifest) != json_bytes(self.design):
            raise ClassicalError("Freeze, runtime, or campaign settings changed. Use a new campaign ID.")
        snapshot = self.directory / "source.tar"
        if snapshot.is_symlink() or sha256_file(snapshot) != self.source["code_sha256"]:
            raise ClassicalError("The campaign producer snapshot failed its digest check.")
        _check_source(snapshot)

    def _run_id(self, key):
        return f"classical-{self.campaign_id}-{key}"

    def _validate_stage(self, directory, key, filenames):
        record_path = directory / "stage.json"
        if record_path.is_symlink():
            raise ClassicalError("A stage record cannot be a symbolic link.")
        record = _read_json(record_path)
        run_id = self._run_id(key)
        attempt_id = identity(record["attempt_id"])
        attempt = _read_json(self.artifact_root / "runs" / run_id / "attempts" / attempt_id / "attempt.json")
        if (record["run_id"] != run_id or attempt["run_id"] != run_id
                or attempt["attempt_id"] != attempt_id or attempt["status"] != "completed"
                or "ended_at" not in attempt
                or attempt["metadata"]["code_sha256"] != self.source["code_sha256"]
                or attempt["metadata"]["data_sha256"] != self.inputs.lineage["eligible_manifest_sha256"]
                or attempt["artifact_sha256"].get("stage") != sha256_file(record_path)
                or attempt["metadata"].get("config_sha256") != record["files"].get("resolved-config.json")
                or record["files"] != _hash_artifacts(directory, filenames)):
            raise ClassicalError("A completed stage failed ledger or artifact verification.")
        return record

    def existing_stage(self, key, filenames):
        directory = self.directory / key
        if directory.exists() or directory.is_symlink():
            return self._validate_stage(directory, key, filenames)
        # Recover publication when the process ended after closing the attempt.
        for staging in sorted(self.directory.glob(f".{key}.*")):
            attempt_id = identity(staging.name.rsplit(".", 1)[-1])
            attempt_path = (self.artifact_root / "runs" / self._run_id(key)
                            / "attempts" / attempt_id / "attempt.json")
            if attempt_path.exists() and _read_json(attempt_path)["status"] == "completed":
                record = self._validate_stage(staging, key, filenames)
                staging.rename(directory)
                return record
        return None

    def stage(self, key, filenames, target, arguments, *, config=None):
        existing = self.existing_stage(key, filenames)
        if existing is not None:
            return existing
        now = utc_now()
        budget = guard_budget(self.artifact_root, now=now)
        if budget["unresolved_attempts"]:
            raise ClassicalError("Reconcile unresolved attempts before starting a classical stage.")
        cutoff_seconds = (datetime.fromisoformat(CUTOFF) - now).total_seconds()
        deadline = time.monotonic() + min(budget["remaining_job_hours"] * 3600, cutoff_seconds)
        run_id = self._run_id(key)
        metadata = {
            "data_sha256": self.inputs.lineage["eligible_manifest_sha256"], "seed": None,
            "device": "cpu", "versions": self.runtime,
            "code_sha256": self.source["code_sha256"], "code_revision": self.source["code_revision"],
            "settings": {"campaign_id": self.campaign_id, "stage": key},
        }
        if config is not None:
            metadata["config_sha256"] = sha256_bytes(json_bytes(config))
        attempt = Attempt(self.artifact_root, run_id, metadata=metadata)
        staging = self.directory / f".{key}.{attempt.attempt_id}"
        staging.mkdir(mode=0o700)
        completed = False
        try:
            with attempt:
                _supervise(target, arguments, staging, attempt, deadline)
                record = {
                    "run_id": run_id, "attempt_id": attempt.attempt_id,
                    "files": _hash_artifacts(staging, filenames),
                }
                if time.monotonic() >= deadline or utc_now() >= datetime.fromisoformat(CUTOFF):
                    raise DeadlineReached()
                attempt.update(artifact_sha256={"stage": save_json(staging / "stage.json", record)})
            completed = True
            staging.rename(self.directory / key)
        finally:
            if not completed:
                shutil.rmtree(staging, ignore_errors=True)
        return record

    def configs(self, vectorizer_sha256):
        output = {}
        for key in RUNS:
            candidate = CANDIDATES.get(key)
            output[key] = {
                **self.design, "source": self.source, "run_id": self._run_id(key), "seed": None,
                "model_family": "tfidf_lr" if candidate else "majority",
                "estimator": self.design["grid"][key] if candidate else None,
                "effective_training_weights": effective_weights(self.counts, candidate.weight) if candidate else None,
                "vectorizer_sha256": vectorizer_sha256 if candidate else None,
            }
        return output

    def selection(self, stages, *, publish):
        comparison = []
        scores = {}
        for key, record in stages.items():
            report = _read_json(self.directory / key / "report.json")
            pooled = report["pooled"]
            scores[key] = pooled["macro_f1"]
            comparison.append({
                "candidate": key, "run_id": record["run_id"], "attempt_id": record["attempt_id"],
                "pooled_macro_f1": scores[key],
                "requests": pooled["classes"]["requests_or_urgent_needs"],
                "missing_or_found_people": pooled["classes"]["missing_or_found_people"],
                "report": f"{key}/report.json", "report_sha256": record["files"]["report.json"],
                "config_sha256": record["files"]["resolved-config.json"],
                "bundle_sha256": record["files"]["selection/predictions.npz"],
                "classifier_sha256": record["files"].get("classifier.joblib"),
            })
        best, tied, winner = rank_lr({key: scores[key] for key in CANDIDATES})
        result = {
            "schema_version": 1, "campaign_id": self.campaign_id, "role": "model_selection",
            "lineage": self.inputs.lineage, "comparison": comparison,
            "tie_tolerance": TIE_TOLERANCE, "best_lr_macro_f1": best, "tied_lr": tied,
            "winner": winner, "winner_macro_f1": scores[winner],
        }
        path = self.directory / "selection.json"
        if publish and not path.exists():
            save_json(path, result)
        elif path.is_symlink() or path.read_bytes() != json_bytes(result):
            raise ClassicalError("The saved selection record differs from the verified reports.")
        return result

    def run(self):
        features = self.stage("features", FEATURE_FILES, _fit_features, (self.inputs,))
        configs = self.configs(features["files"]["vectorizer.joblib"])
        stages = {}
        for key, config in configs.items():
            stages[key] = self.stage(
                key, _run_files(key), _fit_and_score,
                (self.inputs, self.directory, key, config), config=config,
            )
        result = self.selection(stages, publish=True)
        return {
            "status": "completed", "campaign_id": self.campaign_id, "winner": result["winner"],
            "winner_macro_f1": result["winner_macro_f1"],
            "selection_sha256": sha256_file(self.directory / "selection.json"),
        }

    def verify(self):
        features = self.existing_stage("features", FEATURE_FILES)
        stages = {key: self.existing_stage(key, _run_files(key)) for key in RUNS}
        if features is None or any(record is None for record in stages.values()):
            raise ClassicalError("Classical replay requires all seven candidates and shared features.")
        configs = self.configs(features["files"]["vectorizer.joblib"])
        result = self.selection(stages, publish=False)
        key = f"replay-{uuid.uuid4().hex[:12]}"
        stage = self.stage(key, ("replay.json",), _replay, (self.inputs, self.directory, configs, result))
        return {
            **_read_json(self.directory / key / "replay.json"), "campaign_id": self.campaign_id,
            "replay_attempt_id": stage["attempt_id"], "replay_sha256": stage["files"]["replay.json"],
        }


def run_classical(root, freeze, campaign_id, *, source_root=None, verify=False):
    root = Path(root).resolve()
    artifact_root = root / "artifacts"
    runtime = _runtime()
    with _campaign_lock(artifact_root):
        campaign = Campaign(artifact_root, campaign_id, load_inputs(root, freeze), runtime)
        campaign.open(None if verify else source_root or root)
        return campaign.verify() if verify else campaign.run()
