"""Prepare immutable review policies, then run one explicitly authorized holdout."""

from contextlib import contextmanager
from pathlib import Path
import signal
import tempfile
import time

import numpy as np

from .accounting import Attempt, guard_evaluation_budget, identity
from .bundles import BundleExpectation, RoleRow, read_bundle, role_bytes, write_bundle
from .classical_run import _campaign_lock, _check_source
from .evaluation import (BundleInput, SeedCohort, _validate_fitted_policy,
                         fit_review_policy, score_bundle, summarize_seeds)
from .evaluation_access import _authorize_final
from .evaluation_lock import (DEVELOPMENT, EVALUATION_ROLES, TRAINED, RUN_SCOPE, EvaluationError,
                              SelectionLock, checked_json, digest, private_at)
from .evaluation_models import BATCH, asset_inventory, open_predictor, ordered_text
from .provenance import save_json, sha256_bytes, sha256_file, source_snapshot

MARGIN = 1.2
PROBE_ROWS = 256
PROBE_LIMIT_SECONDS = 900


class EvaluationDeadline(KeyboardInterrupt):
    """Inference must finish within its admitted cumulative time."""


@contextmanager
def deadline(seconds):
    until = time.monotonic() + seconds

    def check():
        if time.monotonic() >= until:
            raise EvaluationDeadline("evaluation reached its cumulative time limit")

    def expire(*_):
        raise EvaluationDeadline("evaluation reached its cumulative time limit")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield check
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def save_once(path, record):
    if path.exists():
        raise EvaluationError("immutable evaluation record already exists")
    return save_json(path, record)


class EvaluationCampaign(SelectionLock):
    def __init__(self, root, campaign, lock_sha256):
        super().__init__(root, campaign, lock_sha256)
        self.probe_path = self.path("probe.json")
        self.source_path = self.path("source.tar")
        self.readiness_path = self.path("readiness.json")
        self.manifest_path = self.path("manifest.json")

    def path(self, name):
        return private_at(self.root, (self.directory / name).relative_to(self.root))

    def budget(self, forecast):
        return guard_evaluation_budget(self.artifacts, forecast)

    def attempt(self, stage, backend, admitted):
        return Attempt(self.artifacts, f"evaluation-{self.id}-{stage}", metadata={
            "data_sha256": self.scope["eligible_manifest_sha256"], "device": backend,
            "settings": {"campaign": self.id, "stage": stage, "selection_lock_sha256": self.lock_sha256,
                         "admitted_active_seconds": admitted},
        })

    def check_attempt(self, stage, attempt_id, artifact, sha256):
        attempt_id = identity(attempt_id)
        run_id = f"evaluation-{self.id}-{stage}"
        path = private_at(self.root, f"artifacts/runs/{run_id}/attempts/{attempt_id}/attempt.json")
        from .dataset import _json_load

        record = _json_load(path.read_bytes())
        if (record["attempt_id"] != attempt_id or record["run_id"] != run_id
                or record["metadata"]["settings"]["selection_lock_sha256"] != self.lock_sha256
                or record["metadata"]["data_sha256"] != self.scope["eligible_manifest_sha256"]
                or record["artifact_sha256"].get(artifact) != sha256
                or record["status"] not in ("completed", "failed", "interrupted")
                or "ended_at" not in record):
            raise EvaluationError("published evaluation evidence lacks a matching terminal attempt")

    def runtime(self, backend):
        if self.mode == "synthetic":
            return None, {"backend": "cpu", "profile": "synthetic"}
        return asset_inventory(self.root, backend)

    def predictor(self, key, model_directory, backend):
        if self.mode == "synthetic":
            def invented(texts, check):
                check()
                if key == "majority":
                    return np.full(len(texts), self.lock["majority_label"], dtype=np.int64)
                scores = np.arange(10, dtype=np.float32)
                return np.stack([np.roll(scores, len(text) % 10) for text in texts])
            return invented
        return open_predictor(self, self.runs[key], model_directory, backend)

    def forecasts(self, measured, roles):
        counts = self.data["summary"]["role_counts"]
        return MARGIN * sum(
            measured[key]["startup_seconds"] + sum(
                max(measured[key]["prediction_seconds"], 0.001) * counts[role] / PROBE_ROWS
                + max(measured[key]["scoring_seconds"], 0.01) * counts[role] / PROBE_ROWS
                for role in selected_roles
            ) + 1
            for key, selected_roles in roles.items()
        )

    def probe(self, source_root):
        if self.probe_path.exists():
            record, _ = self.verified_probe()
            return record
        self.directory.mkdir(parents=True, exist_ok=True)
        source = source_snapshot(source_root, self.source_path)
        _check_source(self.source_path)
        backends = ("cpu",) if self.mode == "synthetic" else ("cpu", "mps")
        for backend in backends:
            try:
                limit = min(self.budget(1), PROBE_LIMIT_SECONDS)
                with self.attempt("probe", backend, limit) as attempt, deadline(limit) as check:
                    started = time.monotonic()
                    model_directory, runtime = self.runtime(backend)
                    startup = time.monotonic() - started
                    measured = self.measure(source, model_directory, backend, check)
                    coverage = {key: EVALUATION_ROLES for key in TRAINED}
                    coverage["majority"] = ("final_test",)
                    forecast = self.forecasts(measured, coverage) + MARGIN * startup
                    check()
                    available = self.budget_before_probe(attempt, forecast)
                    record = {
                        "schema_version": 1, "campaign": self.id, "selection_lock_sha256": self.lock_sha256,
                        "backend": backend, "source": source, "runtime": runtime,
                        "probe_rows_per_predictor": PROBE_ROWS, "inference_batch": BATCH,
                        "max_length": 128, "precision": "fp32", "margin": MARGIN,
                        "setup_seconds": startup, "measured": measured,
                        "rows_per_trained_run": sum(self.data["summary"]["role_counts"][role]
                                                    for role in EVALUATION_ROLES),
                        "forecast_seconds": forecast, "available_seconds": available,
                        "attempt": attempt.attempt_id,
                    }
                    attempt.update(artifact_sha256={"probe": digest(record)})
                    save_once(self.probe_path, record)
                return record
            except (RuntimeError, ValueError, EvaluationDeadline) as error:
                if backend == backends[-1]:
                    raise EvaluationError("local inference probe failed or its forecast exceeds the available budget") from error
        raise AssertionError("no local evaluation backend was selected")

    def budget_before_probe(self, attempt, forecast):
        # The ongoing probe is the sole running attempt under the shared operation lock.
        from .accounting import budget_summary

        summary = budget_summary(self.artifacts)
        available = summary["remaining_job_hours"] * 3600
        if summary["unresolved_attempts"] != [attempt.attempt_id] or forecast >= available:
            raise EvaluationError("local evaluation forecast does not fit the reconciled ledger")
        return available

    def measure(self, source, model_directory, backend, check):
        # Half the invented inputs reach or exceed the truncation boundary.
        texts = [f"invented emergency exercise {index}" if index % 2
                 else ("invented planning exercise " * 80) + str(index)
                 for index in range(PROBE_ROWS)]
        values = {}
        with tempfile.TemporaryDirectory(prefix=".probe-", dir=self.directory) as tmp:
            for key in RUN_SCOPE:
                family, seed = RUN_SCOPE[key]
                start = time.monotonic()
                predictor = self.predictor(key, model_directory, backend)
                loaded = time.monotonic()
                prediction = predictor(texts, check)
                inferred = time.monotonic()
                views = []
                for role in (DEVELOPMENT if key != "majority" else ("policy",)):
                    rows = tuple(RoleRow(f"{role}-{i}", "synthetic_local", i % 10) for i in range(PROBE_ROWS))
                    config_sha = digest({"predictor": key})
                    expected = BundleExpectation("synthetic", role, rows, {
                        "archive_sha256": digest("probe archive"),
                        "eligible_manifest_sha256": digest("probe eligible"),
                        "role_manifest_sha256": sha256_bytes(role_bytes(rows)),
                        "code_sha256": source["code_sha256"], "config_sha256": config_sha,
                    })
                    view = BundleInput(Path(tmp) / key / role, expected, f"invented-{key}", family, seed)
                    self.write_predictions(view, prediction, source["code_revision"])
                    views.append(view)
                if key == "majority":
                    score_bundle(views[0])
                else:
                    fitted = fit_review_policy(*views)
                    score_bundle(views[-1], fitted_policy=fitted, fitted_policy_sha256=digest(fitted))
                values[key] = {"startup_seconds": loaded - start, "prediction_seconds": inferred - loaded,
                               "scoring_seconds": time.monotonic() - inferred}
                del predictor
        return values

    def verified_probe(self):
        # The attempt independently binds the initially published runtime and source.
        sha256 = sha256_file(self.probe_path)
        probe = checked_json(self.probe_path, sha256)
        self.check_attempt("probe", probe["attempt"], "probe", sha256)
        if (probe["selection_lock_sha256"] != self.lock_sha256 or probe["campaign"] != self.id
                or sha256_file(self.source_path) != probe["source"]["code_sha256"]):
            raise EvaluationError("probe/source does not match the locked campaign")
        _check_source(self.source_path)
        return probe, sha256

    def inference_config(self, key, probe):
        return {
            "schema_version": 1, "selection_lock_sha256": self.lock_sha256, "predictor": key,
            "parent_provenance": self.runs[key], "runtime": probe["runtime"],
            "code_sha256": probe["source"]["code_sha256"], "batch": BATCH,
            "max_length": 128, "precision": "fp32", "labels": self.data["labels"],
        }

    def view(self, key, role, directory, probe):
        family, seed = RUN_SCOPE[key]
        config_sha256 = digest(self.inference_config(key, probe))
        scope = self.role_scope(role)
        expected = BundleExpectation(self.mode, role, self.rows(role), {
            **{name: value for name, value in scope.items() if name.endswith("_sha256")},
            "config_sha256": config_sha256, "code_sha256": probe["source"]["code_sha256"],
        })
        return BundleInput(directory, expected, f"{self.id}-{key}", family, seed)

    @staticmethod
    def write_predictions(view, predictions, code_revision, access=None):
        rows = view.expected.rows
        kind = "y_pred" if view.model_family == "majority" else "logits"
        write_bundle(view.directory, expected=view.expected, run_id=view.run_id, model_family=view.model_family,
                     seed=view.seed, code_revision=code_revision, final_access=access, arrays={
                         "row_id": np.array([row.row_id for row in rows]),
                         "event_id": np.array([row.event_id for row in rows]),
                         "y_true": np.array([row.y_true for row in rows], dtype=np.int64),
                         kind: predictions,
                     })

    def publish(self, stage, key, role, target, probe, predictions, attempt, check, *, access=None, policy=None,
                readiness_sha256=None):
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=target.parent) as tmp:
            staging = Path(tmp)
            view = self.view(key, role, staging / "bundle", probe)
            self.write_predictions(view, predictions, probe["source"]["code_revision"], access)
            if stage == "final":
                report = self.score(view, access, policy)
                report_sha256 = save_json(staging / "report.json", report)
            else:
                report_sha256 = None
            manifest_sha256 = sha256_file(view.directory / "manifest.json")
            record = {
                "schema_version": 1, "selection_lock_sha256": self.lock_sha256, "predictor": key,
                "role": role, "attempt": attempt.attempt_id, "readiness_sha256": readiness_sha256,
                "bundle_manifest_sha256": manifest_sha256, "report_sha256": report_sha256,
            }
            receipt_sha256 = save_json(staging / "receipt.json", record)
            check()
            # Attest before the atomic rename. An interrupted unrenamed bundle is never a completed result.
            attempt.update(artifact_sha256={f"{key}-{role}": receipt_sha256})
            check()
            if target.exists():
                raise EvaluationError("completed evaluation bundle cannot be overwritten")
            staging.rename(target)
        return record

    def published(self, stage, key, role, target, probe, *, access=None, readiness_sha256=None):
        receipt_path = self.path(target.relative_to(self.directory) / "receipt.json")
        receipt_sha256 = sha256_file(receipt_path)
        record = checked_json(receipt_path, receipt_sha256)
        self.check_attempt(stage, record["attempt"], f"{key}-{role}", receipt_sha256)
        view = self.view(key, role, self.path(target.relative_to(self.directory) / "bundle"), probe)
        manifest_sha256 = sha256_file(self.path(view.directory.relative_to(self.directory) / "manifest.json"))
        if (record["selection_lock_sha256"] != self.lock_sha256 or record["predictor"] != key
                or record["role"] != role or record["bundle_manifest_sha256"] != manifest_sha256
                or record["readiness_sha256"] != readiness_sha256):
            raise EvaluationError("bundle receipt differs from locked evaluation provenance")
        self.path(view.directory.relative_to(self.directory) / "predictions.npz")
        manifest, _ = read_bundle(view.directory, expected=view.expected, final_access=access)
        if (manifest["run_id"], manifest["model_family"], manifest["seed"]) != (
            view.run_id, view.model_family, view.seed
        ):
            raise EvaluationError("saved bundle does not identify the locked predictor")
        return view, record, receipt_sha256, manifest

    def development_target(self, key, role):
        return self.path(f"development/{key}/{role}")

    def final_target(self, key):
        return self.path(f"final/{key}")

    def prepare(self):
        probe, probe_sha256 = self.verified_probe()
        if self.readiness_path.exists():
            sha256 = sha256_file(self.readiness_path)
            self.verified_readiness(probe, probe_sha256, sha256)
            return {"campaign": self.id, "readiness_sha256": sha256}
        pending = {key: [role for role in DEVELOPMENT if not self.development_target(key, role).exists()]
                   for key in TRAINED}
        # Verify completed partial work before consuming any additional prediction allowance.
        inventory = {}
        for key in TRAINED:
            inventory[key] = {}
            for role in DEVELOPMENT:
                if role not in pending[key]:
                    _, record, receipt_sha, manifest = self.published(
                        "prepare", key, role, self.development_target(key, role), probe
                    )
                    inventory[key][role] = self.bundle_ref(record, receipt_sha, manifest["payload_sha256"])
        forecast = self.forecasts(probe["measured"], pending) + MARGIN * probe["setup_seconds"]
        available = self.budget(forecast)
        backend = probe["backend"]
        policies = {}
        with self.attempt("prepare", backend, available) as attempt, deadline(available) as check:
            model_directory, runtime = self.runtime(backend)
            if runtime != probe["runtime"]:
                raise EvaluationError("runtime or cached assets changed after the synthetic probe")
            texts = {role: self.frozen.load_role(role) for role in DEVELOPMENT if any(
                role in roles for roles in pending.values()
            )}
            for key in TRAINED:
                check()
                if pending[key]:
                    predictor = self.predictor(key, model_directory, backend)
                    for role in pending[key]:
                        target = self.development_target(key, role)
                        text = ordered_text(texts[role], self.rows(role), role, self.runs[key]["model_family"])
                        predictions = predictor(text, check)
                        record = self.publish("prepare", key, role, target, probe, predictions, attempt, check)
                        view = self.view(key, role, target / "bundle", probe)
                        manifest, _ = read_bundle(view.directory, expected=view.expected)
                        inventory[key][role] = self.bundle_ref(record, digest(record), manifest["payload_sha256"])
                    del predictor
                views = [self.view(key, role, self.development_target(key, role) / "bundle", probe)
                         for role in DEVELOPMENT]
                record = fit_review_policy(*views)
                path = self.path(f"policies/{key}.json")
                if path.exists():
                    checked_json(path, digest(record))
                else:
                    save_once(path, record)
                policies[key] = digest(record)
            configs = {}
            for key in RUN_SCOPE:
                config = self.inference_config(key, probe)
                path = self.path(f"inference-configs/{key}.json")
                configs[key] = digest(config)
                if path.exists():
                    checked_json(path, configs[key])
                else:
                    save_once(path, config)
            record = {"schema_version": 1, "campaign": self.id, "selection_lock_sha256": self.lock_sha256,
                      "probe_sha256": probe_sha256, "development": inventory, "policies": policies,
                      "configs": configs, "attempt": attempt.attempt_id}
            check()
            attempt.update(artifact_sha256={"readiness": digest(record)})
            save_once(self.readiness_path, record)
        return {"campaign": self.id, "readiness_sha256": digest(record)}

    @staticmethod
    def bundle_ref(receipt, receipt_sha256, payload_sha256):
        return {"receipt_sha256": receipt_sha256, "manifest_sha256": receipt["bundle_manifest_sha256"],
                "payload_sha256": payload_sha256}

    def verified_readiness(self, probe, probe_sha256, readiness_sha256):
        record = checked_json(self.readiness_path, readiness_sha256)
        self.check_attempt("prepare", record["attempt"], "readiness", readiness_sha256)
        if ((record["schema_version"], record["campaign"], record["selection_lock_sha256"],
             record["probe_sha256"]) != (1, self.id, self.lock_sha256, probe_sha256)
                or set(record["policies"]) != set(TRAINED) or set(record["development"]) != set(TRAINED)
                or set(record["configs"]) != RUN_SCOPE.keys()):
            raise EvaluationError("readiness receipt does not match the campaign, runtime and trained predictors")
        for key, sha256 in record["configs"].items():
            if sha256 != digest(self.inference_config(key, probe)):
                raise EvaluationError("readiness inference configuration differs from the locked predictor")
            checked_json(self.path(f"inference-configs/{key}.json"), sha256)
        policies = {}
        for key in TRAINED:
            bindings = {}
            if set(record["development"][key]) != set(DEVELOPMENT):
                raise EvaluationError("readiness requires both frozen development bundles per predictor")
            for role in DEVELOPMENT:
                _, reference, receipt_sha, manifest = self.published(
                    "prepare", key, role, self.development_target(key, role), probe
                )
                if record["development"][key][role] != self.bundle_ref(reference, receipt_sha, manifest["payload_sha256"]):
                    raise EvaluationError("readiness does not bind the verified development bundles")
                bindings[role] = {name: manifest[name] for name in ("role_manifest_sha256", "payload_sha256")}
            expected = record["policies"][key]
            policy = checked_json(self.path(f"policies/{key}.json"), expected)
            if policy["inputs"] != bindings:
                raise EvaluationError("saved policy does not bind both verified development inputs")
            _validate_fitted_policy(policy, expected, {"mode": self.mode, **manifest})
            policies[key] = (policy, expected)
        return policies

    def final_context(self, readiness_sha256):
        probe, probe_sha256 = self.verified_probe()
        policies = self.verified_readiness(probe, probe_sha256, readiness_sha256)
        self.rows("final_test")
        scope = self.role_scope("final_test")
        runs, digests = [], {}
        for key in RUN_SCOPE:
            view = self.view(key, "final_test", self.final_target(key) / "bundle", probe)
            runs.append({"run_id": view.run_id, "model_family": view.model_family, "seed": view.seed,
                         "config_sha256": view.expected.digests["config_sha256"],
                         "code_sha256": view.expected.digests["code_sha256"]})
            digests[view.run_id] = policies[key][1] if key in policies else None
        return probe, policies, _authorize_final(scope=scope, runs=runs, policies=digests)

    @staticmethod
    def score(view, access, policy):
        record, sha256 = policy if policy is not None else (None, None)
        return score_bundle(view, fitted_policy=record, fitted_policy_sha256=sha256, final_access=access)

    def verify_final_result(self, key, probe, access, policies, readiness_sha256):
        target = self.final_target(key)
        view, receipt, sha256, manifest = self.published(
            "final", key, "final_test", target, probe, access=access, readiness_sha256=readiness_sha256
        )
        path = self.path(target.relative_to(self.directory) / "report.json")
        report = checked_json(path, receipt["report_sha256"])
        if report != self.score(view, access, policies.get(key)):
            raise EvaluationError("saved final aggregate differs from bundle replay")
        return report, {"report_sha256": receipt["report_sha256"],
                        **self.bundle_ref(receipt, sha256, manifest["payload_sha256"])}

    def cohort(self, reports, access):
        cohort = SeedCohort(
            digest({"selection_lock_sha256": self.lock_sha256, "epoch": 3}), 3,
            {report["bundle"]["seed"]: report["bundle"]["config_sha256"] for report in reports},
        )
        return summarize_seeds(reports, cohort=cohort, final_access=access)

    def refuse_missing_final_results(self, pending):
        run_dir = self.artifacts / "runs" / f"evaluation-{self.id}-final" / "attempts"
        from .dataset import _json_load

        for path in run_dir.glob("*/attempt.json"):
            record = _json_load(path.read_bytes())
            if (record["status"] == "completed"
                    and record["metadata"]["settings"]["selection_lock_sha256"] == self.lock_sha256
                    and any(f"{key}-final_test" in record["artifact_sha256"] for key in pending)):
                raise EvaluationError("completed final result is missing, preserve the evidence and review the defect")

    def final(self, readiness_sha256, *, verify=False):
        # The independently supplied readiness digest is checked before final predictor payloads or text.
        probe, policies, access = self.final_context(readiness_sha256)
        if verify and not self.manifest_path.exists():
            raise EvaluationError("final evaluation has no published manifest")
        completed = [key for key in RUN_SCOPE if self.final_target(key).exists()]
        pending = [key for key in RUN_SCOPE if key not in completed]
        if self.manifest_path.exists() and pending:
            raise EvaluationError("published final manifest is missing completed predictor output")
        if pending:
            self.refuse_missing_final_results(pending)
        forecast = (self.forecasts(probe["measured"], {key: ("final_test",) for key in pending})
                    + MARGIN * probe["setup_seconds"])
        available = self.budget(max(forecast, 0.01)) if pending else None
        reports, outputs = {}, {}
        for key in completed:
            reports[key], outputs[key] = self.verify_final_result(key, probe, access, policies, readiness_sha256)
        if pending:
            backend = probe["backend"]
            with self.attempt("final", backend, available) as attempt, deadline(available) as check:
                model_directory, runtime = self.runtime(backend)
                if runtime != probe["runtime"]:
                    raise EvaluationError("runtime or cached assets changed after the synthetic probe")
                records = self.frozen.load_final_role(access)
                rows = self.rows("final_test")
                for key in pending:
                    check()
                    family = self.runs[key]["model_family"]
                    texts = ordered_text(records, rows, "final_test", family)
                    predict = self.predictor(key, model_directory, backend)
                    predictions = predict(texts, check)
                    del predict
                    self.publish("final", key, "final_test", self.final_target(key), probe, predictions,
                                 attempt, check, access=access, policy=policies.get(key),
                                 readiness_sha256=readiness_sha256)
            for key in pending:
                reports[key], outputs[key] = self.verify_final_result(key, probe, access, policies, readiness_sha256)
        cohort = self.cohort([reports[key] for key in ("seed42", "seed43", "seed44")], access)
        cohort_path = self.path("transformer-cohort.json")
        if cohort_path.exists() or self.manifest_path.exists() or verify:
            checked_json(cohort_path, digest(cohort))
        else:
            save_once(cohort_path, cohort)
        manifest = {
            "schema_version": 1, "campaign": self.id, "mode": self.mode,
            "primary": self.lock["selection"]["primary"], "selection_lock_sha256": self.lock_sha256,
            "readiness_sha256": readiness_sha256, "role": self.role_scope("final_test"),
            "cohort_sha256": digest(cohort), "outputs": outputs,
        }
        if self.manifest_path.exists():
            checked_json(self.manifest_path, digest(manifest))
        else:
            save_once(self.manifest_path, manifest)
        return {"campaign": self.id, "manifest_sha256": digest(manifest),
                "pooled_macro_f1": {key: report["pooled"]["macro_f1"] for key, report in reports.items()},
                "transformer_cohort": cohort["pooled"]}


def command(args):
    root = Path(args.root).resolve()
    artifacts = private_at(root, "artifacts")
    with _campaign_lock(artifacts):
        campaign = EvaluationCampaign(root, args.campaign_id, args.selection_lock_sha256)
        if args.evaluation_action == "probe":
            return campaign.probe(args.source_root or root)
        if args.evaluation_action == "prepare":
            return campaign.prepare()
        return campaign.final(args.readiness_sha256, verify=args.evaluation_action == "verify")
