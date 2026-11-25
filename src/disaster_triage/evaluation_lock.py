"""Reopen accepted exact-byte data and model selection for Slice 8."""

from collections import Counter
import math
from pathlib import Path
from statistics import mean

from .accounting import identity
from .bundles import RoleRow, role_bytes
from .constants import ARCHIVE_SHA256, LABELS, MODEL_ID, MODEL_REVISION
from .data_policy import ARCHIVE_NAME, resolve_private
from .dataset import FrozenDataset, _json_load
from .provenance import json_bytes, sha256_bytes, sha256_file
from .transformer_data import ELIGIBLE_SHA256, FREEZE_ID, FREEZE_SHA256, is_digest

DEVELOPMENT = ("calibration", "policy")
EVALUATION_ROLES = (*DEVELOPMENT, "final_test")
RUN_SCOPE = {
    "majority": ("majority", None), "lr": ("tfidf_lr", None),
    "seed42": ("distilbert", 42), "seed43": ("distilbert", 43), "seed44": ("distilbert", 44),
}
TRAINED = tuple(key for key in RUN_SCOPE if key != "majority")
ROLE_COUNTS = {"calibration": 2129, "policy": 2121, "final_test": 7595}
FINAL_EVENTS = {"cyclone_idai_2019": 3933, "puebla_mexico_earthquake_2017": 2008,
                "hurricane_matthew_2016": 1654}


class EvaluationError(ValueError):
    """A diagnostic containing only campaign metadata."""


def digest(value):
    return sha256_bytes(json_bytes(value))


def private_at(root, name):
    part = Path(name)
    if part.is_absolute() or ".." in part.parts or not part.parts or part.parts[0] not in ("artifacts", "data"):
        raise EvaluationError("evaluation artifacts must use private paths inside the execution root")
    path = root / part
    if path.resolve() != path:
        raise EvaluationError("evaluation paths cannot contain symbolic links")
    return path


def checked_json(path, expected):
    content = path.read_bytes()
    if not is_digest(expected) or sha256_bytes(content) != expected:
        raise EvaluationError("required evaluation record does not match the independently retained digest")
    value = _json_load(content)
    if not isinstance(value, dict) or content != json_bytes(value):
        raise EvaluationError("evaluation record must be a canonical JSON object")
    return value


class SelectionLock:
    def __init__(self, root, campaign, lock_sha256, *, candidate=None):
        self.root = Path(root).resolve()
        self.artifacts = private_at(self.root, "artifacts")
        self.id = identity(campaign)
        if len(self.id) > 80:
            raise EvaluationError("evaluation campaign identity exceeds 80 characters")
        self.directory = private_at(self.root, f"artifacts/evaluation/{self.id}")
        self.lock_sha256 = lock_sha256
        lock_path = candidate if candidate is not None else self.directory / "selection-lock.json"
        self.lock = checked_json(private_at(self.root, lock_path.relative_to(self.root)), lock_sha256)
        self.mode = self.lock["mode"]
        self.runs = self.lock["runs"]
        if (self.lock["schema_version"] != 1 or self.lock["campaign"] != self.id
                or set(self.runs) != RUN_SCOPE.keys() or self.mode not in ("research", "synthetic")):
            raise EvaluationError("selection lock has the wrong campaign or predictor scope")
        for key, (family, seed) in RUN_SCOPE.items():
            run = self.runs[key]
            if (run["model_family"], run["seed"]) != (family, seed) or (
                    type(run["seed"]) is not type(seed)):
                raise EvaluationError("selection lock requires the declared families and unique seeds")
        self.frozen = FrozenDataset.from_verified_descriptor(
            self.root, self.lock["freeze"]["path"], digest=self.lock["freeze"]["sha256"], mode=self.mode
        )
        self.data = self.frozen.descriptor
        if self.mode == "research":
            self._verify_research_lock()
        self.scope = {"mode": self.mode, "archive_sha256": self.data["archive_sha256"],
                      "eligible_manifest_sha256": self.data["files"]["eligible-manifest.jsonl"]}
        self._rows = {}

    def checked_file(self, name, expected):
        path = private_at(self.root, name)
        if not is_digest(expected) or sha256_file(path) != expected:
            raise EvaluationError("locked file digest mismatch")
        return path

    def checked_json(self, name, expected):
        return checked_json(private_at(self.root, name), expected)

    def _verify_research_lock(self):
        if (self.data["freeze_id"], self.lock["freeze"]["sha256"], self.data["archive_sha256"],
            self.data["files"]["eligible-manifest.jsonl"]) != (
                FREEZE_ID, FREEZE_SHA256, ARCHIVE_SHA256, ELIGIBLE_SHA256
        ):
            raise EvaluationError("selection lock requires the accepted research freeze")
        if any(self.data["summary"]["role_counts"][role] != count for role, count in ROLE_COUNTS.items()):
            raise EvaluationError("evaluation role counts differ from the accepted freeze")
        self.checked_file(f"data/raw/{ARCHIVE_NAME}", ARCHIVE_SHA256)
        for name, sha256 in self.data["files"].items():
            # The holdout text file is checked by the authorized, single-read text loader.
            if name != "texts/final_test.jsonl":
                path = resolve_private(self.root, self.frozen.directory / name)
                self.checked_file(path.relative_to(self.root), sha256)
        self.verify_preserved()
        trainers = []
        for run in self.runs.values():
            parents = run["parents"]
            config = self.checked_json(*parents["training_config"])
            family = run["model_family"]
            wanted = ({"training_config", "checkpoint", "selection_config", "selection_report"}
                      if family == "distilbert" else {"training_config", "model", "vectorizer"}
                      if family == "tfidf_lr" else {"training_config"})
            if set(parents) != wanted or config["labels"] != list(LABELS):
                raise EvaluationError("model training provenance is incomplete")
            lineage = config["lineage"]
            if (lineage["freeze_id"], lineage["eligible_manifest_sha256"]) != (FREEZE_ID, ELIGIBLE_SHA256):
                raise EvaluationError("selected training configuration has incompatible dataset provenance")
            if family == "distilbert":
                if ((config["seed"], config["learning_rate"], run["epoch"], run["steps"],
                     config["scheduler_total_steps"], config["max_length"], config["precision"],
                     config["model_id"], config["model_revision"], config["tokenizer_revision"])
                        != (run["seed"], 2e-5, 3, 4722, 4722, 128, "fp32", MODEL_ID, MODEL_REVISION, MODEL_REVISION)
                        or self.checked_json(*parents["selection_config"]) != {
                            **config, "evaluated_epoch": 3, "checkpoint_sha256": parents["checkpoint"][1],
                        }):
                    raise EvaluationError("transformer config differs from the locked training recipe")
                report = self.checked_json(*parents["selection_report"])
                if report["pooled"]["macro_f1"] != run["selection_macro_f1"]:
                    raise EvaluationError("selected transformer score differs from accepted report")
                trainers.append(run["selection_macro_f1"])
            elif config["model_family"] != family:
                raise EvaluationError("classical family differs from the selected configuration")
            if family == "tfidf_lr" and (
                config["estimator"]["C"] != 10 or config["estimator"]["class_weight"] != "balanced"
                or config["vectorizer_sha256"] != parents["vectorizer"][1]
            ):
                raise EvaluationError("logistic regression differs from the locked balanced C=10 recipe")
            for name in ("checkpoint", "model", "vectorizer"):
                if name in parents:
                    self.checked_file(*parents[name])
        counts = self.data["summary"]["training_counts"]
        label = max(range(len(LABELS)), key=counts.__getitem__)
        transformer = mean(trainers)
        lr = self.runs["lr"]["selection_macro_f1"]
        margin = transformer - lr
        primary = "distilbert" if margin > .001 else "tfidf_lr"
        if (not math.isfinite(margin) or self.lock["majority_label"] != label
                or self.lock["selection"] != {"primary": primary, "tie_tolerance": .001,
                    "distilbert_mean": transformer, "lr": lr, "margin": margin}):
            raise EvaluationError("selection lock does not match the frozen label or family selection rule")
        review = self.lock["review"]
        if len(set(review["reviewers"])) != 2 or not review["source"] or not review["date"]:
            raise EvaluationError("joint review metadata is required")

    def verify_preserved(self):
        for name, expected in self.lock["preserve"].items():
            self.checked_file(name, expected)

    def rows(self, role):
        if role not in EVALUATION_ROLES:
            raise EvaluationError("evaluation role is outside the locked scope")
        if role not in self._rows:
            name = f"roles/{role}.jsonl"
            path = resolve_private(self.root, self.frozen.directory / name)
            content = path.read_bytes()
            if sha256_bytes(content) != self.data["files"][name]:
                raise EvaluationError("ordered role digest mismatch")
            rows = tuple(RoleRow(**_json_load(line)) for line in content.splitlines())
            if role_bytes(rows) != content or not rows or len(rows) != self.data["summary"]["role_counts"][role]:
                raise EvaluationError("ordered role bytes or count mismatch")
            if self.mode == "research" and role == "final_test":
                if (Counter(row.event_id for row in rows) != FINAL_EVENTS
                        or sum(row.y_true == LABELS.index("requests_or_urgent_needs") for row in rows) != 257
                        or Counter(row.event_id for row in rows if row.y_true == LABELS.index(
                            "missing_or_found_people")) != {"cyclone_idai_2019": 19}):
                    raise EvaluationError("held-out event and rare-class supports disagree with the accepted freeze")
            self._rows[role] = rows
        return self._rows[role]

    def role_scope(self, role):
        return {**self.scope, "role": role, "role_manifest_sha256": self.data["files"][f"roles/{role}.jsonl"]}
