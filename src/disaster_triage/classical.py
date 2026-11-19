"""Train-only classical features, reference predictions and the declared LR grid."""
from dataclasses import dataclass
import json
import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .bundles import RoleRow, role_bytes
from .constants import CLASSICAL_SETTINGS, LABELS
from .data_policy import resolve_private
from .dataset import FrozenDataset
from .provenance import sha256_bytes

VECTORIZER = {
    **{key: CLASSICAL_SETTINGS[key] for key in (
        "min_df", "max_df", "sublinear_tf", "max_features",
    )},
    "ngram_range": tuple(CLASSICAL_SETTINGS["ngram_range"]),
    "analyzer": "word", "lowercase": False, "norm": "l2", "smooth_idf": True,
    "use_idf": True, "token_pattern": r"(?u)\b\w\w+\b",
}
TIE_TOLERANCE = 0.001


class ClassicalError(RuntimeError):
    """A fixed operator diagnostic that is safe to print."""


@dataclass(frozen=True)
class Candidate:
    name: str
    c: float
    weight: str | None

    def parameters(self, sklearn_version):
        regularizer = {"penalty": "l2"} if sklearn_version == "1.6.1" else {"l1_ratio": 0.0}
        return {
            **regularizer, "solver": CLASSICAL_SETTINGS["solver"],
            "max_iter": CLASSICAL_SETTINGS["max_iter"], "tol": 1e-4,
            "fit_intercept": True, "warm_start": False, "C": self.c,
            "class_weight": self.weight,
        }


CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        Candidate(f"lr-{c:g}-{'balanced' if weight else 'unweighted'}", c, weight)
        for c in CLASSICAL_SETTINGS["C"] for weight in CLASSICAL_SETTINGS["class_weight"]
    )
}


@dataclass(frozen=True)
class Inputs:
    mode: str
    lineage: dict
    train_text: tuple[str, ...]
    train_labels: np.ndarray
    selection_text: tuple[str, ...]
    selection_rows: tuple[RoleRow, ...]


def aligned_text(records, ordered, role):
    """Join the text view to the verified role manifest by row identity."""
    records_by_id = {record["row_id"]: record for record in records}
    if (len(records_by_id) != len(records)
            or records_by_id.keys() != {row.row_id for row in ordered}
            or len(ordered) != len(records)):
        raise ClassicalError("Text and role manifests have different row membership.")
    texts = []
    for row in ordered:
        record = records_by_id[row.row_id]
        if (record["event_id"], record["y_true"], record["role"]) != (row.event_id, row.y_true, role):
            raise ClassicalError("Text and role manifests disagree on event, label, or role.")
        texts.append(record["classical_text"])
    return tuple(texts)


def load_inputs(root, freeze):
    frozen = FrozenDataset(root, freeze)
    verified = frozen.verification
    roles = {}
    texts = {}
    for role in ("train", "model_selection"):
        content = resolve_private(root, frozen.directory / f"roles/{role}.jsonl").read_bytes()
        rows = tuple(RoleRow(**json.loads(line)) for line in content.splitlines())
        if (sha256_bytes(content) != verified["role_manifest_sha256"][role]
                or role_bytes(rows) != content or not rows
                or len(rows) != verified["role_counts"][role]):
            raise ClassicalError("The frozen role manifest failed its alignment checks.")
        roles[role] = rows
        texts[role] = aligned_text(frozen.load_role(role), rows, role)
    if ({row.row_id for row in roles["train"]} & {row.row_id for row in roles["model_selection"]}
            or {row.event_id for row in roles["train"]} & {row.event_id for row in roles["model_selection"]}):
        raise ClassicalError("Training and selection must use disjoint rows and events.")
    truth = np.array([row.y_true for row in roles["train"]], dtype=np.int64)
    if training_counts(truth).tolist() != verified["training_counts"]:
        raise ClassicalError("Frozen training label counts disagree.")
    lineage = {
        "freeze_id": verified["freeze_id"],
        "freeze_manifest_sha256": verified["freeze_manifest_sha256"],
        "archive_sha256": frozen.descriptor["archive_sha256"],
        "eligible_manifest_sha256": verified["eligible_manifest_sha256"],
        "role_manifest_sha256": {role: verified["role_manifest_sha256"][role] for role in roles},
    }
    return Inputs("research", lineage, texts["train"], truth, texts["model_selection"],
                  roles["model_selection"])


def training_counts(truth):
    counts = np.bincount(truth, minlength=len(LABELS))
    if counts.shape != (len(LABELS),) or not counts.all():
        raise ClassicalError("All ten fixed labels must be present in training.")
    return counts


def effective_weights(counts, class_weight):
    if class_weight is None:
        return [1.0] * len(LABELS)
    return (counts.sum() / (len(LABELS) * counts)).tolist()


def majority_predictions(counts, n_rows):
    return np.full(n_rows, int(counts.argmax()), dtype=np.int64)


def fit_features(train, selection):
    vectorizer = TfidfVectorizer(**VECTORIZER, dtype=np.float64)
    train_features = vectorizer.fit_transform(train)
    return vectorizer, train_features, vectorizer.transform(selection)


def fit_lr(candidate, sklearn_version, train, truth):
    model = LogisticRegression(**candidate.parameters(sklearn_version))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train, truth)
    except ConvergenceWarning:
        raise ClassicalError("Logistic regression did not converge.") from None
    return model, checked_iterations(model)


def checked_iterations(model):
    iterations = model.n_iter_
    if (not np.array_equal(model.classes_, np.arange(len(LABELS)))
            or not iterations.size or (iterations < 0).any()
            or (iterations >= CLASSICAL_SETTINGS["max_iter"]).any()
            or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all()):
        raise ClassicalError("Logistic regression failed class, convergence, or finiteness checks.")
    return iterations.tolist()


def decision_scores(model, features):
    with np.errstate(over="ignore", invalid="ignore"):
        scores = model.decision_function(features).astype(np.float32)
    if scores.shape != (features.shape[0], len(LABELS)) or not np.isfinite(scores).all():
        raise ClassicalError("Logistic regression did not produce finite ten-class logits.")
    return scores


def rank_lr(scores):
    if scores.keys() != CANDIDATES.keys() or any(
        type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1
        for score in scores.values()
    ):
        raise ClassicalError("Six valid LR selection scores are required.")
    best = max(scores.values())
    tied = sorted(
        (candidate for name, candidate in CANDIDATES.items()
         if scores[name] >= best - TIE_TOLERANCE),
        key=lambda candidate: (candidate.c, candidate.weight is not None),
    )
    return best, [candidate.name for candidate in tied], tied[0].name
