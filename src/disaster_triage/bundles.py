"""Prediction-bundle v1 production and shared validation; no metrics."""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from .accounting import identity
from .constants import ARCHIVE_SHA256, EVENTS, LABELS, ROLES
from .provenance import atomic_bytes, json_bytes, sha256_bytes, sha256_file, save_json

DIGEST_KEYS = {
    "archive_sha256", "eligible_manifest_sha256", "role_manifest_sha256",
    "config_sha256", "code_sha256",
}
FIELDS = {
    "schema_version", "run_id", "model_family", "seed", "prediction_kind", "role",
    "n_rows", "labels", *DIGEST_KEYS, "code_revision", "payload_sha256",
}

@dataclass(frozen=True)
class RoleRow:
    row_id: str
    event_id: str
    y_true: int

def role_bytes(rows):
    if len({r.row_id for r in rows}) != len(rows):
        raise ValueError("duplicate role rows")
    result = []
    for row in rows:
        if (not isinstance(row.row_id, str) or not row.row_id or
            not isinstance(row.event_id, str) or not row.event_id or
            type(row.y_true) is not int or not 0 <= row.y_true < 10):
            raise ValueError("invalid ordered role record")
        result.append(json_bytes({"row_id": row.row_id, "event_id": row.event_id,
                                  "y_true": row.y_true}))
    return b"".join(result)

def write_role_manifest(path, rows):
    content = role_bytes(rows)
    atomic_bytes(path, content)
    return sha256_bytes(content)

@dataclass(frozen=True)
class BundleExpectation:
    mode: str
    role: str
    rows: tuple
    digests: dict

    def __post_init__(self):
        if self.mode not in ("synthetic", "research") or self.role not in ROLES:
            raise ValueError("explicit synthetic/research mode and valid role required")
        if set(self.digests) != DIGEST_KEYS or any(
            not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
            for v in self.digests.values()
        ):
            raise ValueError("five valid expected SHA-256 values required")
        if self.digests["role_manifest_sha256"] != sha256_bytes(role_bytes(self.rows)):
            raise ValueError("expected role digest disagrees with ordered rows")
        if self.mode == "research":
            if self.digests["archive_sha256"] != ARCHIVE_SHA256:
                raise ValueError("research archive mismatch")
            events = EVENTS["final_test" if self.role == "final_test" else "development"]
            if any(row.event_id not in events for row in self.rows):
                raise ValueError("research role/event mismatch")
        elif any(not row.event_id.startswith("synthetic_") for row in self.rows):
            raise ValueError("synthetic fixtures must use synthetic event metadata")

def _gate(role):
    if role == "final_test":
        raise ValueError("final-test predictions disabled until Slice 8 selection/scope lock")

def validate_bundle(manifest, arrays, expected):
    _gate(expected.role)
    if not isinstance(manifest, dict) or set(manifest) != FIELDS:
        raise ValueError("missing or unsupported manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported schema version")
    identity(manifest["run_id"])
    family = manifest["model_family"]
    if family not in ("majority", "tfidf_lr", "distilbert"):
        raise ValueError("invalid model family")
    kind = "hard_labels" if family == "majority" else "logits"
    if manifest["prediction_kind"] != kind or manifest["role"] != expected.role:
        raise ValueError("family/kind or role mismatch")
    seed = manifest["seed"]
    if seed is not None and type(seed) is not int:
        raise ValueError("invalid seed type")
    if family == "distilbert" and seed not in (42, 43, 44):
        raise ValueError("invalid transformer seed")
    if type(manifest["n_rows"]) is not int or manifest["n_rows"] != len(expected.rows):
        raise ValueError("invalid row count")
    if manifest["labels"] != list(LABELS):
        raise ValueError("fixed label order mismatch")
    for key, value in expected.digests.items():
        if manifest[key] != value:
            raise ValueError("expected provenance digest mismatch")
    for key in (*DIGEST_KEYS, "payload_sha256"):
        if not isinstance(manifest[key], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[key]):
            raise ValueError("invalid manifest digest")
    revision = manifest["code_revision"]
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("invalid code revision")
    prediction = "y_pred" if kind == "hard_labels" else "logits"
    if set(arrays) != {"row_id", "event_id", "y_true", prediction}:
        raise ValueError("unexpected/missing payload arrays")
    n = len(expected.rows)
    for key in ("row_id", "event_id"):
        array = arrays[key]
        if array.dtype.kind != "U" or array.shape != (n,):
            raise ValueError("identifier arrays must be aligned Unicode vectors")
    y = arrays["y_true"]
    if y.dtype != np.dtype("int64") or y.shape != (n,) or not ((0 <= y) & (y < 10)).all():
        raise ValueError("invalid true-label vector")
    if len(set(arrays["row_id"].tolist())) != n:
        raise ValueError("duplicate payload rows")
    for key, wanted in (
        ("row_id", [r.row_id for r in expected.rows]),
        ("event_id", [r.event_id for r in expected.rows]),
        ("y_true", [r.y_true for r in expected.rows]),
    ):
        if arrays[key].tolist() != wanted:
            raise ValueError("payload disagrees with frozen role order/metadata")
    array = arrays[prediction]
    if kind == "logits":
        if array.dtype != np.dtype("float32") or array.shape != (n, 10) or not np.isfinite(array).all():
            raise ValueError("logits must be finite FP32 N by 10")
    elif array.dtype != np.dtype("int64") or array.shape != (n,) or not ((0 <= array) & (array < 10)).all():
        raise ValueError("invalid hard-prediction vector")

def write_bundle(directory, *, expected, run_id, model_family, seed, arrays, code_revision=None):
    _gate(expected.role)
    manifest = {
        "schema_version": 1, "run_id": run_id, "model_family": model_family,
        "seed": seed, "prediction_kind": "hard_labels" if model_family == "majority" else "logits",
        "role": expected.role, "n_rows": len(expected.rows), "labels": list(LABELS),
        **expected.digests, "code_revision": code_revision,
        "payload_sha256": sha256_bytes(b"pending"),
    }
    validate_bundle(manifest, arrays, expected)
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError("completed bundle cannot be overwritten")
    temporary = Path(tempfile.mkdtemp(prefix=".bundle-", dir=directory.parent))
    try:
        np.savez(temporary / "predictions.npz", **arrays)
        manifest["payload_sha256"] = sha256_file(temporary / "predictions.npz")
        save_json(temporary / "manifest.json", manifest)
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest

def read_bundle(directory, *, expected):
    """Minimal shared transport reader. Evaluation policy stays in Slice 4."""
    _gate(expected.role)
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    _gate(manifest.get("role"))
    if manifest.get("payload_sha256") != sha256_file(directory / "predictions.npz"):
        raise ValueError("payload digest mismatch")
    with np.load(directory / "predictions.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    validate_bundle(manifest, arrays, expected)
    return manifest, arrays
