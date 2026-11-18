import copy
from datetime import datetime, timedelta, timezone
import json
import numpy as np
import pytest
from disaster_triage.accounting import Attempt, budget_summary, guard_budget
from disaster_triage.bundles import (
    BundleExpectation, RoleRow, read_bundle, role_bytes, validate_bundle, write_bundle)
from disaster_triage.constants import ARCHIVE_SHA256
from disaster_triage.provenance import save_json, sha256_bytes

@pytest.fixture
def fixture():
    rows = (RoleRow("example-a", "synthetic_event_alpha", 7),
            RoleRow("example-b", "synthetic_event_alpha", 5))
    digest = sha256_bytes(b"synthetic")
    expected = BundleExpectation("synthetic", "model_selection", rows, {
        "archive_sha256": digest, "eligible_manifest_sha256": digest,
        "role_manifest_sha256": sha256_bytes(role_bytes(rows)),
        "config_sha256": digest, "code_sha256": digest,
    })
    arrays = {"row_id": np.array(["example-a", "example-b"]),
              "event_id": np.array(["synthetic_event_alpha"] * 2),
              "y_true": np.array([7, 5], dtype=np.int64),
              "logits": np.zeros((2, 10), dtype=np.float32)}
    return expected, arrays

@pytest.mark.parametrize("family", ["distilbert", "tfidf_lr", "majority"])
@pytest.mark.parametrize("empty", [False, True])
def test_bundle_roundtrip(tmp_path, fixture, family, empty):
    expected, arrays = fixture
    if empty:
        expected = BundleExpectation("synthetic", "model_selection", (), {
            **expected.digests, "role_manifest_sha256": sha256_bytes(b"")})
        arrays = {key: value[:0] for key, value in arrays.items()}
    if family == "majority":
        del arrays["logits"]
        arrays["y_pred"] = np.zeros(len(expected.rows), dtype=np.int64)
    manifest = write_bundle(tmp_path / "bundle", expected=expected, run_id="synthetic",
                            model_family=family, seed=42 if family == "distilbert" else None,
                            arrays=arrays)
    actual, payload = read_bundle(tmp_path / "bundle", expected=expected)
    assert actual == manifest
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], payload[key])
    with pytest.raises(FileExistsError):
        write_bundle(tmp_path / "bundle", expected=expected, run_id="synthetic",
                     model_family=family, seed=42, arrays=arrays)

@pytest.mark.parametrize("mutation", [
    "row_order", "duplicate", "extra_row", "event", "truth", "index", "objects",
    "shape", "float64", "nonfinite", "extra_array", "missing_array",
    "missing_field", "version", "bool_version", "labels", "hash", "kind",
    "role", "seed", "family", "count",
])
def test_rejections(tmp_path, fixture, mutation):
    expected, arrays = fixture
    manifest = write_bundle(tmp_path / "valid", expected=expected, run_id="synthetic",
                            model_family="distilbert", seed=42, arrays=arrays)
    if mutation == "row_order":
        arrays = {k: v[::-1] for k, v in arrays.items()}
    elif mutation == "duplicate": arrays["row_id"][1] = arrays["row_id"][0]
    elif mutation == "extra_row": arrays["row_id"] = np.append(arrays["row_id"], "extra")
    elif mutation == "event": arrays["event_id"][0] = "synthetic_event_beta"
    elif mutation == "truth": arrays["y_true"][0] = 1
    elif mutation == "index": arrays["y_true"][0] = 10
    elif mutation == "objects": arrays["row_id"] = arrays["row_id"].astype(object)
    elif mutation == "shape": arrays["logits"] = np.zeros((2, 9), dtype=np.float32)
    elif mutation == "float64": arrays["logits"] = arrays["logits"].astype(np.float64)
    elif mutation == "nonfinite": arrays["logits"][0, 0] = np.nan
    elif mutation == "extra_array": arrays["y_pred"] = np.zeros(2, dtype=np.int64)
    elif mutation == "missing_array": del arrays["logits"]
    elif mutation == "missing_field": del manifest["code_sha256"]
    elif mutation == "version": manifest["schema_version"] = 2
    elif mutation == "bool_version": manifest["schema_version"] = True
    elif mutation == "labels": manifest["labels"] = manifest["labels"][::-1]
    elif mutation == "hash": manifest["config_sha256"] = sha256_bytes(b"wrong")
    elif mutation == "kind": manifest["prediction_kind"] = "hard_labels"
    elif mutation == "role": manifest["role"] = "calibration"
    elif mutation == "seed": manifest["seed"] = True
    elif mutation == "family": manifest["model_family"] = "other"
    elif mutation == "count": manifest["n_rows"] = 1
    with pytest.raises(ValueError):
        validate_bundle(manifest, arrays, expected)

def test_payload_corruption_final_gate_and_research_metadata(tmp_path, fixture):
    expected, arrays = fixture
    target = tmp_path / "bundle"
    write_bundle(target, expected=expected, run_id="synthetic", model_family="tfidf_lr",
                 seed=None, arrays=arrays)
    (target / "predictions.npz").write_bytes(b"partial")
    with pytest.raises(ValueError, match="digest"):
        read_bundle(target, expected=expected)
    final = BundleExpectation("synthetic", "final_test", expected.rows, expected.digests)
    with pytest.raises(ValueError, match="Slice 8"):
        write_bundle(tmp_path / "final", expected=final, run_id="synthetic",
                     model_family="tfidf_lr", seed=None, arrays=arrays)
    with pytest.raises(ValueError, match="archive"):
        BundleExpectation("research", expected.role, expected.rows, expected.digests)
    with pytest.raises(ValueError, match="event"):
        BundleExpectation("research", expected.role, expected.rows,
                          {**expected.digests, "archive_sha256": ARCHIVE_SHA256})

class Clock:
    def __init__(self): self.seconds = 0.
    def monotonic(self): return self.seconds
    def now(self):
        return datetime(2026, 11, 18, tzinfo=timezone.utc) + timedelta(seconds=self.seconds)

def test_concurrent_and_resumed_attempts_are_summed_once(tmp_path):
    clock = Clock()
    with Attempt(tmp_path, "left", metadata={"device": "cuda:0"},
                 monotonic=clock.monotonic, now=clock.now):
        with Attempt(tmp_path, "right", metadata={"device": "cuda:1"},
                     monotonic=clock.monotonic, now=clock.now):
            clock.seconds = 3600
    result = budget_summary(tmp_path, now=clock.now())
    assert result["active_job_hours"] == 2
    assert result["wall_span_hours"] == 1
    assert result["gpu_device_hours"] == 2
    with pytest.raises(RuntimeError):
        with Attempt(tmp_path, "left", metadata={"device": "cpu"},
                     monotonic=clock.monotonic, now=clock.now):
            clock.seconds += 1800
            raise RuntimeError("synthetic failure")
    assert budget_summary(tmp_path, now=clock.now())["active_job_hours"] == 2.5
    with Attempt(tmp_path, "left", metadata={"parent_checkpoint_sha256": sha256_bytes(b"checkpoint")},
                 monotonic=clock.monotonic, now=clock.now):
        clock.seconds += 900
    assert budget_summary(tmp_path, now=clock.now())["active_job_hours"] == 2.75

def test_budget_cutoff_and_unfinished_attempt(tmp_path):
    clock = Clock()
    attempt = Attempt(tmp_path, "unfinished", metadata={}, monotonic=clock.monotonic, now=clock.now)
    attempt.__enter__()
    clock.seconds = 60
    attempt.update()
    clock.seconds = 300
    summary = guard_budget(tmp_path, now=clock.now())
    assert summary["active_job_hours"] == pytest.approx(300 / 3600)
    assert len(summary["unresolved_attempts"]) == 1
    assert budget_summary(tmp_path, now=clock.now()) == summary
    clock.seconds = 24 * 3600
    with pytest.raises(RuntimeError, match="exhausted"):
        guard_budget(tmp_path, now=clock.now())
    with pytest.raises(RuntimeError, match="cutoff"):
        guard_budget(tmp_path, now=datetime(2026, 11, 24, tzinfo=timezone.utc))


def test_retained_accounting_import_is_idempotent_and_preserves_budget(tmp_path):
    from disaster_triage.accounting import merge_accounting
    clock = Clock()
    source, restored = tmp_path / "old", tmp_path / "new"
    with Attempt(source, "retained", metadata={}, monotonic=clock.monotonic, now=clock.now):
        clock.seconds = 3600
    assert merge_accounting(source, restored) == 1
    assert merge_accounting(source, restored) == 0
    assert budget_summary(restored, now=clock.now())["active_job_hours"] == 1
    with Attempt(restored, "retained", metadata={}, monotonic=clock.monotonic, now=clock.now):
        clock.seconds += 1800
    assert budget_summary(restored, now=clock.now())["active_job_hours"] == 1.5


def test_reader_gates_declared_final_role_before_opening_payload(tmp_path, fixture):
    expected, arrays = fixture
    target = tmp_path / "misdeclared"
    target.mkdir()
    save_json(target / "manifest.json", {"role": "final_test"})
    # There is no payload: gate must fire before even trying to open one.
    with pytest.raises(ValueError, match="Slice 8"):
        read_bundle(target, expected=expected)


def test_parallel_probe_snapshots_keep_final_attempt_accounting(tmp_path):
    from disaster_triage.accounting import merge_accounting
    clock = Clock()
    active, snapshot, restored = tmp_path / "active", tmp_path / "snapshot", tmp_path / "restored"
    with Attempt(active, "worker-1", metadata={"device": "cuda:1"},
                 monotonic=clock.monotonic, now=clock.now):
        clock.seconds = 10
        merge_accounting(active, snapshot)  # A peer captured a still-running heartbeat.
        clock.seconds = 20
    merge_accounting(active, restored)  # The complete session ledger is retained.
    assert merge_accounting(snapshot, restored) == 0
    assert not budget_summary(restored, now=clock.now())["unresolved_attempts"]
    assert budget_summary(restored, now=clock.now())["active_job_hours"] == 20 / 3600
    assert merge_accounting(active, snapshot) == 1  # Later completion supersedes heartbeat.
    assert not budget_summary(snapshot, now=clock.now())["unresolved_attempts"]
    path = next((snapshot / "runs").glob("*/attempts/*/attempt.json"))
    record = json.loads(path.read_text())
    record["metadata"]["device"] = "cuda:0"  # Different provenance is a real conflict.
    save_json(path, record)
    with pytest.raises(ValueError, match="conflicting"):
        merge_accounting(snapshot, restored)
