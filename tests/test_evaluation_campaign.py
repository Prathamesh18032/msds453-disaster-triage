"""Exercise held-out access, receipts and resumption with invented rows only."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from disaster_triage.accounting import Attempt, budget_summary, guard_budget, guard_evaluation_budget
from disaster_triage.bundles import RoleRow, read_bundle, role_bytes
from disaster_triage.cli import main
from disaster_triage.constants import EVENTS, LABELS
from disaster_triage.data_policy import POLICY_SHA256
from disaster_triage.evaluation import SeedCohort, score_bundle, summarize_seeds
from disaster_triage.evaluation_campaign import EvaluationCampaign
from disaster_triage.evaluation_lock import EVALUATION_ROLES, RUN_SCOPE, EvaluationError, digest
from disaster_triage.provenance import json_bytes, save_json, sha256_bytes


@pytest.fixture
def campaign_fixture(tmp_path):
    files = {"eligible-manifest.jsonl": b"invented eligibility\n"}
    for role in EVALUATION_ROLES:
        rows = tuple(RoleRow(f"{role}-{i}", f"synthetic_{i % 2}", i % 8) for i in range(20))
        files[f"roles/{role}.jsonl"] = role_bytes(rows)
        files[f"texts/{role}.jsonl"] = b"".join(json_bytes({
            "row_id": row.row_id, "event_id": row.event_id, "y_true": row.y_true, "role": role,
            "classical_text": "invented " + "a" * i, "transformer_text": "Invented " + "x" * i,
        }) for i, row in enumerate(rows))
    descriptor = {
        "schema_version": 1, "corpus_kind": "synthetic", "archive_sha256": digest("invented archive"),
        "archive_bytes": 1, "event_rows": {}, "policy_sha256": POLICY_SHA256,
        "labels": list(LABELS), "events": {key: list(events) for key, events in EVENTS.items()},
        "files": {name: sha256_bytes(content) for name, content in files.items()},
        "summary": {"role_counts": dict.fromkeys(EVALUATION_ROLES, 20)},
    }
    freeze_id = digest(descriptor)
    descriptor.update(freeze_id=freeze_id, complete=True)
    freeze = tmp_path / "data" / freeze_id
    for name, content in files.items():
        path = freeze / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    descriptor_hash = save_json(freeze / "freeze.json", descriptor)
    lock = {
        "schema_version": 1, "campaign": "invented", "mode": "synthetic",
        "majority_label": 0, "selection": {"primary": "distilbert"},
        "freeze": {"path": f"data/{freeze_id}", "sha256": descriptor_hash},
        "runs": {key: {"model_family": family, "seed": seed} for key, (family, seed) in RUN_SCOPE.items()},
    }
    lock_path = tmp_path / "artifacts/evaluation/invented/selection-lock.json"
    return tmp_path, freeze, lock_path, save_json(lock_path, lock)


@pytest.fixture
def prepared(campaign_fixture):
    root, freeze, lock_path, lock_sha256 = campaign_fixture
    campaign = EvaluationCampaign(root, "invented", lock_sha256)
    import os

    source = Path(os.environ.get("SLICE8_TEST_SOURCE", Path(__file__).resolve().parents[1]))
    probe = campaign.probe(source)
    assert probe["probe_rows_per_predictor"] == 256
    assert probe["rows_per_trained_run"] == 60
    assert probe["backend"] == "cpu"
    ready = campaign.prepare()["readiness_sha256"]
    return campaign, freeze, lock_path, ready


def test_full_campaign_and_read_only_replay(prepared, monkeypatch):
    campaign, _, _, ready = prepared
    assert campaign.prepare()["readiness_sha256"] == ready
    actual = campaign.final(ready)
    probe, _, access = campaign.final_context(ready)
    majority = campaign.view("majority", "final_test", campaign.final_target("majority") / "bundle", probe)
    with pytest.raises(ValueError, match="Slice 8"):
        read_bundle(majority.directory, expected=majority.expected)
    with pytest.raises(ValueError, match="Slice 8"):
        score_bundle(majority)
    with pytest.raises(ValueError, match="Slice 8"):
        campaign.frozen.load_role("final_test")
    report = score_bundle(majority, final_access=access)
    assert not {"review", "calibration"} & set(report)
    assert report["pooled"]["classes"][LABELS[9]] == {
        "f1": None, "precision": None, "recall": None, "support": 0, "predicted_count": 0,
    }
    manifest = json.loads(campaign.manifest_path.read_bytes())
    assert manifest["readiness_sha256"] == ready
    assert len(manifest["outputs"]) == 5
    for key in ("lr", "seed42", "seed43", "seed44"):
        record = json.loads((campaign.final_target(key) / "report.json").read_bytes())
        assert [item["review_percent"] for item in record["review"]] == [0, 10, 20, 30]
        assert record["review"][0]["pooled"]["coverage"] == 1
        assert set(record["calibration"]) == {"raw", "calibrated"}
    calls = budget_summary(campaign.artifacts)["attempts"]
    assert len(calls) == 3 and all(item["status"] == "completed" for item in calls)
    def forbidden(*_):
        pytest.fail("verification must reuse prediction bytes")
    monkeypatch.setattr(campaign, "predictor", forbidden)
    monkeypatch.setattr(campaign.frozen, "load_final_role", forbidden)
    assert campaign.final(ready, verify=True) == actual
    assert budget_summary(campaign.artifacts)["attempts"] == calls


@pytest.mark.parametrize("mutation", ["wrong_digest", "missing", "policy", "role_order", "bundle", "unresolved"])
def test_final_checks_before_exposure(prepared, monkeypatch, mutation):
    campaign, freeze, _, ready = prepared
    if mutation == "wrong_digest":
        ready = digest("substituted readiness")
    if mutation == "missing":
        campaign.readiness_path.unlink()
    if mutation == "policy":
        path = campaign.path("policies/lr.json")
        original = json.loads(path.read_bytes())
        original["temperature"] *= 2
        save_json(path, original)
    if mutation == "role_order":
        path = freeze / "roles/final_test.jsonl"
        path.write_bytes(b"".join(reversed(path.read_bytes().splitlines(keepends=True))))
    if mutation == "bundle":
        (campaign.development_target("lr", "policy") / "bundle/predictions.npz").write_bytes(b"changed")
    if mutation == "unresolved":
        Attempt(campaign.artifacts, "invented-unresolved", metadata={}).__enter__()
    def forbidden(*_):
        pytest.fail("unready campaign accessed held-out text")
    monkeypatch.setattr(campaign.frozen, "load_final_role", forbidden)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        campaign.final(ready)
    assert not campaign.path("final").exists()


def test_lock_digest_scope_and_model_substitution(campaign_fixture):
    root, _, path, sha256 = campaign_fixture
    with pytest.raises(EvaluationError, match="independently retained"):
        EvaluationCampaign(root, "invented", digest("different lock"))
    lock = json.loads(path.read_bytes())
    lock["runs"]["seed43"]["seed"] = 42
    with pytest.raises(EvaluationError, match="unique seeds"):
        EvaluationCampaign(root, "invented", save_json(path, lock))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        EvaluationCampaign(root, "invented", sha256)
    model = root / "artifacts/invented-checkpoint"
    model.write_bytes(b"original")
    original_sha256 = sha256_bytes(model.read_bytes())
    model.write_bytes(b"substituted")
    lock["runs"]["seed43"]["seed"] = 43
    campaign = EvaluationCampaign(root, "invented", save_json(path, lock))
    with pytest.raises(EvaluationError, match="locked file digest"):
        campaign.checked_file("artifacts/invented-checkpoint", original_sha256)


def test_interrupted_publication_resumes_only_pending(prepared, monkeypatch):
    campaign, _, _, ready = prepared
    original_predictor = campaign.predictor
    original_rename = Path.rename
    failure = True
    calls = []

    def renaming(path, target):
        if failure and target == campaign.final_target("lr"):
            raise KeyboardInterrupt()
        return original_rename(path, target)

    def predictor(key, *args):
        calls.append(key)
        return original_predictor(key, *args)

    monkeypatch.setattr(Path, "rename", renaming)
    monkeypatch.setattr(campaign, "predictor", predictor)
    with pytest.raises(KeyboardInterrupt):
        campaign.final(ready)
    assert calls == ["majority", "lr"]
    assert not list(campaign.path("final").glob(".pending-*"))
    assert not campaign.final_target("lr").exists()
    first = (campaign.final_target("majority") / "receipt.json").read_bytes()
    failure = False
    calls.clear()
    campaign.final(ready)
    assert calls == ["lr", "seed42", "seed43", "seed44"]
    assert (campaign.final_target("majority") / "receipt.json").read_bytes() == first
    assert not budget_summary(campaign.artifacts)["unresolved_attempts"]


def test_completed_final_output_cannot_be_regenerated(prepared):
    campaign, _, _, ready = prepared
    campaign.final(ready)
    campaign.manifest_path.unlink()
    campaign.final_target("lr").rename(campaign.path("archived-lr"))
    with pytest.raises(EvaluationError, match="completed final result is missing"):
        campaign.final(ready)


def test_seed_summary_requires_verified_complete_cohort(prepared):
    campaign, _, _, ready = prepared
    campaign.final(ready)
    _, _, access = campaign.final_context(ready)
    reports = [json.loads((campaign.final_target(f"seed{seed}") / "report.json").read_bytes())
               for seed in (42, 43, 44)]
    cohort = SeedCohort(digest("cohort"), 3, {
        report["bundle"]["seed"]: report["bundle"]["config_sha256"] for report in reports
    })
    with pytest.raises(ValueError, match="Slice 8"):
        summarize_seeds(reports, cohort=cohort)
    with pytest.raises(ValueError, match="unique"):
        summarize_seeds([reports[0], reports[0], reports[1]], cohort=cohort, final_access=access)


def test_cli_requires_independent_digests(campaign_fixture):
    root, _, _, sha256 = campaign_fixture
    with pytest.raises(SystemExit):
        main(["evaluation", "final", "--campaign-id", "invented", "--root", str(root)])
    with pytest.raises(SystemExit):
        main(["evaluation", "final", "--campaign-id", "invented", "--root", str(root),
              "--selection-lock-sha256", sha256])
    assert main(["evaluation", "final", "--campaign-id", "invented", "--root", str(root),
                 "--selection-lock-sha256", sha256, "--readiness-sha256", digest("absent")]) == 1


def test_evaluation_budget_after_cutoff_and_exhaustion(tmp_path):
    after = datetime(2026, 11, 24, 22, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="training cutoff"):
        guard_budget(tmp_path, now=after)
    assert guard_evaluation_budget(tmp_path, 1, now=after) == 86400
    with pytest.raises(RuntimeError, match="remaining"):
        guard_evaluation_budget(tmp_path, 86400, now=after)
    now = lambda: after
    times = iter((0, 0, 86400))
    with Attempt(tmp_path, "completed-cap", metadata={}, monotonic=lambda: next(times), now=now):
        pass
    with pytest.raises(RuntimeError, match="remaining"):
        guard_evaluation_budget(tmp_path, 1, now=after)
