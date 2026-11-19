import copy
from dataclasses import replace
import json
from math import log
from types import SimpleNamespace

import numpy as np
import pytest

from disaster_triage import evaluation
from disaster_triage.bundles import BundleExpectation, RoleRow, role_bytes, write_bundle
from disaster_triage.constants import LABELS
from disaster_triage.evaluation import BundleInput, SeedCohort, fit_review_policy, score_bundle, summarize_seeds
from disaster_triage.provenance import json_bytes, save_json, sha256_bytes, sha256_file


ALPHA = "synthetic_alpha"
BETA = "synthetic_beta"
REQUESTS = LABELS[7]
MISSING = LABELS[4]


def _bundle(root, role, truth, scores, *, events=None, row_prefix=None, family="tfidf_lr",
            seed=None, run_id="synthetic-run", digests=None):
    if events is None:
        events = [ALPHA] * len(truth)
    rows = tuple(
        RoleRow(f"{row_prefix or role}-{index}", event, label)
        for index, (event, label) in enumerate(zip(events, truth, strict=True))
    )
    expected = BundleExpectation("synthetic", role, rows, {
        "archive_sha256": sha256_bytes(b"invented archive"),
        "eligible_manifest_sha256": sha256_bytes(b"invented eligible manifest"),
        "role_manifest_sha256": sha256_bytes(role_bytes(rows)),
        "config_sha256": sha256_bytes(f"invented config {seed}".encode()),
        "code_sha256": sha256_bytes(b"invented producer"),
        **(digests or {}),
    })
    arrays = {
        "row_id": np.array([row.row_id for row in rows], dtype="U"),
        "event_id": np.array(events, dtype="U"),
        "y_true": np.array(truth, dtype=np.int64),
    }
    if family == "majority":
        arrays["y_pred"] = np.array(scores, dtype=np.int64)
    else:
        arrays["logits"] = np.array(scores, dtype=np.float32).reshape(len(rows), len(LABELS))
    directory = root / run_id / role
    write_bundle(directory, expected=expected, run_id=run_id, model_family=family, seed=seed, arrays=arrays)
    return BundleInput(directory, expected, run_id, family, seed)


def _logits(predicted, value=2.0):
    logits = np.zeros((len(predicted), len(LABELS)), dtype=np.float32)
    logits[np.arange(len(predicted)), predicted] = value
    return logits


def _policy_pair(root):
    calibration = _bundle(root, "calibration", [0, 1], np.zeros((2, 10)))
    policy = _bundle(
        root, "policy", [7, 7, 5, 4], _logits([7, 5, 7, 4]), events=[ALPHA, ALPHA, ALPHA, BETA]
    )
    return calibration, policy


def _apply(bundle, fitted):
    return score_bundle(bundle, fitted_policy=fitted, fitted_policy_sha256=sha256_bytes(json_bytes(fitted)))


def test_roundtrip_three_roles_classification_calibration_and_review(tmp_path):
    calibration, policy = _policy_pair(tmp_path)
    selection = _bundle(tmp_path, "model_selection", [7], _logits([7]))
    before = {path: sha256_file(path) for path in tmp_path.rglob("*") if path.is_file()}
    assert score_bundle(selection)["pooled"]["macro_f1"] == pytest.approx(0.1)
    assert score_bundle(calibration)["calibration"]["raw"]["nll"] == pytest.approx(log(10))
    fitted = fit_review_policy(calibration, policy)
    assert fitted["temperature"] == 1.0
    assert fitted["inputs"]["calibration"]["role_manifest_sha256"] == calibration.expected.digests["role_manifest_sha256"]
    assert fitted["inputs"]["policy"]["payload_sha256"] == before[policy.directory / "predictions.npz"]
    record_path = tmp_path / "policy.json"
    digest = save_json(record_path, fitted)
    report = score_bundle(
        policy, fitted_policy=json.loads(record_path.read_text()), fitted_policy_sha256=digest
    )
    assert json.loads(json_bytes(report)) == report
    assert report["fitted_policy_sha256"] == digest
    assert report["bundle"]["payload_sha256"] == before[policy.directory / "predictions.npz"]
    assert report["pooled"]["macro_f1"] == pytest.approx(0.15)
    assert report["events"][ALPHA]["macro_f1"] == pytest.approx(0.05)
    assert report["events"][BETA]["macro_f1"] == pytest.approx(0.1)
    request = report["pooled"]["classes"][REQUESTS]
    assert request == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2, "predicted_count": 2}
    assert report["events"][ALPHA]["classes"][MISSING]["support"] == 0
    assert report["events"][BETA]["classes"][MISSING]["recall"] == 1
    assert report["calibration"]["raw"] == report["calibration"]["calibrated"]
    assert report["review"][0]["pooled"]["requests"] == {
        "support": 2, "recall": 0.5, "deferred": 0, "deferred_fraction": 0.0,
        "accepted_false_negatives": 1, "accepted_false_positives": 1,
    }
    # Even the 30% nominal case leaves tied confidences together.
    assert [entry["pooled"]["deferred"] for entry in report["review"]] == [0, 0, 0, 0]
    assert fit_review_policy(calibration, policy) == fitted
    assert all(sha256_file(path) == expected for path, expected in before.items())


def test_majority_metrics_only_and_zero_denominators(tmp_path):
    bundle = _bundle(tmp_path, "model_selection", [7, 5, 7], [5, 5, 5], family="majority")
    report = score_bundle(bundle)
    assert "calibration" not in report and "review" not in report
    assert tuple(report["pooled"]["classes"]) == LABELS
    assert report["pooled"]["classes"][REQUESTS]["predicted_count"] == 0
    assert report["pooled"]["classes"][REQUESTS]["support"] == 2
    assert report["pooled"]["classes"][REQUESTS]["precision"] == 0
    assert report["pooled"]["classes"][MISSING]["support"] == 0


@pytest.mark.parametrize("family", ["tfidf_lr", "majority"])
def test_empty_scoring_is_unavailable(tmp_path, family):
    report = score_bundle(_bundle(tmp_path, "model_selection", [], [], family=family))
    assert report["pooled"]["macro_f1"] is None
    assert not report["events"]
    assert all(row["support"] == 0 for row in report["pooled"]["classes"].values())
    if family == "tfidf_lr":
        assert report["calibration"]["raw"] == {"nll": None, "ece": None}


def test_unweighted_nll_extreme_logits_and_argmax_ties(tmp_path):
    extreme = np.finfo(np.float32).max
    logits = np.full((2, 10), -extreme, dtype=np.float32)
    logits[:, 0:2] = extreme
    report = score_bundle(_bundle(tmp_path, "model_selection", [0, 7], logits))
    assert report["pooled"]["classes"][LABELS[0]]["predicted_count"] == 2
    assert report["pooled"]["classes"][LABELS[1]]["predicted_count"] == 0
    assert report["calibration"]["raw"]["nll"] == pytest.approx(float(extreme))
    assert report["calibration"]["raw"]["ece"] == 0


def test_ece_bin_boundaries_and_confidence_one():
    boundary = 8 / 15
    confidence = np.array([boundary - 1e-8, boundary, 14 / 15, 1.0])
    correct = np.array([True, False, True, False])
    expected = (1 - confidence[0] + boundary + abs(1 - (14 / 15 + 1))) / 4
    assert evaluation._ece(confidence, correct) == pytest.approx(expected)


def test_temperature_uses_calibration_rows_and_preserves_argmax(tmp_path):
    calibration = _bundle(tmp_path, "calibration", [0, 1, 0, 0], _logits([0] * 4, 9))
    policy = _bundle(tmp_path, "policy", [2, 3], _logits([2, 3], 9))
    fitted = fit_review_policy(calibration, policy)
    assert fitted["temperature"] == pytest.approx(9 / log(27), rel=1e-6)
    calibrated = _apply(policy, fitted)
    raw = score_bundle(policy)
    assert calibrated["pooled"] == raw["pooled"]
    assert calibrated["events"] == raw["events"]
    assert calibrated["calibration"]["calibrated"]["nll"] > calibrated["calibration"]["raw"]["nll"]


def test_global_thresholds_event_workload_and_empty_acceptance(tmp_path):
    calibration = _bundle(tmp_path, "calibration", [0], np.zeros((1, 10)))
    # The weakest group is its own event and entirely deferred at 20% and 30%.
    truth = [7, 7] + [0] * 8
    logits = _logits([5, 7] + [0] * 8, [1.0, 1.0] + [4.0] * 8)
    policy = _bundle(tmp_path, "policy", truth, logits, events=[ALPHA] * 2 + [BETA] * 8)
    fitted = fit_review_policy(calibration, policy)
    report = _apply(policy, fitted)
    assert [row["threshold"] for row in report["review"]] == [row["threshold"] for row in fitted["thresholds"]]
    assert [row["pooled"]["deferred"] for row in report["review"]] == [0, 0, 2, 2]
    case = report["review"][2]
    assert case["pooled"]["coverage"] == 0.8
    assert case["events"][ALPHA]["review_fraction"] == 1
    assert case["events"][ALPHA]["selective_error"] is None
    assert case["events"][ALPHA]["requests"]["deferred_fraction"] == 1
    assert case["events"][ALPHA]["requests"]["accepted_false_negatives"] == 0
    assert case["events"][BETA]["review_fraction"] == 0
    assert case["events"][BETA]["requests"]["deferred_fraction"] is None


@pytest.mark.parametrize("role", ["calibration", "policy"])
def test_empty_role_cannot_fit(tmp_path, role):
    calibration = _bundle(tmp_path, "calibration", [] if role == "calibration" else [0],
                          [] if role == "calibration" else _logits([0]))
    policy = _bundle(tmp_path, "policy", [] if role == "policy" else [0],
                     [] if role == "policy" else _logits([0]))
    with pytest.raises(ValueError, match="nonempty"):
        fit_review_policy(calibration, policy)


@pytest.mark.parametrize("success,value,position", [(False, 1.0, 0.0), (True, np.nan, 0.0), (True, 1.0, np.inf)])
def test_optimizer_failure_or_nonfinite_output(tmp_path, monkeypatch, success, value, position):
    calibration, policy = _policy_pair(tmp_path)
    monkeypatch.setattr(evaluation, "minimize_scalar", lambda *args, **kwargs: SimpleNamespace(
        success=success, fun=value, x=position,
    ))
    with pytest.raises(ValueError, match="optimization failed"):
        fit_review_policy(calibration, policy)


def test_wrong_role_overlap_and_majority_cannot_fit(tmp_path):
    calibration = _bundle(tmp_path, "calibration", [0], _logits([0]), row_prefix="shared")
    policy = _bundle(tmp_path, "policy", [0], _logits([0]), row_prefix="shared")
    with pytest.raises(ValueError, match="requires calibration and policy"):
        fit_review_policy(policy, calibration)
    with pytest.raises(ValueError, match="overlap"):
        fit_review_policy(calibration, policy)
    majority_cal = _bundle(tmp_path, "calibration", [0], [0], family="majority", run_id="majority")
    majority_policy = _bundle(tmp_path, "policy", [0], [0], family="majority", run_id="majority")
    with pytest.raises(ValueError, match="requires classifier logits"):
        fit_review_policy(majority_cal, majority_policy)


@pytest.mark.parametrize("field", ["archive_sha256", "eligible_manifest_sha256", "config_sha256", "code_sha256", "run_id"])
def test_fit_detects_cross_run_or_provenance_mismatch(tmp_path, field):
    calibration = _bundle(tmp_path, "calibration", [0], _logits([0]))
    policy = _bundle(tmp_path, "policy", [0], _logits([0]),
                     run_id="alternate" if field == "run_id" else calibration.run_id,
                     digests={} if field == "run_id" else {field: sha256_bytes(b"other identity")})
    with pytest.raises(ValueError, match="must share"):
        fit_review_policy(calibration, policy)


@pytest.mark.parametrize("field,value", [("run_id", "other"), ("model_family", "majority"), ("seed", 42)])
def test_consumer_checks_independently_expected_run_identity(tmp_path, field, value):
    bundle = _bundle(tmp_path, "model_selection", [0], _logits([0]))
    with pytest.raises(ValueError, match="identity|mismatch"):
        score_bundle(replace(bundle, **{field: value}))


def test_consumer_rejects_changed_provenance_payload_and_row_order(tmp_path):
    bundle = _bundle(tmp_path, "model_selection", [0, 1], _logits([0, 1]))
    different = replace(bundle.expected, digests={**bundle.expected.digests, "code_sha256": sha256_bytes(b"other")})
    with pytest.raises(ValueError, match="provenance digest"):
        score_bundle(replace(bundle, expected=different))
    target = bundle.directory / "predictions.npz"
    with np.load(target, allow_pickle=False) as payload:
        arrays = {key: payload[key][::-1] for key in payload.files}
    np.savez(target, **arrays)
    with pytest.raises(ValueError, match="payload digest"):
        score_bundle(bundle)
    path = bundle.directory / "manifest.json"
    save_json(path, {**json.loads(path.read_text()), "payload_sha256": sha256_file(target)})
    with pytest.raises(ValueError, match="role order"):
        score_bundle(bundle)


def test_fitted_policy_digest_run_and_payload_must_match(tmp_path):
    calibration, policy = _policy_pair(tmp_path)
    fitted = fit_review_policy(calibration, policy)
    with pytest.raises(ValueError, match="together"):
        score_bundle(policy, fitted_policy=fitted)
    digest = sha256_bytes(json_bytes(fitted))
    altered = copy.deepcopy(fitted)
    altered["temperature"] *= 2
    with pytest.raises(ValueError, match="digest"):
        score_bundle(policy, fitted_policy=altered, fitted_policy_sha256=digest)
    different_run = _bundle(tmp_path, "policy", [0], _logits([0]), run_id="different")
    with pytest.raises(ValueError, match="evaluated run"):
        _apply(different_run, fitted)
    alternate_payload = _bundle(
        tmp_path / "alternate-payload", "policy", [7, 7, 5, 4],
        _logits([7, 5, 7, 4], 3), events=[ALPHA, ALPHA, ALPHA, BETA],
    )
    with pytest.raises(ValueError, match="provenance"):
        _apply(alternate_payload, fitted)
    with pytest.raises(ValueError, match="applies to"):
        _apply(calibration, fitted)


def test_score_rejects_final_before_reading_any_payload(tmp_path):
    bundle = BundleInput(
        tmp_path / "missing", BundleExpectation("synthetic", "final_test", (), {
            key: sha256_bytes(b"") for key in (
                "archive_sha256", "eligible_manifest_sha256", "role_manifest_sha256",
                "config_sha256", "code_sha256",
            )
        }), "synthetic-run", "tfidf_lr", None,
    )
    with pytest.raises(ValueError, match="Slice 8"):
        score_bundle(bundle)


def _seed_reports(root):
    truth = [0, 1, 1, 2, 2, 2]
    predictions = [[0, 1, 1, 0, 0, 0], [0, 1, 1, 2, 2, 2], [0, 1, 1, 2, 2, 2]]
    reports = [
        score_bundle(_bundle(root, "model_selection", truth, _logits(predicted),
                             family="distilbert", seed=seed, run_id=f"seed-{seed}"))
        for seed, predicted in zip((42, 43, 44), predictions, strict=True)
    ]
    cohort = SeedCohort(sha256_bytes(b"same selected settings"), 2, {
        report["bundle"]["seed"]: report["bundle"]["config_sha256"] for report in reports
    })
    return reports, cohort


def test_seed_statistics_and_individual_reports(tmp_path):
    # A hand-check of the arithmetic, plus an integration test of scoring and cohorts.
    assert evaluation._seed_stats([{"macro_f1": number} for number in (0.2, 0.3, 0.4)]) == {
        "mean": pytest.approx(0.3), "sample_sd": pytest.approx(0.1),
    }
    reports, cohort = _seed_reports(tmp_path)
    result = summarize_seeds(reversed(reports), cohort=cohort)
    assert result["n_seeds"] == 3 and not result["exploratory"]
    assert result["individual"] == reports
    assert json.loads(json_bytes(result)) == result
    assert result["events"][ALPHA] == result["pooled"]
    assert result["pooled"]["mean"] == pytest.approx(
        sum(report["pooled"]["macro_f1"] for report in reports) / 3
    )
    partial = summarize_seeds(reports[:2], cohort=cohort)
    assert partial["exploratory"] and partial["pooled"]["sample_sd"] > 0
    single = summarize_seeds(reports[:1], cohort=cohort)
    assert single["exploratory"] and single["pooled"] == {"mean": None, "sample_sd": None}
    assert single["individual"] == reports[:1]


def test_seed_cohort_rejects_duplicate_and_mismatched_reports(tmp_path):
    reports, cohort = _seed_reports(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        summarize_seeds([reports[0], reports[0]], cohort=cohort)
    for key, value in {
        "role": "policy", "role_manifest_sha256": sha256_bytes(b"new rows"),
        "eligible_manifest_sha256": sha256_bytes(b"new freeze"),
        "config_sha256": sha256_bytes(b"different selected settings"),
        "model_family": "tfidf_lr",
    }.items():
        changed = copy.deepcopy(reports[:2])
        changed[1]["bundle"][key] = value
        with pytest.raises(ValueError):
            summarize_seeds(changed, cohort=cohort)
