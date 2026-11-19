"""Offline acceptance for the classical grid and private artifacts."""
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
import warnings

from disaster_triage import classical_run
from disaster_triage.accounting import budget_summary
from disaster_triage.bundles import BundleExpectation, RoleRow, read_bundle, role_bytes, write_bundle
from disaster_triage.classical import (
    CANDIDATES, ClassicalError, Inputs, aligned_text, checked_iterations,
    decision_scores, effective_weights, fit_features, fit_lr,
    majority_predictions, rank_lr, training_counts,
)
from disaster_triage.constants import LABELS
from disaster_triage.dataset import FrozenDataset
from disaster_triage.provenance import sha256_bytes

SOURCE_ROOT = Path(os.environ.get("SLICE5_SOURCE_ROOT", Path(__file__).resolve().parents[1]))


@pytest.fixture
def before_cutoff(monkeypatch):
    # Synthetic checks remain runnable after the research cutoff.
    monkeypatch.setattr(classical_run, "utc_now", lambda: datetime.fromisoformat("2026-11-19T12:00:00Z"))


@pytest.fixture
def inputs():
    truth = np.array([*range(10), *range(10), 3, 3], dtype=np.int64)
    rows = tuple(RoleRow(f"selection-{c}", "synthetic_dev", c) for c in range(10))
    lineage = {
        "archive_sha256": sha256_bytes(b"invented archive"),
        "eligible_manifest_sha256": sha256_bytes(b"invented eligibility"),
        "role_manifest_sha256": {
            "train": sha256_bytes(b"invented training manifest"),
            "model_selection": sha256_bytes(role_bytes(rows)),
        },
    }
    return Inputs(
        "synthetic", lineage,
        tuple(f"marker{c} token{c}" for c in truth), truth,
        tuple(f"marker{c} token{c} developmentonly" for c in range(10)), rows,
    )


def test_train_only_vocabulary_idf_and_weights(inputs):
    vectorizer, train, selection = fit_features(inputs.train_text, inputs.selection_text)
    assert "developmentonly" not in vectorizer.vocabulary_
    vocab = vectorizer.vocabulary_
    for c in range(10):
        for term in (f"marker{c}", f"token{c}", f"marker{c} token{c}"):
            expected = 1 + np.log(23 / (5 if c == 3 else 3))
            assert vectorizer.idf_[vocab[term]] == pytest.approx(expected)
    assert train.dtype == selection.dtype == np.float64
    counts = training_counts(inputs.train_labels)
    assert counts.tolist() == [2, 2, 2, 4, 2, 2, 2, 2, 2, 2]
    assert effective_weights(counts, "balanced")[3] == 22 / 40
    assert effective_weights(counts, None) == [1.0] * 10
    assert majority_predictions(counts, 3).tolist() == [3, 3, 3]
    assert majority_predictions(np.ones(10), 2).tolist() == [0, 0]
    with pytest.raises(ClassicalError, match="ten fixed labels"):
        training_counts(np.arange(9))


def test_exact_grid_and_l2_profile_api():
    assert [(c.c, c.weight) for c in CANDIDATES.values()] == [
        (0.1, None), (0.1, "balanced"), (1.0, None),
        (1.0, "balanced"), (10.0, None), (10.0, "balanced"),
    ]
    for candidate in CANDIDATES.values():
        for version, parameter, expected in (("1.6.1", "penalty", "l2"), ("1.9.1", "l1_ratio", 0.0)):
            kwargs = candidate.parameters(version)
            assert kwargs[parameter] == expected
            assert (kwargs["solver"], kwargs["max_iter"], kwargs["tol"]) == ("lbfgs", 2000, 1e-4)
            assert kwargs["fit_intercept"] and not kwargs["warm_start"]


def test_rank_relative_to_max_and_tie_breaks():
    scores = dict.fromkeys(CANDIDATES, 0.0)
    scores.update({"lr-0.1-unweighted": 0.79899, "lr-0.1-balanced": 0.799,
                   "lr-1-unweighted": 0.7995, "lr-10-balanced": 0.8})
    assert rank_lr(scores) == (
        0.8, ["lr-0.1-balanced", "lr-1-unweighted", "lr-10-balanced"], "lr-0.1-balanced",
    )
    scores["lr-0.1-unweighted"] = 0.799
    assert rank_lr(scores)[2] == "lr-0.1-unweighted"
    scores["lr-1-balanced"] = float("nan")
    with pytest.raises(ClassicalError, match="Six valid"):
        rank_lr(scores)


def test_join_reorders_and_rejects_misaligned_records(inputs):
    rows = inputs.selection_rows
    records = [{"row_id": r.row_id, "event_id": r.event_id, "y_true": r.y_true,
                "role": "model_selection", "classical_text": text}
               for r, text in zip(rows, inputs.selection_text, strict=True)]
    assert aligned_text(records[::-1], rows, "model_selection") == inputs.selection_text
    with pytest.raises(ClassicalError, match="membership"):
        aligned_text([*records[:-1], records[0]], rows, "model_selection")
    records[0]["y_true"] = 1
    with pytest.raises(ClassicalError, match="disagree"):
        aligned_text(records, rows, "model_selection")


def test_convergence_and_logit_failures(inputs, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    _, train, selection = fit_features(inputs.train_text, inputs.selection_text)
    candidate = CANDIDATES["lr-0.1-unweighted"]
    version = classical_run._runtime()["scikit-learn"]
    model, iterations = fit_lr(candidate, version, train, inputs.train_labels)
    assert iterations and decision_scores(model, selection).shape == (10, 10)
    model.classes_ = model.classes_[::-1]
    with pytest.raises(ClassicalError, match="class"):
        checked_iterations(model)
    model.classes_ = np.arange(len(LABELS))
    model.n_iter_[0] = 2000
    with pytest.raises(ClassicalError, match="convergence"):
        checked_iterations(model)
    monkeypatch.setattr(model, "decision_function", lambda _: np.full((10, 10), 1e100))
    with pytest.raises(ClassicalError, match="finite"):
        decision_scores(model, selection)

    def warn(*args):
        warnings.warn("invented warning", ConvergenceWarning)

    monkeypatch.setattr(LogisticRegression, "fit", warn)
    with pytest.raises(ClassicalError, match="did not converge"):
        fit_lr(candidate, version, train, inputs.train_labels)


def test_final_test_reader_and_bundle_transport_still_gated(tmp_path, inputs):
    with pytest.raises(ValueError, match="Slice 8"):
        FrozenDataset.load_role(object(), "final_test")
    digest = sha256_bytes(b"invented")
    expected = BundleExpectation("synthetic", "final_test", inputs.selection_rows, {
        "archive_sha256": digest, "eligible_manifest_sha256": digest,
        "config_sha256": digest, "code_sha256": digest,
        "role_manifest_sha256": sha256_bytes(role_bytes(inputs.selection_rows)),
    })
    with pytest.raises(ValueError, match="Slice 8"):
        read_bundle(tmp_path, expected=expected)
    with pytest.raises(ValueError, match="Slice 8"):
        write_bundle(tmp_path, expected=expected, run_id="fixture", model_family="majority",
                     seed=None, arrays={})


def test_installed_campaign_runs_resumes_replays_and_rejects_corruption(tmp_path, inputs, before_cutoff):
    root = tmp_path / "artifacts"
    with classical_run._campaign_lock(root):
        campaign = classical_run.Campaign(root, "invented", inputs, classical_run._runtime())
        campaign.open(SOURCE_ROOT)
        result = campaign.run()
        assert result["status"] == "completed"
        assert len(budget_summary(root)["attempts"]) == 8
        selection = json.loads((campaign.directory / "selection.json").read_bytes())
        assert [entry["candidate"] for entry in selection["comparison"]] == list(classical_run.RUNS)
        lr_scores = {row["candidate"]: row["pooled_macro_f1"] for row in selection["comparison"][1:]}
        maximum = max(lr_scores.values())
        expected = next(key for key, score in lr_scores.items() if score >= maximum - 0.001)
        assert selection["winner"] == expected

        # Simulate death after closing one attempt but before publishing its directory.
        majority = campaign.directory / "majority"
        stage = json.loads((majority / "stage.json").read_bytes())
        majority.rename(campaign.directory / f".majority.{stage['attempt_id']}")
        assert campaign.run() == result
        assert len(budget_summary(root)["attempts"]) == 8
        replay = campaign.verify()
        assert replay["status"] == "verified" and replay["predictions_replayed"] == 7
        assert replay["selection_sha256"] == result["selection_sha256"]
        assert len(budget_summary(root)["attempts"]) == 9

        (campaign.directory / "lr-10-balanced/classifier.joblib").write_bytes(b"corrupt invented weights")
        with pytest.raises(ClassicalError, match="artifact verification"):
            campaign.verify()
        assert len(budget_summary(root)["attempts"]) == 9


def test_interrupted_stage_and_exclusive_lock(tmp_path, inputs, monkeypatch, before_cutoff):
    root = tmp_path / "artifacts"
    with classical_run._campaign_lock(root):
        with pytest.raises(ClassicalError, match="already active"):
            with classical_run._campaign_lock(root):
                pytest.fail("concurrent campaign acquired an exclusive lock")
        campaign = classical_run.Campaign(root, "interruption", inputs, classical_run._runtime())
        campaign.open(SOURCE_ROOT)

        def interrupt(*args):
            raise classical_run.DeadlineReached()

        monkeypatch.setattr(classical_run, "_supervise", interrupt)
        with pytest.raises(classical_run.DeadlineReached):
            campaign.stage("example", (), None, ())
        summary = budget_summary(root)
        assert [row["status"] for row in summary["attempts"]] == ["interrupted"]
        assert not summary["unresolved_attempts"]
        assert not list(campaign.directory.glob("*example*"))



def test_campaign_observes_cutoff_before_start(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(classical_run, "utc_now", lambda: datetime.fromisoformat("2026-11-24T00:00:00+05:30"))
    root = tmp_path / "artifacts"
    with classical_run._campaign_lock(root):
        campaign = classical_run.Campaign(root, "after-cutoff", inputs, classical_run._runtime())
        campaign.open(SOURCE_ROOT)
        with pytest.raises(RuntimeError, match="training cutoff reached"):
            campaign.stage("example", (), None, ())
        assert not budget_summary(root)["attempts"]
