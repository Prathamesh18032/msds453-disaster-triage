"""The final renderer consumes only synthetic, research-shaped aggregate files."""

from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pytest

from disaster_triage.constants import ARCHIVE_SHA256, LABELS
from disaster_triage.evaluation import _classification, _review
from disaster_triage.evidence import EvidenceError, render_evidence
from disaster_triage.final_evidence import EVENT_ORDER, RUNS, SEEDS, load_final_evidence
from disaster_triage.provenance import save_json

HEX = "0" * 64


@pytest.fixture
def aggregate_files(tmp_path):
    manifest_path = tmp_path / "input" / "manifest.json"
    scope = {"mode": "research", "role": "final_test", "archive_sha256": ARCHIVE_SHA256,
             "eligible_manifest_sha256": HEX, "role_manifest_sha256": HEX}
    manifest = {"schema_version": 1, "campaign": "invented", "mode": "research",
                "primary": "distilbert", "role": scope, "selection_lock_sha256": HEX,
                "readiness_sha256": HEX, "outputs": {}}
    truth = np.array([0, 4, 7, 0, 0, 7, 0, 1, 7])
    scopes = {"pooled": slice(None), **{event: slice(i * 3, i * 3 + 3) for i, event in enumerate(EVENT_ORDER)}}
    reports = {}
    for key, (family, seed) in RUNS.items():
        pred = np.zeros_like(truth) if key == "majority" else np.roll(truth, (seed or 0) % 4)
        report = {"schema_version": 1, "labels": list(LABELS), "bundle": {
            **scope, "run_id": f"invented-{key}", "model_family": family, "seed": seed,
            "code_sha256": HEX, "config_sha256": HEX, "payload_sha256": HEX,
        }}
        for name, portion in scopes.items():
            scores = _classification(truth[portion], pred[portion], display=True)
            if name == "pooled":
                report[name] = scores
            else:
                report.setdefault("events", {})[name] = scores
        if key != "majority":
            report["fitted_policy_sha256"] = HEX
            report["calibration"] = {"raw": {"nll": 1.2, "ece": .3}, "calibrated": {"nll": 1.1, "ece": .2}}
            report["review"] = []
            for i, percent in enumerate((0, 10, 20, 30)):
                accept = np.arange(len(truth)) >= i
                outcomes = {name: _review(truth[part], pred[part], accept[part])
                            for name, part in scopes.items()}
                pooled = outcomes.pop("pooled")
                report["review"].append({"review_percent": percent, "threshold": i / 10,
                                         "pooled": pooled, "events": outcomes})
        reports[key] = report
        manifest["outputs"][key] = {"payload_sha256": HEX, "report_sha256": save_json(
            manifest_path.parent / "final" / key / "report.json", report,
        )}

    def stats(name):
        scores = [(reports[key][name] if name == "pooled" else reports[key]["events"][name])["macro_f1"]
                  for key in SEEDS]
        return {"mean": mean(scores), "sample_sd": stdev(scores)}

    cohort = {"schema_version": 1, "scope": scope, "epoch": 3, "n_seeds": 3, "exploratory": False,
              "individual": [reports[key] for key in SEEDS], "pooled": stats("pooled"),
              "events": {event: stats(event) for event in EVENT_ORDER}}
    manifest["cohort_sha256"] = save_json(manifest_path.parent / "transformer-cohort.json", cohort)
    return manifest_path, save_json(manifest_path, manifest), manifest, cohort, reports


def test_final_renders_from_only_the_seven_aggregates(tmp_path, aggregate_files):
    pytest.importorskip("matplotlib")
    path, sha, _, _, _ = aggregate_files
    assert len([item for item in path.parent.rglob("*") if item.is_file()]) == 7
    options = {"mode": "final", "final_manifest_path": path, "final_manifest_sha256": sha}
    result = render_evidence(tmp_path, "final", **options)
    assert result == render_evidence(tmp_path, "final", **options)
    out = Path(result["output"])
    assert len(list(out.glob("*.csv"))) == 5
    assert len(list(out.glob("*.png"))) == len(list(out.glob("*.pdf"))) == 4
    assert "N/A" in (out / "class-metrics.csv").read_text()
    assert load_final_evidence(path, sha).common_labels == (LABELS[0], LABELS[7])
    (out / "class-metrics.csv").write_text("different\n")
    with pytest.raises(EvidenceError, match="already contains"):
        render_evidence(tmp_path, "final", **options)


@pytest.mark.parametrize("defect", ["wrong hash", "missing report", "different role", "duplicate seed", "missing seed"])
def test_invalid_final_evidence_leaves_no_output(tmp_path, aggregate_files, defect):
    path, sha, manifest, cohort, reports = aggregate_files
    if defect == "wrong hash":
        sha = HEX
    elif defect == "missing report":
        (path.parent / "final" / "lr" / "report.json").unlink()
    elif defect == "different role":
        report = deepcopy(reports["lr"])
        report["bundle"]["role"] = "model_selection"
        manifest["outputs"]["lr"]["report_sha256"] = save_json(path.parent / "final" / "lr" / "report.json", report)
        sha = save_json(path, manifest)
    else:
        if defect == "duplicate seed":
            cohort["individual"][2] = cohort["individual"][1]
        else:
            cohort["individual"].pop()
        manifest["cohort_sha256"] = save_json(path.parent / "transformer-cohort.json", cohort)
        sha = save_json(path, manifest)
    with pytest.raises(EvidenceError):
        render_evidence(tmp_path, "final", mode="final", final_manifest_path=path, final_manifest_sha256=sha)
    assert not (tmp_path / "artifacts" / "evidence").exists()


def test_final_rejects_mixed_or_incomplete_inputs(tmp_path, aggregate_files):
    path, sha, *_ = aggregate_files
    with pytest.raises(EvidenceError, match="requires only"):
        render_evidence(tmp_path, "mixed", mode="final", final_manifest_path=path, final_manifest_sha256=sha,
                        selection_path=path)
    with pytest.raises(EvidenceError, match="require final mode"):
        render_evidence(tmp_path, "mixed", final_manifest_path=path)
    with pytest.raises(EvidenceError, match="requires only"):
        render_evidence(tmp_path, "incomplete", mode="final", final_manifest_path=path)
