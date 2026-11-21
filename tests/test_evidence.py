import csv
import json

import pytest

from disaster_triage.cli import main
from disaster_triage.evidence import render_evidence
from disaster_triage.evidence_fixture import synthetic_inputs
from disaster_triage.evidence_inputs import EvidenceError, MISSING, load_evidence
from disaster_triage.provenance import save_json, sha256_file


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    return synthetic_inputs(tmp_path_factory.mktemp("synthetic-aggregates"))


@pytest.fixture
def editable_inputs(inputs, tmp_path):
    from shutil import copytree

    sel, sel_hash, summary, summary_hash = inputs
    classical = tmp_path / "classical"
    transformer = tmp_path / "transformer"
    copytree(sel.parent, classical)
    copytree(summary.parent, transformer)
    return [classical / sel.name, sel_hash, transformer / summary.name, summary_hash]


def _render_development(root, inputs):
    selection, selection_sha, summary, summary_sha = inputs
    return render_evidence(
        root, "evidence", mode="development", selection_path=selection, selection_sha256=selection_sha,
        summary_path=summary, summary_sha256=summary_sha,
    )


def test_synthetic_aggregates_from_evaluator(inputs):
    evidence = load_evidence("synthetic", *inputs)
    assert len(evidence.hashes) == 9
    assert len(evidence.candidates) == 15
    assert evidence.cohort["n_seeds"] == 3
    assert evidence.comparison[0].key == "majority"
    assert [c.bundle["seed"] for c in evidence.comparison[2:]] == [42, 43, 44]
    assert evidence.recipe["learning_rate"] == 2e-5


def test_invalid_hash_leaves_output_unpublished(tmp_path, inputs):
    with pytest.raises(EvidenceError, match="Classical selection SHA-256 mismatch"):
        _render_development(tmp_path, (inputs[0], "0" * 64, *inputs[2:]))
    assert not (tmp_path / "artifacts").exists()


def test_development_requires_explicit_inputs(tmp_path, capsys):
    assert main(["evidence", "render", "--mode", "development", "--output-id", "test",
                 "--root", str(tmp_path)]) == 1
    assert "requires both aggregate paths and their SHA-256 digests" in capsys.readouterr().err
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(("change", "diagnostic"), [
    ("missing_report", "missing or not a regular file"),
    ("report_hash", "Classical majority report SHA-256 mismatch"),
    ("path_traversal", "must name its own candidate report"),
    ("label_order", "canonical ten-label order"),
    ("duplicate_candidate", "six unique LR candidates"),
    ("report_scope", "model-selection freeze or row manifest"),
    ("event_scope", "unexpected model-selection events"),
    ("class_support", "Pooled class counts disagree"),
    ("freeze", "disagree on the frozen dataset"),
    ("wrong_role", "must describe model selection"),
    ("duplicate_seed", "Expected unique cohort seeds"),
    ("incomplete_seeds", "Expected unique cohort seeds"),
    ("cohort_mean", "Sealed cohort mean or sample SD"),
    ("cohort_event_sd", "Sealed cohort mean or sample SD"),
    ("inconsistent_member", "embedded epoch reports"),
    ("duplicate_epoch", "six screening points and two unique repeat endpoints"),
    ("malformed", "malformed or duplicate JSON"),
])
def test_rejects_invalid_evidence(editable_inputs, change, diagnostic):
    selection, _, summary, _ = editable_inputs
    sel = json.loads(selection.read_bytes())
    transformer = json.loads(summary.read_bytes())
    ref = sel["comparison"][0]
    report_path = selection.parent / ref["report"]
    report = json.loads(report_path.read_bytes())
    if change == "missing_report":
        report_path.unlink()
    elif change == "report_hash":
        report_path.write_bytes(b"{}")
    elif change == "path_traversal":
        ref["report"] = "../unrelated.json"
    elif change == "label_order":
        report["labels"].reverse()
        ref["report_sha256"] = save_json(report_path, report)
    elif change == "duplicate_candidate":
        sel["comparison"][1] = ref
    elif change == "report_scope":
        report["bundle"]["role_manifest_sha256"] = "0" * 64
        ref["report_sha256"] = save_json(report_path, report)
    elif change == "event_scope":
        report["events"].pop("synthetic_beta")
        ref["report_sha256"] = save_json(report_path, report)
    elif change == "class_support":
        classes = report["events"]["synthetic_alpha"]["classes"]
        classes["caution_and_advice"]["support"] -= 1
        classes[MISSING]["support"] += 1
        ref["report_sha256"] = save_json(report_path, report)
    elif change == "freeze":
        transformer["freeze_id"] = "0" * 64
    elif change == "wrong_role":
        transformer["role"] = "final_test"
    elif change == "duplicate_seed":
        transformer["cohort"]["individual"][1] = transformer["cohort"]["individual"][0]
    elif change == "incomplete_seeds":
        transformer["cohort"]["individual"].pop()
    elif change == "cohort_mean":
        transformer["cohort"]["pooled"]["mean"] = 0.1
    elif change == "cohort_event_sd":
        transformer["cohort"]["events"]["synthetic_beta"]["sample_sd"] = 0.99
    elif change == "inconsistent_member":
        transformer["cohort"]["individual"][1]["calibration"] = {}
    elif change == "duplicate_epoch":
        transformer["reports"][1] = transformer["reports"][0]

    summary_hash = save_json(summary, transformer)
    if change == "malformed":
        summary.write_bytes(b'{"schema_version":1,"schema_version":1}')
        summary_hash = sha256_file(summary)
    with pytest.raises(EvidenceError, match=diagnostic):
        load_evidence("synthetic", selection, save_json(selection, sel), summary, summary_hash)


def test_csv_undefined_metrics_and_repeatable_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    first = render_evidence(tmp_path, "evidence")
    output = tmp_path / "artifacts" / "evidence" / "evidence"
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert len(manifest["outputs"]) == 9
    assert all(sha256_file(output / name) == digest for name, digest in manifest["outputs"].items())
    assert first["manifest_sha256"] == sha256_file(output / "manifest.json")
    assert render_evidence(tmp_path, "evidence") == first
    with (output / "class-metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 450
    missing = next(r for r in rows if (r["candidate"], r["scope"], r["label"])
                   == ("majority", "synthetic_beta", MISSING))
    assert (missing["recall"], missing["precision"], missing["f1"]) == ("N/A", "N/A", "0.0")


def test_conflicting_output_is_preserved(tmp_path):
    pytest.importorskip("matplotlib")
    target = tmp_path / "artifacts" / "evidence" / "occupied"
    target.mkdir(parents=True)
    (target / "existing.txt").write_text("earlier evidence")
    with pytest.raises(EvidenceError, match="Choose a new output ID"):
        render_evidence(tmp_path, "occupied")
    assert [path.name for path in target.parent.iterdir()] == ["occupied"]
    assert (target / "existing.txt").read_text() == "earlier evidence"


def test_plot_failure_leaves_output_unpublished(tmp_path, monkeypatch):
    from disaster_triage import evidence_figures

    def stop(directory, *args):
        (directory / "partial.png").touch()
        raise OSError("simulated plot failure")

    monkeypatch.setattr(evidence_figures, "render_figures", stop)
    with pytest.raises(OSError, match="simulated plot failure"):
        render_evidence(tmp_path, "evidence")
    assert not list((tmp_path / "artifacts" / "evidence").iterdir())
