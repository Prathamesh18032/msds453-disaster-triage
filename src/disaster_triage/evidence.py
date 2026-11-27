"""Publish reproducible CSVs and figures from verified study aggregates."""

import csv
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .constants import LABELS
from .evidence_inputs import EvidenceError, ROLE, _correct, load_evidence
from .provenance import save_json, sha256_file

MODE_LABELS = {
    "synthetic": "Synthetic", "development": "Development — model selection",
    "final": "Held-out test events",
}


def _output_id(output_id):
    if not isinstance(output_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", output_id):
        raise EvidenceError("Output ID must contain 1–80 letters, digits, hyphens, or underscores.")
    return output_id


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _scopes(report, events):
    yield "pooled", report["pooled"]
    yield from ((event, report["events"][event]) for event in events)


def _tables(directory, evidence):
    reference_rows = {scope: metrics["n_rows"] for scope, metrics
                      in _scopes(evidence.comparison[0].report, evidence.events)}

    def comparisons():
        for candidate in evidence.comparison:
            for scope, metrics in _scopes(candidate.report, evidence.events):
                yield (scope, metrics["n_rows"], candidate.bundle["model_family"], candidate.key,
                       candidate.bundle["seed"], metrics["macro_f1"], "", "")
        cohort = evidence.cohort
        for scope, stats in _scopes(cohort, evidence.events):
            yield (scope, reference_rows[scope], "distilbert", "cohort_mean", "", stats["mean"],
                   stats["sample_sd"], cohort["n_seeds"])

    def class_metrics():
        for candidate in evidence.candidates:
            for scope, metrics in _scopes(candidate.report, evidence.events):
                for label in LABELS:
                    cell = metrics["classes"][label]
                    yield (candidate.stage, candidate.key, candidate.epoch, scope, label,
                           cell["precision"] if cell["predicted_count"] else "N/A",
                           cell["recall"] if cell["support"] else "N/A",
                           cell["f1"], cell["support"], cell["predicted_count"])

    _write_csv(directory / "candidate-scores.csv",
               ("stage", "candidate", "model_family", "run_id", "config_sha256", "seed",
                "C", "class_weight", "learning_rate", "epoch", "selected", "pooled_macro_f1"),
               ((c.stage, c.key, c.bundle["model_family"], c.bundle["run_id"], c.bundle["config_sha256"],
                 c.bundle["seed"], c.c, c.class_weight, c.learning_rate, c.epoch,
                 str(c.selected).lower(), c.report["pooled"]["macro_f1"]) for c in evidence.candidates))
    _write_csv(directory / "model-comparison.csv",
               ("scope", "n_rows", "model_family", "candidate", "seed", "macro_f1", "sample_sd", "n_seeds"),
               comparisons())
    _write_csv(directory / "class-metrics.csv",
               ("stage", "candidate", "epoch", "scope", "label", "precision", "recall", "f1",
                "support", "predicted_count"), class_metrics())


def _verify_existing(target, expected):
    if target.is_dir() and not target.is_symlink() and {p.name for p in target.iterdir()} == expected.keys():
        paths = {target / name: digest for name, digest in expected.items()}
        if all(path.is_file() and not path.is_symlink() and sha256_file(path) == digest
               for path, digest in paths.items()):
            return
    raise EvidenceError("Output ID already contains different evidence. Choose a new output ID.")


def _final_tables(directory, evidence):
    from .final_evidence import EVENT_ORDER, TRAINED, metrics

    scopes = ("pooled", *EVENT_ORDER)
    reports = evidence.reports
    common = evidence.common_labels
    comparison = []
    classes = []
    event_support = []
    calibration = []
    review = []
    for scope in scopes:
        for key, report in reports.items():
            score = metrics(report, scope)
            comparison.append((scope, score["n_rows"], key, report["bundle"]["seed"],
                               score["macro_f1"], "", ""))
            for label in LABELS:
                cell = score["classes"][label]
                correct = _correct(cell)
                classes.append((scope, key, label, cell["support"], cell["predicted_count"], correct,
                                cell["predicted_count"] - correct, cell["support"] - correct,
                                *("N/A" if cell[name] is None else cell[name]
                                  for name in ("precision", "recall", "f1"))))
        stats = evidence.scores(scope)
        reference = metrics(reports["majority"], scope)
        comparison.append((scope, reference["n_rows"], "cohort_mean", "", stats["mean"],
                           stats["sample_sd"], evidence.cohort["n_seeds"]))
        if scope != "pooled":
            common_score = evidence.common_score("cohort_mean", scope)
            event_support.extend((scope, label, reference["classes"][label]["support"], label in common,
                                  stats["mean"], common_score)
                                 for label in LABELS)
    for key in TRAINED:
        report = reports[key]
        for variant, scores in report["calibration"].items():
            calibration.append((key, variant, report["pooled"]["n_rows"], scores["nll"], scores["ece"]))
        for scenario in report["review"]:
            for scope in scopes:
                cell = metrics(scenario, scope)
                requests = cell["requests"]
                review.append((scope, key, scenario["review_percent"], scenario["threshold"],
                               cell["accepted"], cell["deferred"], cell["coverage"], cell["review_fraction"],
                               cell["wrong_accepted"], cell["selective_error"], requests["support"],
                               requests["deferred"], requests["deferred_fraction"],
                               requests["accepted_false_negatives"], requests["accepted_false_positives"]))
    _write_csv(directory / "model-comparison.csv",
               ("scope", "n_rows", "predictor", "seed", "macro_f1", "sample_sd", "n_seeds"), comparison)
    _write_csv(directory / "class-metrics.csv",
               ("scope", "predictor", "label", "support", "predicted", "true_positives", "false_positives",
                "false_negatives", "precision", "recall", "f1"), classes)
    _write_csv(directory / "event-support.csv",
               ("event", "label", "support", "shared_label", "distilbert_mean_fixed_ten",
                "distilbert_mean_common_labels"), event_support)
    _write_csv(directory / "calibration.csv", ("predictor", "probabilities", "n_rows", "nll", "ece_15_bins"),
               calibration)
    _write_csv(directory / "review.csv",
               ("scope", "predictor", "nominal_review_percent", "threshold", "accepted", "deferred", "coverage",
                "actual_review_fraction", "wrong_accepted", "selective_error", "true_requests", "deferred_requests",
                "true_request_deferred_fraction", "accepted_request_false_negatives",
                "accepted_request_false_positives"), review)

    (directory / "evidence-index.md").write_text(
        "# Held-out evidence index\n\n"
        "| Finding | Figure | Source table |\n|---|---|---|\n"
        "| Pooled, event and seed comparisons | 4 | model-comparison.csv |\n"
        "| Event support and exploratory common-label ordering | 5 | event-support.csv |\n"
        "| Request and missing/found recall and false positives | 6 | class-metrics.csv |\n"
        "| Simulated review workload and accepted errors | 7 | review.csv |\n"
        "| Calibration before and after temperature scaling | — | calibration.csv |\n",
        encoding="utf-8",
    )


def _publish(staged, target, digests):
    if target.exists() or target.is_symlink():
        _verify_existing(target, digests)
        return
    try:
        staged.rename(target)
    except OSError:
        if not target.exists() and not target.is_symlink():
            raise
        # A concurrent publisher may have completed after the existence check.
        _verify_existing(target, digests)


def render_evidence(
    root, output_id, *, mode="synthetic", selection_path=None, selection_sha256=None,
    summary_path=None, summary_sha256=None, final_manifest_path=None, final_manifest_sha256=None,
):
    """Render into an ignored output directory, or verify an existing identical run."""
    _output_id(output_id)
    inputs = (selection_path, selection_sha256, summary_path, summary_sha256)
    final_inputs = (final_manifest_path, final_manifest_sha256)
    if mode != "final" and any(value is not None for value in final_inputs):
        raise EvidenceError("Final manifest arguments require final mode.")
    if mode == "synthetic":
        if any(value is not None for value in inputs):
            raise EvidenceError("Synthetic mode uses built-in invented aggregates.")
        from .evidence_fixture import synthetic_inputs

        with TemporaryDirectory(prefix="triage-evidence-fixture-") as temporary:
            evidence = load_evidence(mode, *synthetic_inputs(Path(temporary)))
    elif mode == "development":
        if any(value is None for value in inputs):
            raise EvidenceError("Development mode requires both aggregate paths and their SHA-256 digests.")
        evidence = load_evidence(mode, *inputs)
    elif mode == "final":
        if any(value is not None for value in inputs) or any(value is None for value in final_inputs):
            raise EvidenceError("Final mode requires only the final manifest path and its SHA-256 digest.")
        from .final_evidence import load_final_evidence

        evidence = load_final_evidence(*final_inputs)
    else:
        raise EvidenceError("Evidence mode must be synthetic, development, or final.")

    artifacts = Path(root).resolve() / "artifacts"
    parent = artifacts / "evidence"
    if artifacts.is_symlink() or parent.is_symlink():
        raise EvidenceError("Evidence storage must be a local artifacts/evidence directory.")
    target = parent / output_id
    from .evidence_figures import render_figures

    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=parent, prefix=".pending-") as temporary:
        staged = Path(temporary) / output_id
        staged.mkdir()
        if mode == "final":
            from .final_figures import FIGURES

            _final_tables(staged, evidence)
            plot_version = render_figures(staged, evidence, MODE_LABELS[mode], figures=FIGURES)
            scope = evidence.manifest["role"]
            lineage = {
                "campaign": evidence.manifest["campaign"], "role": scope["role"],
                "role_manifest_sha256": scope["role_manifest_sha256"],
                "runs": [{"candidate": key, "run_id": report["bundle"]["run_id"],
                          "seed": report["bundle"]["seed"], "config_sha256": report["bundle"]["config_sha256"]}
                         for key, report in evidence.reports.items()],
            }
        else:
            _tables(staged, evidence)
            plot_version = render_figures(staged, evidence, MODE_LABELS[mode])
            lineage = {
                "role": ROLE, "freeze_id": evidence.freeze_id,
                "role_manifest_sha256": evidence.scope["role_manifest_sha256"],
                "selected_lr": evidence.comparison[1].key, "distilbert_recipe": evidence.recipe,
                "runs": [{"candidate": c.key, "epoch": c.epoch, "run_id": c.bundle["run_id"],
                          "config_sha256": c.bundle["config_sha256"]} for c in evidence.candidates],
            }
        manifest = {
            "schema_version": 1, "mode": mode, "matplotlib": plot_version, "inputs": evidence.hashes,
            "outputs": {path.name: sha256_file(path) for path in sorted(staged.iterdir())},
            **lineage,
        }
        manifest_sha = save_json(staged / "manifest.json", manifest)
        _publish(staged, target, {**manifest["outputs"], "manifest.json": manifest_sha})
    return {"output": str(target), "manifest_sha256": manifest_sha}
