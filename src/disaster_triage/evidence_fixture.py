"""Invented evaluator inputs for the default evidence walkthrough."""

import numpy as np

from .bundles import BundleExpectation, RoleRow, role_bytes, write_bundle
from .constants import LABELS
from .evaluation import BundleInput, SeedCohort, score_bundle, summarize_seeds
from .evidence_inputs import CLASSICAL_GRID, MISSING, REPEATS, REQUESTS, ROLE, SCREEN_RATES, SYNTHETIC_EVENTS
from .provenance import json_bytes, save_json, sha256_bytes


def synthetic_inputs(root):
    """Write producer-shaped fixtures from the same evaluator used for development."""
    rows = tuple(
        RoleRow(f"invented-{event}-{i}", event, label)
        for event in SYNTHETIC_EVENTS
        for i, label in enumerate(
            [index for index in range(len(LABELS))
             if event == SYNTHETIC_EVENTS[0] or LABELS[index] != MISSING]
            + [LABELS.index(REQUESTS), *([LABELS.index(MISSING)] if event == SYNTHETIC_EVENTS[0] else [])]
        )
    )
    freeze_id = sha256_bytes(b"invented freeze")
    data_digests = {
        "archive_sha256": sha256_bytes(b"invented archive"),
        "eligible_manifest_sha256": sha256_bytes(b"invented eligible manifest"),
        "role_manifest_sha256": sha256_bytes(role_bytes(rows)),
    }
    truth = np.array([row.y_true for row in rows], dtype=np.int64)

    def report(key, family, seed, epoch, mistakes, offset=0):
        run_id = f"invented-{key}"
        expected = BundleExpectation("synthetic", ROLE, rows, {
            **data_digests,
            "config_sha256": sha256_bytes(json_bytes([key, epoch])),
            "code_sha256": sha256_bytes(b"invented producer"),
        })
        arrays = {
            "row_id": np.array([row.row_id for row in rows], dtype="U"),
            "event_id": np.array([row.event_id for row in rows], dtype="U"),
            "y_true": truth,
        }
        if family == "majority":
            arrays["y_pred"] = np.full_like(truth, LABELS.index(REQUESTS))
        else:
            positions = (np.arange(mistakes) + offset) % len(rows)
            predicted = truth.copy()
            predicted[positions] = (predicted[positions] + 1) % len(LABELS)
            logits = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
            logits[np.arange(len(rows)), predicted] = 3.0
            arrays["logits"] = logits
        directory = root / "bundles" / key / str(epoch)
        write_bundle(directory, expected=expected, run_id=run_id, model_family=family, seed=seed, arrays=arrays)
        return score_bundle(BundleInput(directory, expected, run_id, family, seed))

    comparison = []
    classical = root / "classical" / "selection.json"
    for i, key in enumerate(("majority", *CLASSICAL_GRID)):
        aggregate = report(key, "tfidf_lr" if i else "majority", None, None, 14 - i, i)
        bundle = aggregate["bundle"]
        name = f"{key}/report.json"
        comparison.append({
            "candidate": key, "run_id": bundle["run_id"], "config_sha256": bundle["config_sha256"],
            "bundle_sha256": bundle["payload_sha256"], "pooled_macro_f1": aggregate["pooled"]["macro_f1"],
            "report": name, "report_sha256": save_json(classical.parent / name, aggregate),
        })
    winner = max(comparison[1:], key=lambda ref: ref["pooled_macro_f1"])
    classical_sha = save_json(classical, {
        "schema_version": 1, "role": ROLE,
        "lineage": {**data_digests, "freeze_id": freeze_id,
                    "role_manifest_sha256": {ROLE: data_digests["role_manifest_sha256"]}},
        "comparison": comparison, "winner": winner["candidate"], "winner_macro_f1": winner["pooled_macro_f1"],
    })
    reports = [
        {"key": key, "epoch": epoch, "aggregate": report(key, "distilbert", 42, epoch, errors)}
        for key, curve in zip(SCREEN_RATES, ((9, 6, 3), (10, 8, 7)), strict=True)
        for epoch, errors in enumerate(curve, 1)
    ]
    selected = reports[2]["aggregate"]
    recipe = {"learning_rate": 2e-5, "epoch": 3, "scheduler_total_steps": 12}
    for i, (key, seed) in enumerate(REPEATS.items(), 1):
        reports.append({"key": key, "epoch": 3,
                        "aggregate": report(key, "distilbert", seed, 3, 3 + i, i + 2)})
    members = [selected, *(ref["aggregate"] for ref in reports[-2:])]
    cohort = summarize_seeds(
        members, cohort=SeedCohort(
            sha256_bytes(json_bytes(recipe)), 3,
            {r["bundle"]["seed"]: r["bundle"]["config_sha256"] for r in members},
        ),
    )
    transformer = root / "transformer" / "development-summary.json"
    transformer_sha = save_json(transformer, {
        "schema_version": 1, "role": ROLE, "freeze_id": freeze_id, "full_schedule_complete": True,
        "reports": reports, "cohort": cohort,
        "selection": {**recipe, "key": reports[2]["key"],
                      "pooled_macro_f1": selected["pooled"]["macro_f1"],
                      "eval_config_sha256": selected["bundle"]["config_sha256"],
                      "checkpoint_sha256": sha256_bytes(b"invented checkpoint identity")},
    })
    return classical, classical_sha, transformer, transformer_sha
