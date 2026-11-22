"""Record joint approval against the accepted Slice 5–7 exact-byte inputs."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from disaster_triage.accounting import budget_summary, identity
from disaster_triage.dataset import _json_load
from disaster_triage.evaluation_lock import EvaluationError, RUN_SCOPE, SelectionLock, checked_json, private_at
from disaster_triage.provenance import save_json, sha256_bytes
from disaster_triage.transformer_data import FREEZE_ID, FREEZE_SHA256

CLASSICAL = Path("artifacts/classical/slice-5-2026-11-19")
TRANSFORMER = Path("artifacts/transformer/slice-6-2026-11-20")
EVIDENCE = Path("artifacts/evidence/slice-7-2026-11-21")
FREEZE = Path("data/processed/freezes") / FREEZE_ID

CLASSICAL_ACCEPTANCE = "af60f6f56b1bbe154bda7e8be0e94ef911c23e3e462dab7d80459bc93db5167c"
REPEAT_ACCEPTANCE = "aa698fac02b1f9041fe0019ad857cd757b971a8575f9f801e27a41c1596f194f"
EVIDENCE_ACCEPTANCE = "b63a97b81757139c909c9cb0d0f70a50ac497f431176b5cbcc82860289c61edc"


def build_lock(root, campaign, ledger_reference, review):
    preserve = {}

    def record(path, digest):
        content = private_at(root, path).read_bytes()
        if sha256_bytes(content) != digest:
            raise EvaluationError("accepted input digest mismatch")
        preserve[str(path)] = digest
        return _json_load(content)

    classical = record(CLASSICAL / "acceptance.json", CLASSICAL_ACCEPTANCE)
    repeat = record(TRANSFORMER / "returns/repeat-acceptance.json", REPEAT_ACCEPTANCE)
    evidence = record("artifacts/evidence-acceptance/slice-7-2026-11-21/acceptance.json", EVIDENCE_ACCEPTANCE)
    selection = record(CLASSICAL / "selection.json", classical["selection_sha256"])
    handoff = record(TRANSFORMER / "slice-8-handoff.json", repeat["slice_8_handoff_sha256"])
    summary = record(TRANSFORMER / "development-summary.json", repeat["development_summary_sha256"])
    manifest = record(EVIDENCE / "manifest.json", evidence["output_manifest_sha256"])
    frozen = checked_json(private_at(root, FREEZE / "freeze.json"), FREEZE_SHA256)
    ledger = checked_json(ledger_reference, evidence["ledger_inventory_sha256"])
    actual = budget_summary(root / "artifacts")
    if (classical["status"] != "accepted" or repeat["status"] != "accepted"
            or actual != ledger["budget"] or len(ledger["ledger"]) != 30
            or actual["unresolved_attempts"] or any(item["status"] != "completed" for item in actual["attempts"])
            or ledger["handoff_sha256"] != evidence["slice_8_handoff_sha256"]
            or not summary["full_schedule_complete"] or handoff["scope"] != "joint_selection_pending"
            or handoff["selection"] != repeat["selection"]
            or handoff["development_summary_sha256"] != repeat["development_summary_sha256"]
            or selection["winner"] != repeat["lr"]["winner"] or selection["winner"] != "lr-10-balanced"
            or selection["winner_macro_f1"] != repeat["lr"]["macro_f1"]
            or selection["lineage"]["freeze_id"] != summary["freeze_id"]
            or summary["freeze_id"] != manifest["freeze_id"] or manifest["freeze_id"] != FREEZE_ID
            or manifest["selected_lr"] != selection["winner"]):
        raise EvaluationError("accepted development records, frozen roles or preflight ledger disagree")
    preserve.update(ledger["ledger"])
    for name, sha256 in manifest["outputs"].items():
        preserve[str(EVIDENCE / name)] = sha256
    for name, sha256 in manifest["inputs"].items():
        family, relative = name.split("/", maxsplit=1)
        source = {"classical": CLASSICAL, "transformer": TRANSFORMER}[family]
        preserve[str(source / relative)] = sha256
    preserve["artifacts/evidence-notebooks/slice-7-2026-11-21.executed.ipynb"] = evidence["executed_notebook_sha256"]
    for item in selection["comparison"]:
        run_path = CLASSICAL / item["candidate"]
        attempt_path = Path("artifacts/runs") / item["run_id"] / "attempts" / item["attempt_id"] / "attempt.json"
        attempt = record(attempt_path, ledger["ledger"][str(attempt_path)])
        run = record(run_path / "stage.json", attempt["artifact_sha256"]["stage"])
        if (run["files"]["selection/predictions.npz"] != item["bundle_sha256"]
                or run["files"]["resolved-config.json"] != item["config_sha256"]
                or run["files"]["report.json"] != item["report_sha256"]
                or run["files"].get("classifier.joblib") != item["classifier_sha256"]):
            raise EvaluationError("accepted classical stage does not match its selection record")
        preserve.update({str(run_path / name): sha256 for name, sha256 in run["files"].items()})

    chosen = {entry["candidate"]: entry for entry in selection["comparison"]}
    baseline = repeat["lr"]
    lr = chosen[selection["winner"]]
    if (lr["classifier_sha256"] != baseline["model_sha256"]
            or classical["vectorizer_sha256"] != baseline["vectorizer_sha256"]):
        raise EvaluationError("selected classical artifacts disagree across accepted receipts")
    majority = chosen["majority"]
    runs = {
        "majority": {"parents": {"training_config": [
            str(CLASSICAL / "majority/resolved-config.json"), majority["config_sha256"],
        ]}, "selection_macro_f1": majority["pooled_macro_f1"]},
        "lr": {"parents": {
            "training_config": [str(CLASSICAL / selection["winner"] / "resolved-config.json"), lr["config_sha256"]],
            "model": [baseline["model_path"], baseline["model_sha256"]],
            "vectorizer": [baseline["vectorizer_path"], baseline["vectorizer_sha256"]],
        }, "selection_macro_f1": lr["pooled_macro_f1"]},
    }
    for seed, model in repeat["seed_models"].items():
        runs[f"seed{seed}"] = {
            "parents": {
                "training_config": [model["config_path"], model["config_sha256"]],
                "checkpoint": [model["checkpoint_path"], model["checkpoint_sha256"]],
                "selection_config": [model["evaluation_config_path"], model["evaluation_config_sha256"]],
                "selection_report": [model["development_report_path"], model["development_report_sha256"]],
            },
            "epoch": model["epoch"], "steps": repeat["selection"]["scheduler_total_steps"],
            "selection_macro_f1": model["pooled_macro_f1"],
        }
    for key, (family, seed) in RUN_SCOPE.items():
        runs[key].update(model_family=family, seed=seed)
    rate = mean(runs[f"seed{seed}"]["selection_macro_f1"] for seed in (42, 43, 44))
    lr_score = lr["pooled_macro_f1"]
    if rate != summary["cohort"]["pooled"]["mean"] or rate != repeat["cohort_pooled"]["mean"]:
        raise EvaluationError("pooled selection scores do not reproduce the declared three-seed cohort")

    counts = frozen["summary"]["training_counts"]
    return {
        "schema_version": 1, "mode": "research", "campaign": campaign, "review": review,
        "freeze": {"path": str(FREEZE), "sha256": FREEZE_SHA256},
        "majority_label": max(range(len(counts)), key=counts.__getitem__),
        "selection": {"primary": "distilbert" if rate - lr_score > .001 else "tfidf_lr",
                      "tie_tolerance": .001, "distilbert_mean": rate, "lr": lr_score, "margin": rate - lr_score},
        "runs": runs, "preserve": preserve,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--ledger-reference", type=Path, required=True)
    parser.add_argument("--reviewer", action="append", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--approval-source", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    campaign = identity(args.campaign_id)
    review = {"reviewers": args.reviewer, "source": args.approval_source, "date": args.date}
    value = build_lock(root, campaign, args.ledger_reference, review)
    target = private_at(root, f"artifacts/evaluation/{campaign}/selection-lock.json")
    if target.exists():
        raise EvaluationError("selection lock already exists")
    # Validate the entire candidate lock before publishing it at its permanent campaign identity.
    with TemporaryDirectory(prefix=".lock-", dir=target.parent.parent) as temp:
        candidate = Path(temp) / target.name
        sha256 = save_json(candidate, value)
        SelectionLock(root, campaign, sha256, candidate=candidate)
        if target.exists():
            raise EvaluationError("selection lock already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(candidate, target)
    print(json.dumps({"selection_lock_sha256": sha256, "primary": value["selection"]["primary"],
                      "majority_label": value["majority_label"], "preserved_paths": len(value["preserve"])}))


if __name__ == "__main__":
    main()
