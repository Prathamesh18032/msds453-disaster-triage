"""Prepare or summarize the local Slice 9 private error review."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from disaster_triage.bundles import read_bundle
from disaster_triage.classical import ClassicalError
from disaster_triage.constants import EVENTS, LABELS
from disaster_triage.evaluation_campaign import EvaluationCampaign
from disaster_triage.evaluation_lock import EvaluationError, TRAINED, checked_json, private_at
from disaster_triage.evaluation_models import ordered_text
from disaster_triage.evidence import _publish, _write_csv
from disaster_triage.evidence_inputs import EvidenceError, MISSING, REQUESTS, _read
from disaster_triage.provenance import json_bytes, save_json, sha256_bytes, sha256_file

THEMES = (
    "clear model error", "overlapping labels", "context needed",
    "apparent annotation disagreement", "unclear",
)
MISSING_INDEX = LABELS.index(MISSING)
REQUEST_INDEX = LABELS.index(REQUESTS)
EVENT_ORDER = EVENTS["final_test"]


def matrix_rows(key, report, truth, pred, events):
    for scope in ("pooled", *EVENT_ORDER):
        selected = np.ones(len(truth), dtype=bool) if scope == "pooled" else events == scope
        matrix = np.bincount(
            len(LABELS) * truth[selected] + pred[selected], minlength=len(LABELS) ** 2,
        ).reshape(len(LABELS), len(LABELS))
        scores = report["pooled"] if scope == "pooled" else report["events"][scope]
        classes = [scores["classes"][label] for label in LABELS]
        if (matrix.sum() != scores["n_rows"]
                or not np.array_equal(matrix.sum(axis=1), [cell["support"] for cell in classes])
                or not np.array_equal(matrix.sum(axis=0), [cell["predicted_count"] for cell in classes])
                or not np.array_equal(matrix.diagonal(), [
                    round((cell["recall"] or 0) * cell["support"]) for cell in classes
                ])):
            raise EvaluationError("Confusion counts disagree with accepted report marginals.")
        for i, label in enumerate(LABELS):
            for j, prediction in enumerate(LABELS):
                yield scope, key, label, prediction, int(matrix[i, j])


def select_cases(row_ids, truth, pred, events, manifest_sha):
    groups = [
        ("missing/found miss", 6, (truth == MISSING_INDEX) & (pred != MISSING_INDEX)),
        ("missing/found false positive", 6, (truth != MISSING_INDEX) & (pred == MISSING_INDEX)),
    ]
    for event in EVENT_ORDER:
        groups.extend((
            (f"{event} request miss", 2, (events == event) & (truth == REQUEST_INDEX) & (pred != REQUEST_INDEX)),
            (f"{event} request false positive", 2,
             (events == event) & (truth != REQUEST_INDEX) & (pred == REQUEST_INDEX)),
        ))
    picked, counts, selected = [], [], set()
    for group, quota, mask in groups:
        order = sorted(np.flatnonzero(mask).tolist(), key=lambda i: sha256_bytes(
            json_bytes([manifest_sha, str(row_ids[i])]),
        ))
        available = [i for i in order if i not in selected]
        chosen = available[:quota]
        picked.extend((group, i) for i in chosen)
        selected.update(chosen)
        counts.append({"group": group, "quota": quota, "available": len(order),
                       "available_after_deduplication": len(available), "selected": len(chosen)})
    return picked, counts


def prepare(args):
    base = private_at(args.root, f"artifacts/evaluation/{args.campaign_id}")
    manifest = checked_json(base / "manifest.json", args.manifest_sha256)
    if (manifest["selection_lock_sha256"], manifest["readiness_sha256"]) != (
        args.selection_lock_sha256, args.readiness_sha256,
    ):
        raise EvaluationError("The expected lock and readiness digests must match the accepted manifest.")
    campaign = EvaluationCampaign(args.root, args.campaign_id, args.selection_lock_sha256)
    probe, _, access = campaign.final_context(args.readiness_sha256)
    if (manifest["campaign"] != campaign.id or manifest["role"] != campaign.role_scope("final_test")
            or manifest["mode"] != "research"):
        raise EvaluationError("Accepted manifest must match the locked research campaign and held-out role.")
    combined = []
    for key in TRAINED:
        view, receipt, receipt_sha, bundle = campaign.published(
            "final", key, "final_test", campaign.final_target(key), probe, access=access,
            readiness_sha256=args.readiness_sha256,
        )
        ref = {"report_sha256": receipt["report_sha256"],
               **campaign.bundle_ref(receipt, receipt_sha, bundle["payload_sha256"])}
        if ref != manifest["outputs"][key]:
            raise EvaluationError("Saved bundle reference differs from the accepted final manifest.")
        _, arrays = read_bundle(view.directory, expected=view.expected, final_access=access)
        truth, pred, events = arrays["y_true"], arrays["logits"].argmax(axis=1), arrays["event_id"]
        report = checked_json(campaign.final_target(key) / "report.json", ref["report_sha256"])
        combined.extend(matrix_rows(key, report, truth, pred, events))
        if key == "seed42":
            sample, counts = select_cases(arrays["row_id"], truth, pred, events, args.manifest_sha256)
            labels = [(LABELS[truth[i]], LABELS[pred[i]]) for _, i in sample]

    text = ordered_text(campaign.frozen.load_final_role(access), campaign.rows("final_test"),
                        "final_test", "distilbert")
    # Only the selected, model-view text leaves memory, in an ignored offline worksheet.
    cases = [{"case": case, "group": group, "true": truth, "predicted": pred, "text": text[i]}
             for case, ((group, i), (truth, pred)) in enumerate(zip(sample, labels, strict=True), 1)]
    return cases, counts, combined


def worksheet(cases, sample_sha):
    payload = json.dumps({"cases": cases, "sample_sha256": sample_sha, "themes": THEMES}, ensure_ascii=True)
    return HTML.replace("__PAYLOAD__", payload.replace("<", "\\u003c").replace(">", "\\u003e")
                        .replace("&", "\\u0026"))


HTML = """<!DOCTYPE html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'">
<title>Private held-out error review</title>
<style>body{margin:2rem auto;padding:0 1rem;max-width:58rem;font:16px/1.5 system-ui;color:#172635}
article{border:1px solid #ccd5df;padding:1rem;margin-block:1rem;border-radius:5px}
.message{white-space:pre-wrap;overflow-wrap:anywhere}select,input,button{font:inherit;padding:.3rem;margin:.25rem}</style>
<h1>Private held-out error review</h1>
<p>For each case, pick one category based on the supplied label, prediction, and text.
Use “unclear” when the evidence is insufficient. The selected cases are a targeted sample.
Open this worksheet locally and keep it private.</p>
<label>Reviewer <input id="reviewer"></label>
<label>Review date <input type="date" id="date"></label>
<main id="cases"></main>
<button id="save">Download decisions</button>
<p id="status" role="status"></p>
<script>
const payload = __PAYLOAD__;
const get = id => document.getElementById(id);
function append(parent, tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  parent.append(node);
  return node;
}
const fields = payload.cases.map(item => {
  const card = append(get("cases"), "article", "");
  append(card, "h2", "Case " + item.case);
  append(card, "p", item.group);
  append(card, "p", "Supplied: " + item.true + " | Predicted: " + item.predicted);
  append(card, "p", item.text).className = "message";
  const field = append(card, "select", "");
  field.setAttribute("aria-label", "Category for case " + item.case);
  field.add(new Option("Choose a category", ""));
  payload.themes.forEach(theme => field.add(new Option(theme, theme)));
  return field;
});
get("save").onclick = () => {
  const reviewer = get("reviewer").value.trim();
  const reviewed_at = get("date").value;
  if (!reviewer || !reviewed_at || fields.some(field => !field.value)) {
    get("status").textContent = "Enter the reviewer, date, and a category for each case.";
    return;
  }
  const rows = payload.cases.map((item, i) => ({case: item.case, theme: fields[i].value}));
  const data = {sample_sha256: payload.sample_sha256, reviewer, reviewed_at, rows};
  const url = URL.createObjectURL(new Blob([JSON.stringify(data)], {type: "application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "slice-9-review-decisions.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  get("status").textContent = "Decisions downloaded. Keep the file local.";
};
</script></html>
"""


def summarize(sample, decisions):
    if (set(decisions) != {"sample_sha256", "reviewer", "reviewed_at", "rows"}
            or decisions["sample_sha256"] != sha256_bytes(json_bytes(sample))
            or not isinstance(decisions["reviewer"], str) or not decisions["reviewer"].strip()
            or not isinstance(decisions["reviewed_at"], str) or not decisions["reviewed_at"].strip()
            or len(decisions["rows"]) != len(sample["cases"])):
        raise EvidenceError("Decisions must cover the complete, matching private sample.")
    by_group = {group["group"]: Counter() for group in sample["groups"]}
    for case, decision in zip(sample["cases"], decisions["rows"], strict=True):
        if (set(decision) != {"case", "theme"} or type(decision["case"]) is not int
                or case["case"] != decision["case"] or decision["theme"] not in THEMES):
            raise EvidenceError("Expected one valid review category for each case in worksheet order.")
        by_group[case["group"]][decision["theme"]] += 1
    return {
        "sample_sha256": decisions["sample_sha256"], "campaign_manifest_sha256": sample["campaign_manifest_sha256"],
        "reviewed": len(sample["cases"]),
        "groups": [{**group, "reviewed": by_group[group["group"]].total(),
                    "themes": dict(by_group[group["group"]])} for group in sample["groups"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--selection-lock-sha256", required=True)
    parser.add_argument("--readiness-sha256", required=True)
    parser.add_argument("--decisions", type=Path, help="Summarize a completed local worksheet")
    args = parser.parse_args()
    args.root = args.root.resolve()
    try:
        target = private_at(args.root, f"artifacts/evidence/{args.campaign_id}-error-review")
        cases, groups, confusion = prepare(args)
        sample = {"campaign_manifest_sha256": args.manifest_sha256, "groups": groups,
                  "cases": [{name: case[name] for name in ("case", "group")} for case in cases]}
        # Bind the reviewed text and label pair inventory without exposing them in the summary.
        sample["case_content_sha256"] = sha256_bytes(json_bytes(cases))
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent, prefix=".review-") as directory:
            staged = Path(directory) / target.name
            staged.mkdir(mode=0o700)
            sample_sha = save_json(staged / "sample.json", sample)
            _write_csv(staged / "confusion-counts.csv",
                       ("scope", "predictor", "true_label", "predicted_label", "count"), confusion)
            (staged / "review.html").write_text(worksheet(cases, sample_sha), encoding="utf-8")
            _publish(staged, target, {file.name: sha256_file(file) for file in staged.iterdir()})
        if args.decisions:
            decisions = _read(args.decisions, sha256_file(args.decisions), "Review decisions")
            result = summarize(sample, decisions)
            result["confusion_sha256"] = sha256_file(target / "confusion-counts.csv")
            path = target.parent / f"{target.name}-summary.json"
            if path.exists() or path.is_symlink():
                _read(path, sha256_bytes(json_bytes(result)), "Existing review summary")
            else:
                save_json(path, result)
            print(json.dumps({"reviewed": result["reviewed"], "summary": str(path)}))
        else:
            print(json.dumps({"selected": len(cases), "sample_sha256": sample_sha,
                              "worksheet": str(target / "review.html"), "groups": groups}))
    except (ClassicalError, EvidenceError, EvaluationError, OSError, ValueError, TypeError, KeyError, AttributeError):
        parser.exit(1, "Private review failed. Check verified inputs, output directory, and decision coverage.\n")


if __name__ == "__main__":
    main()
