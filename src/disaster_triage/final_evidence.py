"""Read-only projection of the accepted held-out aggregates."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean, stdev

from .constants import ARCHIVE_SHA256, EVENTS, LABELS
from .evidence_inputs import (
    EvidenceError, REQUESTS, _correct, _digest, _indexed, _read, _require,
    _same_number, _support, _validate_report,
)
from .provenance import json_bytes

ROLE = "final_test"
RUNS = {
    "majority": ("majority", None), "lr": ("tfidf_lr", None),
    "seed42": ("distilbert", 42), "seed43": ("distilbert", 43), "seed44": ("distilbert", 44),
}
TRAINED = tuple(key for key in RUNS if key != "majority")
SEEDS = TRAINED[1:]
EVENT_ORDER = EVENTS[ROLE]


@dataclass(frozen=True)
class FinalEvidence:
    manifest: dict
    reports: dict
    cohort: dict
    hashes: dict

    @property
    def common_labels(self):
        reference = self.reports["majority"]
        return tuple(label for label in LABELS if all(
            reference["events"][event]["classes"][label]["support"] > 0 for event in EVENT_ORDER
        ))

    def scores(self, scope):
        return self.cohort["pooled"] if scope == "pooled" else self.cohort["events"][scope]

    def common_score(self, key, event):
        if key == "cohort_mean":
            return mean(self.common_score(seed, event) for seed in SEEDS)
        cells = self.reports[key]["events"][event]["classes"]
        return mean(cells[label]["f1"] for label in self.common_labels)


def _validate_review(point, metrics):
    rows = metrics["n_rows"]
    accepted, deferred, wrong = (point[name] for name in ("accepted", "deferred", "wrong_accepted"))
    _require(all(type(n) is int and n >= 0 for n in (accepted, deferred, wrong))
             and accepted + deferred == rows and wrong <= accepted,
             "Invalid held-out review counts.")
    _require(_same_number(point["coverage"], accepted / rows)
             and _same_number(point["review_fraction"], deferred / rows)
             and (_same_number(point["selective_error"], wrong / accepted) if accepted
                  else point["selective_error"] is None),
             "Held-out review rates disagree with the counts.")
    cell = metrics["classes"][REQUESTS]
    requests = point["requests"]
    support = cell["support"]
    missed, false_positive, sent = (requests[name] for name in (
        "accepted_false_negatives", "accepted_false_positives", "deferred",
    ))
    _require(all(type(n) is int and n >= 0 for n in (missed, false_positive, sent))
             and requests["support"] == support
             and requests["recall"] == cell["recall"]
             and missed <= support - _correct(cell)
             and false_positive <= cell["predicted_count"] - _correct(cell)
             and sent <= min(support, deferred) and missed + sent <= support
             and missed + false_positive <= wrong
             and (_same_number(requests["deferred_fraction"], sent / support) if support
                  else requests["deferred_fraction"] is None),
             "Held-out request counts disagree with the class or review totals.")


def _validate_policy(report):
    _require(_digest(report["fitted_policy_sha256"]), "Missing frozen policy reference.")
    for variant in ("raw", "calibrated"):
        scores = report["calibration"][variant]
        nll = scores["nll"]
        _require(type(nll) in (int, float) and isfinite(nll) and nll >= 0
                 and _same_number(scores["ece"], scores["ece"]), "Invalid held-out calibration scores.")
    scenarios = report["review"]
    _require([point["review_percent"] for point in scenarios] == [0, 10, 20, 30],
             "Expected the four frozen held-out review scenarios.")
    thresholds = [point["threshold"] for point in scenarios]
    _require(all(_same_number(n, n) for n in thresholds) and thresholds == sorted(thresholds)
             and thresholds[0] == 0, "Invalid frozen review thresholds.")
    for point in scenarios:
        _require(set(point["events"]) == set(EVENT_ORDER), "Incomplete event review metrics.")
        _validate_review(point["pooled"], report["pooled"])
        for event in EVENT_ORDER:
            _validate_review(point["events"][event], report["events"][event])
        if point["review_percent"] == 0:
            _require(point["pooled"]["deferred"] == 0, "Unfiltered review scenario must retain all rows.")


def load_final_evidence(path, expected):
    """Open exactly seven aggregate files. No bundle or dataset access is granted."""
    path = Path(path)
    manifest = _read(path, expected, "Final manifest")
    try:
        scope = manifest["role"]
        refs = manifest["outputs"]
        campaign = manifest["campaign"]
        _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
                 and manifest["mode"] == scope["mode"] == "research" and scope["role"] == ROLE
                 and scope["archive_sha256"] == ARCHIVE_SHA256
                 and all(_digest(scope[field]) for field in ("eligible_manifest_sha256", "role_manifest_sha256"))
                 and all(_digest(manifest[field]) for field in ("selection_lock_sha256", "readiness_sha256"))
                 and manifest["primary"] == "distilbert" and refs.keys() == RUNS.keys(),
                 "Invalid final manifest scope or predictor set.")
        hashes = {"manifest.json": expected}
        reports = {}
        for key, (family, seed) in RUNS.items():
            relative = Path("final") / key / "report.json"
            parent = path.parent / relative.parent
            _require(not parent.is_symlink() and not parent.parent.is_symlink(),
                     "Final report folders must be local directories.")
            ref = refs[key]
            report = _read(path.parent / relative, ref["report_sha256"], "Final predictor report")
            _validate_report(report, scope, EVENT_ORDER, family=family, seed=seed)
            bundle = report["bundle"]
            _require(bundle["run_id"] == f"{campaign}-{key}" and _digest(bundle["code_sha256"])
                     and bundle["payload_sha256"] == ref["payload_sha256"],
                     "Held-out report disagrees with its predictor reference.")
            if key in TRAINED:
                _validate_policy(report)
            else:
                _require(not ({"review", "calibration", "fitted_policy_sha256"} & report.keys()),
                         "The majority reference only supplies classification metrics.")
            reports[key] = report
            hashes[relative.as_posix()] = ref["report_sha256"]
        reference = reports["majority"]["events"]
        _require(all(_support(report["events"][event]) == _support(reference[event])
                     for report in reports.values() for event in EVENT_ORDER),
                 "Held-out reports disagree on class support.")
        cohort = _read(path.parent / "transformer-cohort.json", manifest["cohort_sha256"], "Final cohort")
        _require(cohort["schema_version"] == 1 and cohort["scope"] == scope and cohort["epoch"] == 3
                 and cohort["n_seeds"] == 3 and cohort["exploratory"] is False
                 and set(cohort["events"]) == set(EVENT_ORDER), "Invalid held-out cohort metadata.")
        individual = _indexed(cohort["individual"], lambda report: report["bundle"]["seed"],
                              {42, 43, 44}, "Expected unique final cohort seeds 42, 43, and 44.")
        _require(all(json_bytes(individual[RUNS[key][1]]) == json_bytes(reports[key]) for key in SEEDS),
                 "Held-out cohort and individual reports disagree.")
        for name in ("pooled", *EVENT_ORDER):
            stats = cohort["pooled"] if name == "pooled" else cohort["events"][name]
            scores = [metrics(reports[key], name)["macro_f1"] for key in SEEDS]
            _require(_same_number(stats["mean"], mean(scores))
                     and _same_number(stats["sample_sd"], stdev(scores)),
                     "Held-out cohort mean or sample SD disagrees with the reports.")
        hashes["transformer-cohort.json"] = manifest["cohort_sha256"]
        evidence = FinalEvidence(manifest, reports, cohort, hashes)
        _require(bool(evidence.common_labels), "Held-out events have no labels with shared true support.")
    except EvidenceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        raise EvidenceError("Malformed held-out aggregate structure or values.") from None
    return evidence


def metrics(report, scope):
    return report["pooled"] if scope == "pooled" else report["events"][scope]
