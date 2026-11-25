"""CPU evaluation of validated prediction bundles and fitted development policies."""

from dataclasses import dataclass
from math import exp, isfinite
from pathlib import Path
import re
from statistics import mean, stdev

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from .bundles import BundleExpectation, read_bundle
from .constants import LABELS
from .evaluation_access import require_final_access
from .provenance import json_bytes, sha256_bytes


REQUESTS = LABELS.index("requests_or_urgent_needs")
LABEL_COUNT = len(LABELS)
ECE_BINS = 15
REVIEW_PERCENTS = (0, 10, 20, 30)
TEMPERATURE_BOUNDS = (-5.0, 5.0)  # log(T)
TRANSFORMER_SEEDS = frozenset((42, 43, 44))
ROLE_DIGESTS = ("role_manifest_sha256", "payload_sha256")
DATA_KEYS = ("mode", "archive_sha256", "eligible_manifest_sha256")
RUN_KEYS = (*DATA_KEYS, "run_id", "model_family", "seed", "config_sha256", "code_sha256")


@dataclass(frozen=True)
class BundleInput:
    """Directory and trusted expectations supplied independently of its manifest."""

    directory: Path
    expected: BundleExpectation
    run_id: str
    model_family: str
    seed: int | None


@dataclass(frozen=True)
class SeedCohort:
    """Selected shared settings/epoch and the allowed seed-specific config digests."""

    settings_sha256: str
    epoch: int
    config_sha256_by_seed: dict[int, str]


def _fields(mapping, keys):
    return {key: mapping[key] for key in keys}


def _is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _load(bundle, final_access=None):
    if bundle.expected.role == "final_test":
        require_final_access(final_access, {
            "mode": bundle.expected.mode, "role": bundle.expected.role, **bundle.expected.digests,
        })
        final_access.check_run({
            "run_id": bundle.run_id, "model_family": bundle.model_family, "seed": bundle.seed,
            **bundle.expected.digests,
        })
    manifest, arrays = read_bundle(bundle.directory, expected=bundle.expected, final_access=final_access)
    if (manifest["run_id"], manifest["model_family"], manifest["seed"]) != (
        bundle.run_id, bundle.model_family, bundle.seed
    ) or type(manifest["seed"]) is not type(bundle.seed):
        raise ValueError("prediction bundle run, family, or seed mismatch")
    return {
        "mode": bundle.expected.mode,
        **_fields(manifest, RUN_KEYS[1:] + ("role", *ROLE_DIGESTS)),
    }, arrays


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _classification(truth, predicted, *, display=False):
    counts = np.bincount(
        LABEL_COUNT * truth + predicted, minlength=LABEL_COUNT ** 2
    ).reshape(LABEL_COUNT, LABEL_COUNT)
    classes = {}
    for index, label in enumerate(LABELS):
        correct = int(counts[index, index])
        support = int(counts[index, :].sum())
        predicted_count = int(counts[:, index].sum())
        classes[label] = {
            "precision": _ratio(correct, predicted_count) or 0.0,
            "recall": _ratio(correct, support) or 0.0,
            "f1": _ratio(2 * correct, support + predicted_count) or 0.0,
            "support": support,
            "predicted_count": predicted_count,
        }
    macro_f1 = mean(row["f1"] for row in classes.values()) if len(truth) else None
    if display:
        for row in classes.values():
            if not row["predicted_count"]:
                row["precision"] = None
            if not row["support"]:
                row["recall"] = None
            if not row["support"] + row["predicted_count"]:
                row["f1"] = None
    return {
        "n_rows": len(truth),
        "macro_f1": macro_f1,
        "classes": classes,
    }


def _nll(truth, log_probs):
    return -float(log_probs[np.arange(len(truth)), truth].mean())


def _ece(confidence, correct):
    bins = np.minimum((confidence * ECE_BINS).astype(np.int64), ECE_BINS - 1)
    accuracy_sums = np.bincount(bins, weights=correct, minlength=ECE_BINS)
    confidence_sums = np.bincount(bins, weights=confidence, minlength=ECE_BINS)
    return float(np.abs(accuracy_sums - confidence_sums).sum() / len(confidence))


def _probability_metrics(truth, predicted, log_probs):
    confidence = np.exp(log_probs.max(axis=1))
    return {"nll": _nll(truth, log_probs), "ece": _ece(confidence, truth == predicted)}, confidence


def _fit_temperature(logits, truth):
    def objective(log_temperature):
        return _nll(truth, log_softmax(logits / exp(log_temperature), axis=1))

    baseline = objective(0.0)
    result = minimize_scalar(
        objective, bounds=TEMPERATURE_BOUNDS, method="bounded", options={"xatol": 1e-6}
    )
    if (
        not result.success
        or not isfinite(result.fun)
        or not TEMPERATURE_BOUNDS[0] <= result.x <= TEMPERATURE_BOUNDS[1]
    ):
        raise ValueError("temperature optimization failed or returned a nonfinite result")
    return exp(result.x) if baseline - result.fun > 1e-12 else 1.0


def fit_review_policy(calibration: BundleInput, policy: BundleInput) -> dict:
    """Fit one temperature and three global thresholds on separate development roles."""
    if (calibration.expected.role, policy.expected.role) != ("calibration", "policy"):
        raise ValueError("review policy fitting requires calibration and policy roles")
    calibration_id, calibration_arrays = _load(calibration)
    policy_id, policy_arrays = _load(policy)
    run = _fields(calibration_id, RUN_KEYS)
    if run != _fields(policy_id, RUN_KEYS):
        raise ValueError("calibration and policy must share the run and provenance")
    if run["model_family"] == "majority":
        raise ValueError("review policy fitting requires classifier logits")
    if set(calibration_arrays["row_id"]).intersection(policy_arrays["row_id"]):
        raise ValueError("calibration and policy rows overlap")
    if not len(calibration_arrays["y_true"]) or not len(policy_arrays["y_true"]):
        raise ValueError("calibration and policy fitting require nonempty roles")
    temperature = _fit_temperature(
        calibration_arrays["logits"].astype(np.float64), calibration_arrays["y_true"]
    )
    policy_logits = policy_arrays["logits"].astype(np.float64)
    confidence = np.sort(np.exp(log_softmax(policy_logits / temperature, axis=1).max(axis=1)))
    thresholds = [
        {"review_percent": percent, "threshold": float(confidence[percent * len(confidence) // 100])}
        for percent in REVIEW_PERCENTS[1:]
    ]
    return {
        "schema_version": 1,
        "run": run,
        "inputs": {
            "calibration": _fields(calibration_id, ROLE_DIGESTS),
            "policy": _fields(policy_id, ROLE_DIGESTS),
        },
        "temperature": temperature,
        "thresholds": [{"review_percent": 0, "threshold": 0.0}, *thresholds],
    }


def _validate_fitted_policy(record, expected_digest, bundle):
    if bundle["role"] not in ("policy", "final_test") or bundle["model_family"] == "majority":
        raise ValueError("a fitted policy applies to the same run’s logits")
    if not _is_digest(expected_digest) or sha256_bytes(json_bytes(record)) != expected_digest:
        raise ValueError("fitted-policy digest mismatch")
    if (
        not isinstance(record, dict)
        or set(record) != {"schema_version", "run", "inputs", "temperature", "thresholds"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["run"] != _fields(bundle, RUN_KEYS)
    ):
        raise ValueError("fitted policy does not match the evaluated run")
    inputs = record["inputs"]
    if (
        not isinstance(inputs, dict) or set(inputs) != {"calibration", "policy"}
        or any(not isinstance(item, dict) or set(item) != set(ROLE_DIGESTS)
               or not all(map(_is_digest, item.values())) for item in inputs.values())
        or (bundle["role"] == "policy" and inputs["policy"] != _fields(bundle, ROLE_DIGESTS))
    ):
        raise ValueError("fitted policy input provenance mismatch")
    temperature = record["temperature"]
    thresholds = record["thresholds"]
    if (
        type(temperature) not in (float, int)
        or not exp(TEMPERATURE_BOUNDS[0]) <= temperature <= exp(TEMPERATURE_BOUNDS[1])
        or not isinstance(thresholds, list)
        or len(thresholds) != len(REVIEW_PERCENTS)
    ):
        raise ValueError("invalid fitted temperature or thresholds")
    previous = 0.0
    for percent, scenario in zip(REVIEW_PERCENTS, thresholds, strict=True):
        if (
            not isinstance(scenario, dict)
            or set(scenario) != {"review_percent", "threshold"}
            or type(scenario["review_percent"]) is not int or scenario["review_percent"] != percent
            or type(scenario["threshold"]) not in (float, int)
            or not previous <= scenario["threshold"] <= (1.0 if percent else 0.0)
        ):
            raise ValueError("invalid fitted review thresholds")
        previous = scenario["threshold"]


def _review(truth, predicted, accepted):
    n_accepted = int(accepted.sum())
    n_deferred = len(truth) - n_accepted
    wrong_accepted = int(np.count_nonzero(accepted & (truth != predicted)))
    true_requests = truth == REQUESTS
    predicted_requests = predicted == REQUESTS
    request_support = int(true_requests.sum())
    deferred_requests = int(np.count_nonzero(~accepted & true_requests))
    return {
        "accepted": n_accepted,
        "deferred": n_deferred,
        "wrong_accepted": wrong_accepted,
        "coverage": _ratio(n_accepted, len(truth)),
        "review_fraction": _ratio(n_deferred, len(truth)),
        "selective_error": _ratio(wrong_accepted, n_accepted),
        "requests": {
            "support": request_support,
            "recall": _ratio(int(np.count_nonzero(true_requests & predicted_requests)), request_support),
            "deferred": deferred_requests,
            "deferred_fraction": _ratio(deferred_requests, request_support),
            "accepted_false_negatives": int(np.count_nonzero(accepted & true_requests & ~predicted_requests)),
            "accepted_false_positives": int(np.count_nonzero(accepted & ~true_requests & predicted_requests)),
        },
    }


def score_bundle(
    bundle: BundleInput, *, fitted_policy: dict | None = None,
    fitted_policy_sha256: str | None = None, final_access=None,
) -> dict:
    """Score predictions and apply a saved policy by verified digest."""
    final = bundle.expected.role == "final_test"
    if (fitted_policy is None) != (fitted_policy_sha256 is None):
        raise ValueError("supply the fitted policy and its expected digest together")
    if final:
        require_final_access(final_access, {
            "mode": bundle.expected.mode, "role": "final_test", **bundle.expected.digests,
        })
        final_access.check_policy(bundle.run_id, fitted_policy_sha256)
        if fitted_policy is not None and _is_digest(fitted_policy_sha256):
            if sha256_bytes(json_bytes(fitted_policy)) != fitted_policy_sha256:
                raise ValueError("fitted-policy digest mismatch")
    metadata, arrays = _load(bundle, final_access)
    if fitted_policy is not None:
        _validate_fitted_policy(fitted_policy, fitted_policy_sha256, metadata)

    truth = arrays["y_true"]
    event_ids = arrays["event_id"]
    logits = arrays.get("logits")
    predicted = arrays["y_pred"] if logits is None else logits.argmax(axis=1)
    event_masks = {event: event_ids == event for event in sorted(set(event_ids.tolist()))}
    report = {
        "schema_version": 1,
        "bundle": metadata,
        "labels": list(LABELS),
        "pooled": _classification(truth, predicted, display=final),
        "events": {
            event: _classification(truth[mask], predicted[mask], display=final)
            for event, mask in event_masks.items()
        },
    }
    if logits is None:
        return report
    if not len(truth):
        report["calibration"] = {"raw": {"nll": None, "ece": None}}
        return report

    logits = logits.astype(np.float64)
    raw, _ = _probability_metrics(truth, predicted, log_softmax(logits, axis=1))
    report["calibration"] = {"raw": raw}
    if fitted_policy is not None:
        calibrated, confidence = _probability_metrics(
            truth, predicted, log_softmax(logits / fitted_policy["temperature"], axis=1)
        )
        report["calibration"]["calibrated"] = calibrated
        report["fitted_policy_sha256"] = fitted_policy_sha256
        report["review"] = []
        for scenario in fitted_policy["thresholds"]:
            accepted = confidence >= scenario["threshold"]
            report["review"].append({
                **scenario,
                "pooled": _review(truth, predicted, accepted),
                "events": {
                    event: _review(truth[mask], predicted[mask], accepted[mask])
                    for event, mask in event_masks.items()
                },
            })
    return report


def _seed_stats(scores):
    values = [entry["macro_f1"] for entry in scores]
    if all(value is None for value in values):
        return {"mean": None, "sample_sd": None}
    if any(type(value) not in (float, int) or not 0 <= value <= 1 for value in values):
        raise ValueError("seed reports must agree on score availability and contain valid macro-F1")
    return {
        "mean": mean(values) if len(values) >= 2 else None,
        "sample_sd": stdev(values) if len(values) >= 2 else None,
    }


def summarize_seeds(reports, *, cohort: SeedCohort, final_access=None) -> dict:
    """Summarize comparable DistilBERT scores, retaining each individual report."""
    configs = cohort.config_sha256_by_seed
    if (
        not _is_digest(cohort.settings_sha256)
        or type(cohort.epoch) is not int or not 1 <= cohort.epoch <= 3
        or not configs or not configs.keys() <= TRANSFORMER_SEEDS
        or not all(_is_digest(digest) for digest in configs.values())
    ):
        raise ValueError("declare selected settings, epoch, and seed-specific config digests")

    reports = tuple(reports)
    if not reports:
        raise ValueError("at least one DistilBERT report is required")
    scope_keys = (*DATA_KEYS, "role", "role_manifest_sha256")
    scope = _fields(reports[0]["bundle"], scope_keys)
    final = scope["role"] == "final_test"
    if final:
        require_final_access(final_access, scope)
        if configs.keys() != TRANSFORMER_SEEDS or len(reports) != len(TRANSFORMER_SEEDS):
            raise ValueError("final cohort requires all three locked seeds")
    events = sorted(reports[0]["events"])
    by_seed = {}
    for report in reports:
        bundle = report["bundle"]
        seed = bundle["seed"]
        if bundle["model_family"] != "distilbert" or seed not in configs or seed in by_seed:
            raise ValueError("expected unique declared DistilBERT seeds")
        if bundle["config_sha256"] != configs[seed]:
            raise ValueError("seed configuration does not match the declared cohort")
        if (
            report["labels"] != list(LABELS) or _fields(bundle, scope_keys) != scope
            or sorted(report["events"]) != events
        ):
            raise ValueError("seed reports must cover the same development role and frozen rows")
        if final:
            final_access.check_run(bundle)
            final_access.check_policy(bundle["run_id"], report.get("fitted_policy_sha256"))
        by_seed[seed] = report
    ordered = [by_seed[seed] for seed in sorted(by_seed)]
    return {
        "schema_version": 1,
        "scope": scope,
        "settings_sha256": cohort.settings_sha256,
        "epoch": cohort.epoch,
        "n_seeds": len(ordered),
        "exploratory": by_seed.keys() != TRANSFORMER_SEEDS,
        "pooled": _seed_stats([report["pooled"] for report in ordered]),
        "events": {event: _seed_stats([report["events"][event] for report in ordered]) for event in events},
        "individual": ordered,
    }
