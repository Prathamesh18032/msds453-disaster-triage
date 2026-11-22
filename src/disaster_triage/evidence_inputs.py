"""Validate the aggregate-only interface from the two development producers."""

from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path
import re
from statistics import mean

from .constants import ARCHIVE_SHA256, CLASSICAL_SETTINGS, EVENTS, LABELS
from .evaluation import SeedCohort, summarize_seeds
from .provenance import json_bytes, sha256_bytes

ROLE = "model_selection"
CLASSICAL_GRID = {
    f"lr-{c:g}-{weight or 'unweighted'}": (c, weight or "unweighted")
    for c in CLASSICAL_SETTINGS["C"] for weight in CLASSICAL_SETTINGS["class_weight"]
}
SCREEN_RATES = {"seed42-lr2e-5": 2e-5, "seed42-lr5e-5": 5e-5}
REPEATS = {"seed43": 43, "seed44": 44}
SYNTHETIC_EVENTS = ("synthetic_alpha", "synthetic_beta")
REQUESTS = "requests_or_urgent_needs"
MISSING = "missing_or_found_people"


class EvidenceError(ValueError):
    """Fixed diagnostic safe to show without printing input records."""


@dataclass(frozen=True)
class Candidate:
    stage: str
    key: str
    report: dict
    selected: bool = False
    c: float | None = None
    class_weight: str | None = None
    learning_rate: float | None = None
    epoch: int | None = None

    @property
    def bundle(self):
        return self.report["bundle"]


@dataclass(frozen=True)
class Evidence:
    events: tuple[str, ...]
    freeze_id: str
    scope: dict
    candidates: tuple[Candidate, ...]
    comparison: tuple[Candidate, ...]
    cohort: dict
    recipe: dict
    hashes: dict


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_score(value):
    return type(value) in (int, float) and 0 <= value <= 1


def _same_number(value, expected):
    return _is_score(value) and isclose(
        value, expected, rel_tol=0, abs_tol=1e-12,
    )


def _unique_json(pairs):
    result = dict(pairs)
    _require(len(result) == len(pairs), "Aggregate JSON contains duplicate fields.")
    return result


def _read(path, expected, label):
    _require(_digest(expected), f"{label} requires a lowercase SHA-256 digest.")
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or not a regular file.")
    try:
        content = path.read_bytes()
    except OSError:
        raise EvidenceError(f"Could not read {label.lower()}.") from None
    _require(sha256_bytes(content) == expected, f"{label} SHA-256 mismatch.")
    try:
        data = json.loads(content, object_pairs_hook=_unique_json)
    except (ValueError, RecursionError):
        raise EvidenceError(f"{label} contains malformed or duplicate JSON fields.") from None
    _require(isinstance(data, dict), f"{label} must be a JSON object.")
    return data


def _indexed(records, index, expected, message):
    result = {index(record): record for record in records}
    _require(len(result) == len(records) and result.keys() == expected, message)
    return result


def _support(metrics):
    return tuple(metrics["classes"][label]["support"] for label in LABELS)


def _validate_score(metrics):
    classes = metrics["classes"]
    _require(set(classes) == set(LABELS), "Report class table must contain all ten labels.")
    rows = metrics["n_rows"]
    _require(type(rows) is int and rows > 0, "Report scope must have a positive row count.")
    for cell in classes.values():
        support, predicted = cell["support"], cell["predicted_count"]
        _require(all(type(count) is int and 0 <= count <= rows for count in (support, predicted)),
                 "Report contains invalid class support or predicted counts.")
        _require(_is_score(cell["recall"]), "Report contains invalid recall.")
        correct = round(cell["recall"] * support)
        _require(0 <= correct <= min(support, predicted)
                 and _same_number(cell["recall"], correct / support if support else 0)
                 and _same_number(cell["precision"], correct / predicted if predicted else 0)
                 and _same_number(cell["f1"], 2 * correct / (support + predicted)
                                  if support + predicted else 0),
                 "Report class metrics disagree with their denominators.")
    _require(sum(_support(metrics)) == rows
             and sum(cell["predicted_count"] for cell in classes.values()) == rows
             and _same_number(metrics["macro_f1"], mean(cell["f1"] for cell in classes.values())),
             "Report row counts or fixed-label macro-F1 disagree with its class table.")


def _validate_report(report, scope, events, *, family, seed):
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1,
             "Unsupported evaluator report schema.")
    bundle = report["bundle"]
    _require(all(bundle.get(field) == value for field, value in scope.items()),
             "Report scope disagrees with the model-selection freeze or row manifest.")
    _require(report["labels"] == list(LABELS), "Report differs from canonical ten-label order.")
    _require(bundle["model_family"] == family and bundle["seed"] == seed
             and type(bundle["seed"]) is type(seed), "Unexpected report family or seed.")
    _require(isinstance(bundle["run_id"], str)
             and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", bundle["run_id"])
             and all(_digest(bundle[key]) for key in ("config_sha256", "payload_sha256")),
             "Invalid report run or configuration reference.")
    _require(set(report["events"]) == set(events), "Report has unexpected model-selection events.")
    pooled = report["pooled"]
    by_event = [report["events"][event] for event in events]
    for metrics in (pooled, *by_event):
        _validate_score(metrics)
    for label in LABELS:
        cell = pooled["classes"][label]
        events_cells = [metrics["classes"][label] for metrics in by_event]
        _require(all(cell[field] == sum(row[field] for row in events_cells)
                     for field in ("support", "predicted_count"))
                 and round(cell["recall"] * cell["support"]) ==
                 sum(round(row["recall"] * row["support"]) for row in events_cells),
                 "Pooled class counts disagree with the event totals.")


def _classical(selection, path, scope, events):
    refs = _indexed(selection["comparison"], lambda item: item["candidate"],
                    {"majority", *CLASSICAL_GRID}, "Expected one majority and six unique LR candidates.")
    winner = selection["winner"]
    _require(winner in CLASSICAL_GRID, "Invalid accepted LR winner reference.")
    hashes, candidates = {}, []
    for key, ref in refs.items():
        name = f"{key}/report.json"
        _require(ref["report"] == name and not (path.parent / key).is_symlink(),
                 "Classical report reference must name its own candidate report.")
        report = _read(path.parent / name, ref["report_sha256"], f"Classical {key} report")
        family = "majority" if key == "majority" else "tfidf_lr"
        _validate_report(report, scope, events, family=family, seed=None)
        bundle = report["bundle"]
        _require((ref["run_id"], ref["config_sha256"], ref["bundle_sha256"])
                 == (bundle["run_id"], bundle["config_sha256"], bundle["payload_sha256"])
                 and _same_number(ref["pooled_macro_f1"], report["pooled"]["macro_f1"]),
                 "Classical report disagrees with its selection reference.")
        c, weight = CLASSICAL_GRID.get(key, (None, None))
        candidates.append(Candidate("classical", key, report, key == winner, c, weight))
        hashes[f"classical/{name}"] = ref["report_sha256"]
    by_key = {candidate.key: candidate for candidate in candidates}
    _require(len({c.bundle["run_id"] for c in candidates}) == len(candidates)
             and _same_number(selection["winner_macro_f1"], by_key[winner].report["pooled"]["macro_f1"]),
             "Classical candidates must have unique runs and agree with the accepted LR score.")
    ordered = tuple(by_key[key] for key in ("majority", *CLASSICAL_GRID))
    return ordered, (by_key["majority"], by_key[winner]), hashes


def _transformer(summary, scope, events):
    recipe = summary["selection"]
    key, epoch = recipe["key"], recipe["epoch"]
    _require(key in SCREEN_RATES and type(epoch) is int and 1 <= epoch <= 3
             and recipe["learning_rate"] == SCREEN_RATES[key]
             and type(recipe["scheduler_total_steps"]) is int and recipe["scheduler_total_steps"] > 0
             and _digest(recipe["checkpoint_sha256"]) and _digest(recipe["eval_config_sha256"]),
             "Invalid sealed DistilBERT selection recipe.")
    screens = [(name, step) for name in SCREEN_RATES for step in range(1, 4)]
    repeats = [(name, epoch) for name in REPEATS]
    index = _indexed(summary["reports"], lambda item: (item["key"], item["epoch"]),
                     set(screens + repeats), "Expected six screening points and two unique repeat endpoints.")
    candidates = []
    for name, step in screens + repeats:
        ref = index[name, step]
        _require(type(ref["epoch"]) is int, "Invalid epoch report reference.")
        report = ref["aggregate"]
        _validate_report(report, scope, events, family="distilbert", seed=REPEATS.get(name, 42))
        candidate = Candidate(
            "repeat" if name in REPEATS else "screen", name, report,
            name in REPEATS or (name, step) == (key, epoch),
            learning_rate=SCREEN_RATES.get(name, recipe["learning_rate"]), epoch=step,
        )
        candidates.append(candidate)

    screens_by_run = [{c.bundle["run_id"] for c in candidates if c.key == key} for key in SCREEN_RATES]
    _require(all(len(run_ids) == 1 for run_ids in screens_by_run)
             and len({c.bundle["run_id"] for c in candidates}) == 4
             and len({c.bundle["config_sha256"] for c in candidates}) == len(candidates),
             "DistilBERT epoch reports have inconsistent run or configuration references.")
    chosen = tuple(candidate for candidate in candidates if candidate.selected)
    selected = chosen[0]
    _require(recipe["eval_config_sha256"] == selected.bundle["config_sha256"]
             and _same_number(recipe["pooled_macro_f1"], selected.report["pooled"]["macro_f1"]),
             "Selected DistilBERT recipe disagrees with its epoch report.")
    sealed = summary["cohort"]
    by_seed = _indexed(sealed["individual"], lambda report: report["bundle"]["seed"],
                       {42, 43, 44}, "Expected unique cohort seeds 42, 43, and 44.")
    _require(all(json_bytes(by_seed[c.bundle["seed"]]) == json_bytes(c.report) for c in chosen),
             "Selected seed reports differ from their embedded epoch reports.")
    settings = {field: recipe[field] for field in ("learning_rate", "epoch", "scheduler_total_steps")}
    computed = summarize_seeds(
        (c.report for c in chosen),
        cohort=SeedCohort(sha256_bytes(json_bytes(settings)), epoch,
                          {c.bundle["seed"]: c.bundle["config_sha256"] for c in chosen}),
    )
    metadata = ("schema_version", "scope", "settings_sha256", "epoch", "n_seeds", "exploratory")
    _require(all(json_bytes(sealed[field]) == json_bytes(computed[field]) for field in metadata)
             and set(sealed["events"]) == set(events), "Sealed cohort metadata or event scope disagrees.")
    for actual, expected in [(sealed["pooled"], computed["pooled"])] + [
        (sealed["events"][event], computed["events"][event]) for event in events
    ]:
        _require(all(_same_number(actual[field], expected[field]) for field in ("mean", "sample_sd")),
                 "Sealed cohort mean or sample SD disagrees with the selected reports.")
    return tuple(candidates), chosen, computed, {
        **settings, "key": key, "eval_config_sha256": recipe["eval_config_sha256"],
    }


def load_evidence(mode, selection_path, selection_sha256, summary_path, summary_sha256):
    """Read exactly two subject records and their seven verified classical reports."""
    selection = _read(selection_path, selection_sha256, "Classical selection")
    summary = _read(summary_path, summary_sha256, "DistilBERT summary")
    try:
        _require(all(type(record["schema_version"]) is int and record["schema_version"] == 1
                     for record in (selection, summary)), "Unsupported producer aggregate schema.")
        _require(selection["role"] == summary["role"] == ROLE,
                 "Both producer records must describe model selection.")
        lineage = selection["lineage"]
        freeze_id = lineage["freeze_id"]
        _require(_digest(freeze_id) and freeze_id == summary["freeze_id"],
                 "Producer records disagree on the frozen dataset.")
        research = mode == "development"
        scope = {
            "mode": "research" if research else "synthetic", "role": ROLE,
            "archive_sha256": lineage["archive_sha256"],
            "eligible_manifest_sha256": lineage["eligible_manifest_sha256"],
            "role_manifest_sha256": lineage["role_manifest_sha256"][ROLE],
        }
        _require(all(_digest(scope[field]) for field in (
            "archive_sha256", "eligible_manifest_sha256", "role_manifest_sha256",
        )) and (not research or scope["archive_sha256"] == ARCHIVE_SHA256),
            "Invalid model-selection lineage.")
        events = EVENTS["development"] if research else SYNTHETIC_EVENTS
        classical, reference, hashes = _classical(selection, Path(selection_path), scope, events)
        _require(summary["full_schedule_complete"] is True, "DistilBERT development schedule is incomplete.")
        transformer, chosen, cohort, recipe = _transformer(summary, scope, events)
        candidates = classical + transformer
        base = candidates[0].report
        _require(all(_support(c.report["events"][event]) == _support(base["events"][event])
                     for c in candidates for event in events),
                 "Candidates disagree on model-selection event or class support.")
    except EvidenceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        raise EvidenceError("Malformed development aggregate structure or values.") from None
    return Evidence(events, freeze_id, scope, candidates, reference + chosen, cohort, recipe, {
        "classical/selection.json": selection_sha256, "transformer/development-summary.json": summary_sha256,
        **hashes,
    })
