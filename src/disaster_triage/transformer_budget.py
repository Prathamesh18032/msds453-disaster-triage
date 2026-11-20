"""Forecast actual campaign work and enforce paired independent time allocations."""

from datetime import datetime
import math
from statistics import mean

from .constants import CUTOFF

GRACE_SECONDS = 2
PILOT_WORKER_SECONDS = 1800
MARGIN = 1.20


def overlap_rate(first, second):
    """Compare a warm serial sample with the simultaneous part of a paired pilot."""
    serial = [end - start for step, start, end in first["step_seconds"] if 5 < step <= 50]
    left = [item for item in first["step_seconds"] if item[0] > 50]
    right = [item for item in second["step_seconds"] if item[0] > 5]
    if len(serial) < 10 or not left or not right:
        return {"pair": False, "serial_steps_per_second": None, "paired_steps_per_second": None}
    start = max(left[0][1], right[0][1])
    stop = min(left[-1][2], right[-1][2])
    counts = [sum(t0 >= start and t1 <= stop for _, t0, t1 in trace) for trace in (left, right)]
    serial_rate = 1 / mean(serial)
    pair_rate = sum(counts) / (stop - start) if min(counts) >= 10 and stop > start else None
    return {"pair": pair_rate is not None and pair_rate > serial_rate,
            "serial_steps_per_second": serial_rate, "paired_steps_per_second": pair_rate}


def pilot_estimate(telemetry):
    durations = [end - start for step, start, end in telemetry["step_seconds"] if step > 5]
    if len(durations) < 10 or any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Insufficient validated pilot timing to admit longer work.")
    return {
        "optimizer_step_seconds": mean(durations),
        "startup_seconds": telemetry["startup_seconds"],
        "checkpoint_seconds": telemetry["checkpoint_seconds"],
        "checkpoint_bytes": telemetry["checkpoint_bytes"],
        "inference_row_seconds": telemetry["inference_seconds"] / telemetry["inference_rows"],
    }


def evaluation_reserve(inputs, estimate):
    # Three possible transformers plus LR. Use the measured transformer rate
    # for all four as a conservative proxy. Allow one model load per run.
    rows = inputs.lineage["slice8_prediction_rows_per_model"]
    return MARGIN * 4 * (rows * estimate["inference_row_seconds"] + estimate["startup_seconds"])


def run_seconds(*, estimate, config, steps, completed_epochs, selection_rows, screening):
    total = config["steps_per_epoch"] * config["max_epochs"]
    remaining = max(0, total - steps)
    evals = config["max_epochs"] - completed_epochs if screening else 1
    checkpoints = math.ceil(remaining / 100) + evals
    return MARGIN * (
        remaining * estimate["optimizer_step_seconds"]
        + (checkpoints + 2) * estimate["checkpoint_seconds"]
        + evals * selection_rows * estimate["inference_row_seconds"]
        + estimate["startup_seconds"]
        + 2 * GRACE_SECONDS
    )


def time_remaining(now, budget, *, backend, quota_hours, session_hours, segment_hours):
    for value in (quota_hours, session_hours, segment_hours):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Time allowances must be finite positive hours.")
    if budget["unresolved_attempts"]:
        raise ValueError("Reconcile unresolved attempt records before starting research.")
    wall = (datetime.fromisoformat(CUTOFF) - now).total_seconds()
    active = budget["remaining_job_hours"] * 3600
    provider = quota_hours * 3600 if backend == "cuda" and quota_hours is not None else math.inf
    if min(wall, active, provider) <= 0:
        raise ValueError("Research cutoff or compute budget reached.")
    return {
        "active": active,
        "wall": wall,
        "provider": provider,
        "segment": min(wall, segment_hours * 3600,
                       session_hours * 3600 if session_hours is not None else math.inf),
    }


def admits_group(seconds, available, reserve, *, paired):
    wall = max(seconds) if paired else sum(seconds)
    return (sum(seconds) + reserve <= min(available["active"], available["provider"])
            and wall <= available["wall"])


def allocation(available, workers, reserve, *, pilot=False):
    """Give each worker a hard interval whose sum cannot exceed either allowance."""
    maximum = min(
        (available["active"] - reserve) / workers,
        (available["provider"] - reserve) / workers,
        available["segment"],
    )
    if pilot:
        maximum = min(maximum, PILOT_WORKER_SECONDS)
    if maximum <= GRACE_SECONDS * 2:
        raise ValueError("Insufficient budget or session time for the requested stage.")
    return maximum - GRACE_SECONDS


def choose_group(candidates, available, reserve, pair):
    """Admit one group in protocol order. Reforecast before the next group."""
    if len(candidates) >= 2 and pair:
        group = tuple(candidates[:2])
        if admits_group([seconds for _, seconds in group], available, reserve, paired=True):
            return tuple(key for key, _ in group)
    for key, seconds in candidates:
        if admits_group([seconds], available, reserve, paired=False):
            return (key,)
    return ()
