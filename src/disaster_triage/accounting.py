"""Per-attempt accounting; unfinished attempts accrue conservatively until reconciled."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
import uuid
from .constants import CUTOFF
from .provenance import save_json

def utc_now():
    return datetime.now(timezone.utc)

def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError("invalid opaque run/attempt identity")
    return value

class Attempt:
    def __init__(self, root, run_id, *, metadata, attempt_id=None,
                 monotonic=time.monotonic, now=utc_now):
        self.root, self.run_id = Path(root), identity(run_id)
        self.attempt_id = identity(attempt_id or uuid.uuid4().hex)
        allowed = {"config_sha256", "code_sha256", "code_revision", "data_sha256",
                   "model_revision", "versions", "seed", "device", "settings",
                   "parent_checkpoint_sha256", "provider_quota_before", "provider_quota_after"}
        if set(metadata) - allowed:
            raise ValueError("unapproved attempt metadata")
        self.metadata = metadata
        self.monotonic, self.now = monotonic, now

    def __enter__(self):
        self.path = self.root / "runs" / self.run_id / "attempts" / self.attempt_id / "attempt.json"
        self.path.parent.mkdir(parents=True, exist_ok=False)
        self.started = self.monotonic()
        self.record = {
            "schema_version": 1, "run_id": self.run_id, "attempt_id": self.attempt_id,
            "status": "running", "started_at": self.now().isoformat(),
            "heartbeat_at": self.now().isoformat(), "active_seconds": 0.,
            "metadata": self.metadata, "progress": {}, "artifact_sha256": {},
        }
        self.update()
        return self

    def update(self, *, progress=None, artifact_sha256=None):
        if progress is not None:
            if set(progress) - {"epoch", "optimizer_steps", "next_position"} or any(
                type(v) is not int or v < 0 for v in progress.values()
            ):
                raise ValueError("invalid progress metadata")
            self.record["progress"] = progress
        if artifact_sha256 is not None:
            if any(not re.fullmatch(r"[a-z0-9_-]+", k) or
                   not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                   for k, v in artifact_sha256.items()):
                raise ValueError("invalid artifact digest metadata")
            self.record["artifact_sha256"].update(artifact_sha256)
        self.record["active_seconds"] = max(0., self.monotonic() - self.started)
        self.record["heartbeat_at"] = self.now().isoformat()
        save_json(self.path, self.record)

    def __exit__(self, kind, value, traceback):
        self.record["status"] = (
            "completed" if kind is None else
            "interrupted" if issubclass(kind, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        # Exception messages can contain user inputs. Record the status only.
        self.update()
        self.record["ended_at"] = self.now().isoformat()
        save_json(self.path, self.record)

def budget_summary(root, *, now=None):
    now = now or utc_now()
    if now.tzinfo is None:
        raise ValueError("accounting requires timezone-aware time")
    records, seen = [], set()
    active, gpu = 0., 0.
    starts, ends = [], []
    unresolved = []
    for path in sorted((Path(root) / "runs").glob("*/attempts/*/attempt.json")):
        record = json.loads(path.read_text())
        key = (identity(record["run_id"]), identity(record["attempt_id"]))
        if key in seen:
            raise ValueError("duplicate attempt identity")
        seen.add(key)
        if record["status"] not in ("running", "completed", "failed", "interrupted"):
            raise ValueError("invalid attempt status")
        duration = record["active_seconds"]
        if type(duration) not in (float, int) or not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid recorded duration")
        end = datetime.fromisoformat(record.get("ended_at", record["heartbeat_at"]))
        if record["status"] == "running":
            duration += max(0., (now - end).total_seconds())
            end = max(end, now)
            unresolved.append(record["attempt_id"])
        starts.append(datetime.fromisoformat(record["started_at"]))
        ends.append(end)
        active += duration
        device = record["metadata"].get("device", "cpu")
        if device.startswith("cuda:") or device == "mps":
            gpu += duration
        records.append({"run_id": key[0], "attempt_id": key[1],
                        "status": record["status"], "active_seconds": duration,
                        "provider_quota_before": record["metadata"].get("provider_quota_before"),
                        "provider_quota_after": record["metadata"].get("provider_quota_after")})
    return {
        "active_job_hours": active / 3600, "remaining_job_hours": max(0., 24 - active / 3600),
        "gpu_device_hours": gpu / 3600,
        "wall_span_hours": (max(ends) - min(starts)).total_seconds() / 3600 if starts else 0.,
        "unresolved_attempts": unresolved, "attempts": records,
    }

def guard_budget(root, *, now=None):
    now = now or utc_now()
    summary = budget_summary(root, now=now)
    if now >= datetime.fromisoformat(CUTOFF):
        raise RuntimeError("training cutoff reached")
    if summary["active_job_hours"] >= 24:
        raise RuntimeError("cumulative active-job budget exhausted")
    return summary


def merge_accounting(source, destination):
    """Import retained attempt records once; conflicting histories need review."""
    source, destination = Path(source), Path(destination)
    imported = 0
    for path in sorted((source / "runs").glob("*/attempts/*/attempt.json")):
        content = path.read_bytes()
        record = json.loads(content)
        run_id, attempt_id = identity(record["run_id"]), identity(record["attempt_id"])
        if path.parent.name != attempt_id or path.parent.parent.parent.name != run_id:
            raise ValueError("accounting path/identity mismatch")
        duration = record["active_seconds"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid imported duration")
        target = destination / "runs" / run_id / "attempts" / attempt_id / "attempt.json"
        if target.exists():
            if target.read_bytes() == content:
                continue
            previous = json.loads(target.read_bytes())
            if _advances(previous, record):
                from .provenance import atomic_bytes
                atomic_bytes(target, content)
                imported += 1
                continue
            if _advances(record, previous):
                continue
            raise ValueError("conflicting retained attempt history; reconcile with evidence")
        from .provenance import atomic_bytes
        atomic_bytes(target, content)
        imported += 1
    return imported


def _advances(earlier, later):
    """Only consistent heartbeat progression may supersede a running snapshot."""
    if earlier["status"] != "running":
        return False
    identity_fields = ("schema_version", "run_id", "attempt_id", "started_at", "metadata")
    if any(earlier.get(key) != later.get(key) for key in identity_fields):
        return False
    if later["status"] not in ("running", "completed", "failed", "interrupted"):
        return False
    if later["active_seconds"] < earlier["active_seconds"]:
        return False
    if datetime.fromisoformat(later["heartbeat_at"]) < datetime.fromisoformat(earlier["heartbeat_at"]):
        return False
    if later.get("progress", {}).get("optimizer_steps", 0) < earlier.get("progress", {}).get("optimizer_steps", 0):
        return False
    return all(later.get("artifact_sha256", {}).get(key) == value
               for key, value in earlier.get("artifact_sha256", {}).items())


def guard_evaluation_budget(root, seconds, *, now=None):
    """Admit inference and scoring under the shared cap, independent of training cutoff."""
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("evaluation needs a positive measured forecast")
    summary = budget_summary(root, now=now)
    if summary["unresolved_attempts"]:
        raise RuntimeError("reconcile unresolved attempts before evaluation")
    available = summary["remaining_job_hours"] * 3600
    if seconds >= available:
        raise RuntimeError("evaluation exceeds the remaining cumulative budget")
    return available
