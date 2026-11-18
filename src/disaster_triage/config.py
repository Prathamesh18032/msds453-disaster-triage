"""Strict TOML configuration; research-template execution remains gated."""
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import re
import tomllib
from .constants import CLASSICAL_SETTINGS, CUTOFF, EVENTS, LABELS, MODEL_ID, MODEL_REVISION
from .provenance import save_json

@dataclass(frozen=True)
class RunConfig:
    mode: str = "synthetic"
    model_revision: str = MODEL_REVISION
    device: str = "cpu"
    seed: int = 42
    precision: str = "fp32"
    physical_batch: int = 8
    accumulation: int = 2
    learning_rate: float = 2e-5
    max_epochs: int = 3
    max_length: int = 128
    weight_decay: float = .01
    beta1: float = .9
    beta2: float = .999
    epsilon: float = 1e-8
    clip_norm: float = 1.
    label_smoothing: float = 0.
    checkpoint_interval: int = 100
    active_job_hours_cap: float = 24.
    cutoff: str = CUTOFF
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    cache_dir: str = ".cache"

    def __post_init__(self):
        if not isinstance(self.model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("model_revision must be a full immutable commit")
        if self.mode not in ("synthetic", "research_template"):
            raise ValueError("mode must be synthetic or research_template")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|mps|cuda:[0-9]+", self.device):
            raise ValueError("device must be cpu, mps, or an explicit cuda index")
        for name in ("seed", "physical_batch", "accumulation", "max_epochs",
                     "max_length", "checkpoint_interval"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("learning_rate", "weight_decay", "beta1", "beta2", "epsilon",
                     "clip_norm", "label_smoothing", "active_job_hours_cap"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
        if self.seed not in (42, 43, 44):
            raise ValueError("seed must be 42, 43, or 44")
        if (self.physical_batch, self.accumulation) not in ((8, 2), (4, 4), (2, 8)):
            raise ValueError("accepted physical/accumulation pairs are 8/2, 4/4, 2/8")
        if self.learning_rate not in (2e-5, 5e-5) or not 1 <= self.max_epochs <= 3:
            raise ValueError("learning rate or epoch ceiling violates the contract")
        fixed = {
            "precision": "fp32", "max_length": 128, "weight_decay": .01,
            "beta1": .9, "beta2": .999, "epsilon": 1e-8, "clip_norm": 1.,
            "label_smoothing": 0., "checkpoint_interval": 100,
            "active_job_hours_cap": 24., "cutoff": CUTOFF,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} violates the adopted execution contract")
        for name in ("data_dir", "artifact_dir", "cache_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty relative path")
            path = Path(value)
            expected_base = {"data_dir": "data", "artifact_dir": "artifacts", "cache_dir": ".cache"}[name]
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != expected_base:
                raise ValueError("storage paths must stay inside the supplied root")
        if len({self.data_dir, self.artifact_dir, self.cache_dir}) != 3:
            raise ValueError("data, artifact, and cache paths must be distinct")

    def paths(self, root):
        root = Path(root).resolve()
        result = {}
        for key in ("data_dir", "artifact_dir", "cache_dir"):
            path = (root / getattr(self, key)).resolve()
            if not path.is_relative_to(root) or path == root:
                raise ValueError("storage resolves outside the execution root")
            result[key] = path
        return result

def load_config(path):
    content = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(content) - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError("unknown configuration fields: " + ", ".join(sorted(unknown)))
    return RunConfig(**content)

def resolved_config(config, *, versions, model_revision, training_counts, scheduler_total_steps=None):
    if config.mode != "synthetic":
        raise ValueError("research execution awaits the Slice 3 eligible-data freeze")
    from .loss import class_weights
    weights = class_weights(training_counts).tolist()
    return {
        "schema_version": 1, **asdict(config), "labels": list(LABELS),
        "events": {key: list(value) for key, value in EVENTS.items()}, "classical": CLASSICAL_SETTINGS,
        "model_id": "synthetic_linear_6x10_dropout_0.2" if model_revision == "offline-tiny-v1" else MODEL_ID,
        "model_revision": model_revision, "tokenizer_revision": None if model_revision == "offline-tiny-v1" else model_revision,
        "versions": versions, "training_counts": list(training_counts), "weights": weights,
        "effective_batch": 16, "scheduler": "linear", "warmup_steps": 0,
        "optimizer": "AdamW", "decay_policy": "all_parameters",
        "scheduler_total_steps": scheduler_total_steps,
        "data_loader_workers": 0, "max_runs": 4, "max_total_epochs": 12,
        "synthetic_data": True,
    }

def save_resolved(path, config, **kwargs):
    result = resolved_config(config, **kwargs)
    return result, save_json(path, result)
