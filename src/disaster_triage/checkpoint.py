"""Atomic recovery at complete optimizer boundaries."""
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from .provenance import sha256_file

def rng_state(device, order_generator):
    numpy = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "torch_cpu": torch.get_rng_state(), "order": order_generator.get_state(),
        "accelerator": None,
    }
    if device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    return state

def restore_rng(state, device, order_generator):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32),
                         numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(state["torch_cpu"])
    order_generator.set_state(state["order"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"])

def validate_progress(progress):
    required = {"epoch", "optimizer_steps", "permutation", "next_position"}
    if set(progress) != required:
        raise ValueError("invalid checkpoint progress keys")
    for key in ("epoch", "optimizer_steps", "next_position"):
        if type(progress[key]) is not int or progress[key] < 0:
            raise ValueError("invalid checkpoint progress")
    permutation = progress["permutation"]
    if (not isinstance(permutation, list) or
        any(type(i) is not int for i in permutation) or
        sorted(permutation) != list(range(len(permutation))) or
        progress["next_position"] > len(permutation)):
        raise ValueError("invalid checkpoint permutation/cursor")

def checkpoint_due(step, *, epoch_end=False):
    return epoch_end or (step > 0 and step % 100 == 0)

def save_checkpoint(path, *, model, optimizer, scheduler, progress, compatibility,
                    device, order_generator, lineage):
    validate_progress(progress)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("checkpoint only after a completed, cleared optimizer window")
    state = {
        "schema_version": 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "progress": progress, "compatibility": compatibility,
        "rng": rng_state(device, order_generator), "lineage": lineage,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="checkpoint-pending-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha256_file(path)

def restore_checkpoint(path, *, model, optimizer, scheduler, expected, device,
                       order_generator, expected_sha256=None):
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("checkpoint digest mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema_version", "model", "optimizer", "scheduler", "progress",
                "compatibility", "rng", "lineage"}
    if not isinstance(state, dict) or set(state) != required or state["schema_version"] != 1:
        raise ValueError("unsupported or incomplete checkpoint")
    if state["compatibility"] != expected:
        raise ValueError("checkpoint configuration/source/data/runtime mismatch")
    validate_progress(state["progress"])
    current = model.state_dict()
    if set(current) != set(state["model"]) or any(
        current[k].shape != state["model"][k].shape or current[k].dtype != state["model"][k].dtype
        for k in current
    ):
        raise ValueError("checkpoint model structure mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"], device, order_generator)
    return state
