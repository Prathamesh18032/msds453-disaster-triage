"""Explicit runtime probing; no automatic backend fallback."""
import importlib.metadata
import os
import platform
import random
import sys

PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "transformers",
            "huggingface-hub", "safetensors", "datasketch")

def versions():
    result = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result

def doctor(config, root):
    import torch
    from .profiles import profile_errors
    inventory = versions()
    profile = "kaggle-py313" if sys.version_info[:2] == (3, 13) else "macos-py312"
    errors = profile_errors(profile, platform.python_version(), inventory)
    return {
        "versions": inventory, "python_supported": sys.version_info[:2] in ((3, 12), (3, 13)),
        "runtime_profile": profile, "profile_errors": errors,
        "platform": platform.system(), "architecture": platform.machine(),
        "requested_device": config.device,
        "requested_device_available": (
            config.device == "cpu" or
            (config.device == "mps" and torch.backends.mps.is_available()) or
            (config.device.startswith("cuda:") and torch.cuda.is_available() and
             int(config.device.split(":")[1]) < torch.cuda.device_count())
        ),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_build": torch.version.cuda,
        "storage": {k: str(v) for k, v in config.paths(root).items()},
        "kaggle_quota": "Unknown - user verification required",
        "kaggle_session_limits": "Unknown - user verification required",
    }

def configure_cache(config, root):
    cache = config.paths(root)["cache_dir"]
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache / "huggingface" / "hub")
    return cache

def select_device(name):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if name == "cpu":
        return torch.device(name)
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device(name)
    if name.startswith("cuda:") and torch.cuda.is_available():
        index = int(name.split(":")[1])
        if index < torch.cuda.device_count():
            torch.cuda.set_device(index)
            return torch.device(name)
    raise ValueError("requested backend unavailable; select another backend explicitly")

def seed_runtime(seed, device):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator
