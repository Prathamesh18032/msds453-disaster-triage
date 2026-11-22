"""Offline inference with the campaign's saved model and preprocessing bytes."""

from importlib.metadata import version
import os
from pathlib import Path

import numpy as np

from .classical import aligned_text, checked_iterations, decision_scores
from .constants import LABELS, MODEL_ID, MODEL_REVISION
from .provenance import sha256_file
from .transformer_data import aligned_projection

BATCH = 8
CPU_THREADS = 4
ASSETS = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt")


def asset_inventory(root, backend):
    from huggingface_hub import snapshot_download
    import torch

    from .transformer_run import runtime_inventory

    if backend not in ("cpu", "mps"):
        raise ValueError("evaluation supports the pinned local CPU and MPS runtimes")
    runtime, _ = runtime_inventory(backend, user_privacy=False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(CPU_THREADS)
    cache = root / ".cache" / "huggingface" / "hub"
    if cache.resolve() != cache:
        raise ValueError("evaluation cache must be inside the local root")
    directory = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache, local_files_only=True,
    ))
    if directory.name != MODEL_REVISION or not directory.is_relative_to(cache):
        raise ValueError("evaluation model revision mismatch")
    assets = {}
    for filename in ASSETS:
        path = directory / filename
        if not path.resolve().is_relative_to(cache) or not path.is_file():
            raise ValueError("missing local pinned evaluation asset")
        assets[filename] = sha256_file(path)
    runtime.update({
        "torch_threads": torch.get_num_threads(), "tokenizers_parallelism": os.environ["TOKENIZERS_PARALLELISM"],
        "numeric_threads_lr": 1, "joblib": version("joblib"), "threadpoolctl": version("threadpoolctl"),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "assets": assets,
    })
    return directory, runtime


def open_predictor(campaign, run, model_directory, backend):
    family = run["model_family"]
    if family == "majority":
        return lambda texts, check: np.full(len(texts), campaign.lock["majority_label"], dtype=np.int64)
    if family == "tfidf_lr":
        import joblib
        from threadpoolctl import threadpool_limits

        vectorizer = joblib.load(campaign.checked_file(*run["parents"]["vectorizer"]))
        model = joblib.load(campaign.checked_file(*run["parents"]["model"]))
        checked_iterations(model)

        def predict(texts, check):
            check()
            with threadpool_limits(limits=1):
                return decision_scores(model, vectorizer.transform(texts))

        return predict

    import torch

    from .transformer import distilbert_assets, encode_role, infer

    config = campaign.checked_json(*run["parents"]["training_config"])
    checkpoint = campaign.checked_file(*run["parents"]["checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (state["schema_version"] != 1 or state["compatibility"] != config
            or state["progress"]["epoch"] != run["epoch"]
            or state["progress"]["optimizer_steps"] != run["steps"]
            or state["progress"]["next_position"] != config["lineage"]["rows"]["train"]):
        raise ValueError("selected checkpoint does not match locked config and training progress")
    device = torch.device(backend)
    tokenizer, model = distilbert_assets(model_directory, config, device)
    current = model.state_dict()
    if any(state["model"][key].dtype != value.dtype for key, value in current.items()):
        raise ValueError("selected checkpoint model dtype mismatch")
    model.load_state_dict(state["model"], strict=True)

    def predict(texts, check):
        check()
        encoded, _ = encode_role(tokenizer, list(texts))
        return infer(model, tokenizer, encoded, range(len(encoded)), BATCH, device,
                     check_deadline=check)

    return predict


def ordered_text(records, rows, role, family):
    if family != "distilbert":
        return aligned_text(records, rows, role)
    # Check the same alignment and text projection used for training and selection.
    from json import loads

    return tuple(loads(line)["text"] for line in aligned_projection(rows, records, role).splitlines())
