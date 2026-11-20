"""Train and score independent DistilBERT runs on the two admitted roles."""

from dataclasses import asdict
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .bundles import BundleExpectation, write_bundle
from .checkpoint import checkpoint_due, restore_checkpoint, save_checkpoint
from .config import RunConfig
from .constants import ARCHIVE_SHA256, LABELS, MODEL_ID, MODEL_REVISION
from .evaluation import BundleInput, SeedCohort, score_bundle, summarize_seeds
from .loss import Batch, class_weights, make_optimizer, optimizer_window
from .provenance import json_bytes, save_json, sha256_file, sha256_bytes
from .runtime import seed_runtime, select_device
from .transformer_data import read_json

SCREEN_KEYS = ("seed42-lr2e-5", "seed42-lr5e-5")
REPEAT_KEYS = ("seed43", "seed44")
KEYS = SCREEN_KEYS + REPEAT_KEYS
TOLERANCE = .001
PILOT_STEPS = 100
PILOT_SAMPLE = 256
EPOCH_FILES = ("checkpoint.pt", "eval-config.json", "report.json",
               "model_selection/manifest.json", "model_selection/predictions.npz")


def steps_per_epoch(rows):
    return math.ceil(rows / 16)


def select_screen(scores):
    """Rank the complete declared search using the unrounded maximum."""
    wanted = {(key, epoch) for key in SCREEN_KEYS for epoch in range(1, 4)}
    if scores.keys() != wanted or not all(math.isfinite(x) and 0 <= x <= 1 for x in scores.values()):
        raise ValueError("Selection requires all six completed screening scores.")
    best = max(scores.values())
    return min(key for key, score in scores.items() if best - score <= TOLERANCE + 1e-15)


def resolved_run(key, *, campaign, inputs, source, runtime, physical_batch, device, selection=None):
    if key in SCREEN_KEYS:
        seed, rate, stop = 42, (2e-5 if key == SCREEN_KEYS[0] else 5e-5), 3
    elif key in REPEAT_KEYS and selection is not None:
        seed = int(key[-2:])
        rate, stop = selection["learning_rate"], selection["epoch"]
    else:
        raise ValueError("A repeat requires complete screening selection.")
    config = RunConfig(
        device=device, physical_batch=physical_batch, accumulation=16 // physical_batch,
        seed=seed, learning_rate=rate, max_epochs=stop,
    )
    recipe = asdict(config)
    recipe["mode"] = "research" if inputs.lineage["archive_sha256"] == ARCHIVE_SHA256 else "synthetic"
    return {
        **recipe, "schema_version": 1, "campaign": campaign, "key": key,
        "run_id": f"transformer-{campaign}-{key}",
        "model_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION, "attention": "eager",
        "labels": list(LABELS), "lineage": inputs.lineage, "source": source,
        "runtime": runtime, "effective_batch": 16, "scheduler": "linear",
        "warmup_steps": 0, "optimizer": "AdamW", "decay_policy": "all_parameters",
        "training_counts": inputs.lineage["training_counts"],
        "weights": class_weights(inputs.lineage["training_counts"]).tolist(),
        "steps_per_epoch": steps_per_epoch(len(inputs.rows["train"])),
        "scheduler_total_steps": 3 * steps_per_epoch(len(inputs.rows["train"])),
        "selection_sha256": sha256_bytes(json_bytes(selection)) if key in REPEAT_KEYS else None,
    }


def distilbert_assets(source, config, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=10, id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
        local_files_only=True, use_safetensors=True, attn_implementation=config["attention"],
    ).to(device=device, dtype=torch.float32)
    return tokenizer, model


def encode_role(tokenizer, texts):
    """Tokenize in bounded chunks and retain only truncated model inputs."""
    encoded, truncated = [], 0
    for start in range(0, len(texts), 256):
        batch = tokenizer(texts[start:start + 256], truncation=False, verbose=False)
        for ids in batch["input_ids"]:
            if len(ids) > 128:
                truncated += 1
                ids = [*ids[:127], tokenizer.sep_token_id]
            encoded.append({"input_ids": ids})
    return encoded, truncated


def batch_inputs(tokenizer, encoded, indices, device):
    padded = tokenizer.pad([encoded[index] for index in indices], padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in padded.items()}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def infer(model, tokenizer, encoded, indices, batch_size, device):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            inputs = batch_inputs(tokenizer, encoded, indices[start:start + batch_size], device)
            logits = model(**inputs).logits
            if logits.dtype != torch.float32 or logits.shape != (len(inputs["input_ids"]), 10):
                raise ValueError("Expected N-by-10 FP32 inference logits.")
            chunks.append(logits.cpu().numpy())
    synchronize(device)
    scores = np.concatenate(chunks)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite inference logits.")
    return scores


def bundle_input(directory, config, rows, config_sha256):
    lineage = config["lineage"]
    expected = BundleExpectation(config["mode"], "model_selection", rows, {
        "archive_sha256": lineage["archive_sha256"],
        "eligible_manifest_sha256": lineage["eligible_manifest_sha256"],
        "role_manifest_sha256": lineage["role_sha256"]["model_selection"],
        "code_sha256": config["source"]["code_sha256"],
        "config_sha256": config_sha256,
    })
    return BundleInput(directory / "model_selection", expected, config["run_id"],
                       "distilbert", config["seed"])


def publish_epoch(directory, config, inputs, epoch, checkpoint, logits, attempt_id, checkpoint_attempt_id):
    target = directory / "epochs" / f"e{epoch}"
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".epoch-", dir=target.parent) as staging:
        staged = Path(staging)
        os.link(checkpoint, staged / "checkpoint.pt")
        evaluation = {**config, "evaluated_epoch": epoch,
                      "checkpoint_sha256": sha256_file(staged / "checkpoint.pt")}
        digest = save_json(staged / "eval-config.json", evaluation)
        view = bundle_input(staged, config, inputs.rows["model_selection"], digest)
        arrays = {
            "row_id": np.asarray([row.row_id for row in view.expected.rows], dtype=str),
            "event_id": np.asarray([row.event_id for row in view.expected.rows], dtype=str),
            "y_true": np.asarray([row.y_true for row in view.expected.rows], dtype=np.int64),
            "logits": logits,
        }
        write_bundle(view.directory, expected=view.expected, run_id=view.run_id,
                     model_family=view.model_family, seed=view.seed, arrays=arrays,
                     code_revision=config["source"]["code_revision"])
        save_json(staged / "report.json", score_bundle(view))
        save_json(staged / "epoch.json", {
            "epoch": epoch, "attempt_id": attempt_id, "checkpoint_attempt_id": checkpoint_attempt_id,
            "config_sha256": sha256_bytes(json_bytes(config)),
            "files": {name: sha256_file(staged / name) for name in EPOCH_FILES},
        })
        staged.rename(target)


def verify_epoch(directory, config, inputs, epoch, *, load_checkpoint=False):
    target = directory / "epochs" / f"e{epoch}"
    receipt = read_json(target / "epoch.json")
    if (receipt["epoch"] != epoch or set(receipt["files"]) != set(EPOCH_FILES)
            or receipt["config_sha256"] != sha256_bytes(json_bytes(config))
            or any(sha256_file(target / name) != digest for name, digest in receipt["files"].items())):
        raise ValueError("Epoch receipt does not match its artifacts and resolved run.")
    expected = {**config, "evaluated_epoch": epoch,
                "checkpoint_sha256": receipt["files"]["checkpoint.pt"]}
    if (target / "eval-config.json").read_bytes() != json_bytes(expected):
        raise ValueError("Evaluation config does not bind the correct checkpoint.")
    view = bundle_input(target, config, inputs.rows["model_selection"], receipt["files"]["eval-config.json"])
    report = score_bundle(view)
    if json_bytes(report) != (target / "report.json").read_bytes():
        raise ValueError("Stored aggregate report differs from replayed metrics.")
    if load_checkpoint:
        state = torch.load(target / "checkpoint.pt", map_location="cpu", weights_only=True)
        progress = state["progress"]
        if (state["compatibility"] != config or state["schema_version"] != 1
                or progress["epoch"] != epoch or progress["optimizer_steps"] != epoch * config["steps_per_epoch"]
                or progress["next_position"] != len(inputs.rows["train"]) or not state["model"]
                or not state["optimizer"] or not state["scheduler"] or not state["rng"]
                or state["lineage"].get("attempt_id") != receipt["checkpoint_attempt_id"]):
            raise ValueError("Selected checkpoint is incomplete or does not match epoch progress.")
    return report, receipt


def verify_rolling(directory):
    pointer = directory / "latest.json"
    if not pointer.exists():
        return None
    latest = read_json(pointer)
    filename = latest["file"]
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Invalid recovery checkpoint name.")
    checkpoint = directory / "rolling" / filename
    if checkpoint.is_symlink() or sha256_file(checkpoint) != latest["sha256"]:
        raise ValueError("Rolling recovery checkpoint failed digest verification.")
    return latest


def publish_recovery(directory, model, optimizer, scheduler, progress, config, device,
                     order, attempt_id, parent_sha256, previous):
    from secrets import token_hex
    name = f"{attempt_id}-{progress['optimizer_steps']}-{token_hex(4)}.pt"
    target = directory / "rolling" / name
    digest = save_checkpoint(
        target, model=model, optimizer=optimizer, scheduler=scheduler, progress=progress,
        compatibility=config, device=device, order_generator=order,
        lineage={"attempt_id": attempt_id, "parent_checkpoint_sha256": parent_sha256},
    )
    latest = {
        "file": name, "sha256": digest, "attempt_id": attempt_id,
        "progress": {key: value for key, value in progress.items() if key != "permutation"},
    }
    save_json(directory / "latest.json", latest)
    if previous is not None:
        (directory / "rolling" / previous["file"]).unlink(missing_ok=True)
    return latest


def do_training(directory, config, inputs, model_root, stage, attempt_id, deadline, *,
                latest, ready_event, go_event, halfway_event, trial=None, load_model=distilbert_assets):
    """Advance only complete optimizer windows; pause at the saved pilot boundary."""
    from types import SimpleNamespace

    device = select_device(config["device"])
    order = seed_runtime(config["seed"], device)
    model_started = time.monotonic()
    tokenizer, model = load_model(model_root, config, device)
    train, train_truncated = encode_role(tokenizer, inputs.texts["train"])
    selection, selection_truncated = encode_role(tokenizer, inputs.texts["model_selection"])
    labels = [row.y_true for row in inputs.rows["train"]]
    weights = torch.tensor(config["weights"], dtype=torch.float32, device=device)
    optimizer, scheduler = make_optimizer(model, SimpleNamespace(**config), config["scheduler_total_steps"])
    startup_seconds = time.monotonic() - model_started
    parent_digest = None if latest is None else latest["sha256"]
    progress = {"epoch": 1, "optimizer_steps": 0, "next_position": 0, "permutation": []}
    if latest is not None:
        state = restore_checkpoint(
            directory / "rolling" / latest["file"], model=model, optimizer=optimizer,
            scheduler=scheduler, expected=config, device=device, order_generator=order,
            expected_sha256=parent_digest,
        )
        progress = state["progress"]
        del state
    if ready_event is not None:
        ready_event.set()
    journal = directory / "pilot-progress.json"
    previous_pilot = read_json(journal) if stage == "pilot" and journal.exists() else {}
    timing = previous_pilot.get("step_seconds", [])
    checkpoint_seconds = previous_pilot.get("checkpoint_seconds", [])
    pilot_path = directory / "pilot.json"
    pilot_goal = min(PILOT_STEPS, config["steps_per_epoch"])
    if stage == "pilot" and progress["optimizer_steps"] > pilot_goal:
        raise ValueError("Pilot cannot start after the pilot step boundary.")
    done = False
    while not done:
        epoch = progress["epoch"]
        if not progress["permutation"]:
            progress["permutation"] = torch.randperm(len(train), generator=order).tolist()
        current_position = progress["next_position"]
        if current_position < len(train) and (stage != "pilot" or progress["optimizer_steps"] < pilot_goal):
            if time.monotonic() >= deadline:
                done = True
                continue
            chosen = progress["permutation"][current_position:current_position + 16]
            model.train()
            started = time.monotonic()
            batches = []
            physical = config["physical_batch"]
            for start in range(0, len(chosen), physical):
                indices = chosen[start:start + physical]
                batches.append(Batch(batch_inputs(tokenizer, train, indices, device),
                                     torch.tensor([labels[i] for i in indices], device=device)))
            optimizer_window(model, batches, weights, optimizer, scheduler)
            synchronize(device)
            progress["optimizer_steps"] += 1
            progress["next_position"] += len(chosen)
            if stage == "pilot":
                timing = [item for item in timing if item[0] < progress["optimizer_steps"]]
                timing.append([progress["optimizer_steps"], started, time.monotonic()])
                save_json(journal, {"step_seconds": timing, "checkpoint_seconds": checkpoint_seconds})
                if halfway_event is not None and progress["optimizer_steps"] == 50:
                    halfway_event.set()
                    go_event.wait()
            at_end = progress["next_position"] == len(train)
            pilot_pause = stage == "pilot" and progress["optimizer_steps"] == pilot_goal
            if checkpoint_due(progress["optimizer_steps"], epoch_end=at_end) or pilot_pause:
                started = time.monotonic()
                latest = publish_recovery(directory, model, optimizer, scheduler, progress, config, device,
                                          order, attempt_id, parent_digest, latest)
                checkpoint_seconds.append(time.monotonic() - started)
                if stage == "pilot":
                    save_json(journal, {"step_seconds": timing, "checkpoint_seconds": checkpoint_seconds})
            continue

        if stage == "pilot":
            if latest is None or latest["progress"]["optimizer_steps"] != pilot_goal:
                raise ValueError("Pilot requires its verified optimizer-boundary checkpoint.")
            count = min(PILOT_SAMPLE, len(selection))
            sample = np.linspace(0, len(selection) - 1, count, dtype=int).tolist()
            started = time.monotonic()
            infer(model, tokenizer, selection, sample, config["physical_batch"], device)
            inference_seconds = time.monotonic() - started
            save_json(pilot_path, {
                "step_seconds": timing,
                "startup_seconds": startup_seconds, "checkpoint_seconds": max(checkpoint_seconds, default=0.),
                "checkpoint_bytes": (directory / "rolling" / latest["file"]).stat().st_size,
                "inference_seconds": inference_seconds, "inference_rows": count,
                "truncated": {"train": train_truncated, "model_selection": selection_truncated},
                "attempt_id": attempt_id, "trial": trial if not previous_pilot else None,
            })
            done = True
        else:
            epoch_dir = directory / "epochs" / f"e{epoch}"
            if not epoch_dir.exists() and (config["key"] in SCREEN_KEYS or epoch == config["max_epochs"]):
                if time.monotonic() >= deadline:
                    done = True
                    continue
                scores = infer(model, tokenizer, selection, list(range(len(selection))),
                               config["physical_batch"], device)
                publish_epoch(directory, config, inputs, epoch, directory / "rolling" / latest["file"],
                              scores, attempt_id, latest["attempt_id"])
            if epoch == config["max_epochs"]:
                save_json(directory / "complete.json", {
                    "last_epoch": epoch, "attempt_id": attempt_id,
                    "truncated": {"train": train_truncated, "model_selection": selection_truncated},
                })
                done = True
            else:
                progress = {**progress, "epoch": epoch + 1, "next_position": 0, "permutation": []}

    saved_steps = 0 if latest is None else latest["progress"]["optimizer_steps"]
    if progress["optimizer_steps"] > saved_steps and progress["permutation"]:
        publish_recovery(directory, model, optimizer, scheduler, progress, config, device,
                         order, attempt_id, parent_digest, latest)


def summarize_selected(reports, selection, configs):
    cohort = SeedCohort(
        settings_sha256=sha256_bytes(json_bytes({
            "learning_rate": selection["learning_rate"], "epoch": selection["epoch"],
            "scheduler_total_steps": selection["scheduler_total_steps"],
        })),
        epoch=selection["epoch"],
        config_sha256_by_seed={seed: configs[seed] for seed in sorted(reports)},
    )
    return summarize_seeds(reports.values(), cohort=cohort)
