"""Explicit synthetic verification jobs; never reads HumAID."""
import json
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from .accounting import Attempt, guard_budget, merge_accounting
from .bundles import BundleExpectation, RoleRow, read_bundle, write_bundle, write_role_manifest
from .checkpoint import restore_checkpoint, save_checkpoint
from .config import save_resolved
from .constants import LABELS, MODEL_ID
from .loss import Batch, class_weights, make_optimizer, optimizer_window
from .provenance import save_json, sha256_bytes, sha256_file, source_snapshot
from .runtime import configure_cache, seed_runtime, select_device, versions

COUNTS = tuple(range(1, 11))

class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.2)
        self.linear = torch.nn.Linear(6, 10)
    def forward(self, features):
        return self.linear(self.dropout(features))

def synthetic_data(device):
    generator = torch.Generator().manual_seed(901)
    x = torch.randn(sum(COUNTS), 6, generator=generator).to(device)
    y = torch.repeat_interleave(torch.arange(10), torch.tensor(COUNTS)).to(device)
    return x, y

def _batches(x, y, permutation, start, config):
    indices = permutation[start:start + 16]
    return [Batch({"features": x[indices[i:i + config.physical_batch]]},
                  y[indices[i:i + config.physical_batch]])
            for i in range(0, len(indices), config.physical_batch)]

def _job(config, root, source_root, run_id, model_revision, total_steps=4):
    if config.mode != "synthetic":
        raise ValueError("smoke accepts synthetic mode only")
    configure_cache(config, root)
    device = select_device(config.device)
    order = seed_runtime(config.seed, device)
    artifact = config.paths(root)["artifact_dir"]
    guard_budget(artifact)
    directory = artifact / "verification" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(source_root, directory / "source.tar")
    resolved, config_hash = save_resolved(
        directory / "resolved-config.json", config, versions=versions(),
        model_revision=model_revision, training_counts=COUNTS, scheduler_total_steps=total_steps,
    )
    data_hash = sha256_bytes(b"synthetic-v1:901:features-55x6:counts-1-through-10")
    compatibility = {
        "config_sha256": config_hash, "code_sha256": source["code_sha256"],
        "resolved_config": resolved,
        "data_sha256": data_hash, "versions": versions(), "device": str(device),
        "model_revision": model_revision, "scheduler_total_steps": total_steps,
    }
    metadata = {**compatibility, "seed": config.seed, "code_revision": source["code_revision"]}
    metadata.pop("scheduler_total_steps")
    metadata.pop("resolved_config")
    save_json(directory / "source-manifest.json", source)
    return directory, artifact, device, order, compatibility, metadata

def _assert_models(left, right):
    for key, value in left.items():
        torch.testing.assert_close(value.cpu(), right[key].cpu(), rtol=1e-5, atol=1e-6)

def export_fixtures(directory, compatibility):
    rows = (RoleRow("example-001", "synthetic_event_alpha", 7),
            RoleRow("example-002", "synthetic_event_alpha", 5))
    role_hash = write_role_manifest(directory / "role.jsonl", rows)
    digest = sha256_bytes(b"explicitly synthetic source")
    expected = BundleExpectation("synthetic", "model_selection", rows, {
        "archive_sha256": digest, "eligible_manifest_sha256": sha256_bytes(b"synthetic eligible"),
        "role_manifest_sha256": role_hash, "config_sha256": compatibility["config_sha256"],
        "code_sha256": compatibility["code_sha256"],
    })
    common = {"row_id": np.array([r.row_id for r in rows], dtype="U"),
              "event_id": np.array([r.event_id for r in rows], dtype="U"),
              "y_true": np.array([r.y_true for r in rows], dtype=np.int64)}
    logits = np.zeros((2, 10), dtype=np.float32)
    logits[0, 7], logits[1, 5] = 2., 2.
    for family, prediction in (
        ("tfidf_lr", {"logits": logits}),
        ("majority", {"y_pred": np.array([5, 5], dtype=np.int64)}),
    ):
        target = directory / "bundles" / family / expected.role
        write_bundle(target, expected=expected, run_id="synthetic-" + family,
                     model_family=family, seed=None, arrays={**common, **prediction})
        read_bundle(target, expected=expected)
    save_json(directory / "bundle-expectation.json", {
        "mode": "synthetic", "role": expected.role, "digests": expected.digests,
        "rows": [r.__dict__ for r in rows],
        "note": "Invented bundle fixtures, not fitted classifier outputs.",
    })

def run_offline(config, root, source_root):
    run_id = "synthetic-" + uuid.uuid4().hex[:12]
    directory, artifact, device, order, compatibility, metadata = _job(
        config, root, source_root, run_id, "offline-tiny-v1",
    )
    with Attempt(artifact, run_id, metadata=metadata) as attempt:
        model = TinyClassifier().to(device)
        optimizer, scheduler = make_optimizer(model, config, 4)
        x, y = synthetic_data(device)
        permutation = torch.randperm(len(y), generator=order).tolist()
        weights = class_weights(COUNTS).to(device)
        optimizer_window(model, _batches(x, y, permutation, 0, config), weights, optimizer, scheduler)
        progress = {"epoch": 0, "optimizer_steps": 1, "permutation": permutation, "next_position": 16}
        checkpoint = directory / "recovery-probe.pt"
        checkpoint_hash = save_checkpoint(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, progress=progress,
            compatibility=compatibility, device=device, order_generator=order,
            lineage={"run_id": run_id, "attempt_id": attempt.attempt_id,
                     "active_seconds_before_save": attempt.monotonic() - attempt.started},
        )
        optimizer_window(model, _batches(x, y, permutation, 16, config), weights, optimizer, scheduler)
        reference = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(reference, directory / "expected-after.pt")
        draws = [random.random(), float(np.random.random()), float(torch.rand(1).item())]
        save_json(directory / "probe-expectation.json", {
            "run_id": run_id, "compatibility": compatibility,
            "checkpoint_sha256": checkpoint_hash,
            "reference_sha256": sha256_file(directory / "expected-after.pt"),
            "draws": draws,
        })
        restored = TinyClassifier().to(device)
        other_optimizer, other_scheduler = make_optimizer(restored, config, 4)
        state = restore_checkpoint(
            checkpoint, model=restored, optimizer=other_optimizer, scheduler=other_scheduler,
            expected=compatibility, device=device, order_generator=order,
            expected_sha256=checkpoint_hash,
        )
        optimizer_window(restored, _batches(x, y, state["progress"]["permutation"], 16, config),
                         weights, other_optimizer, other_scheduler)
        _assert_models(reference, restored.state_dict())
        assert draws == [random.random(), float(np.random.random()), float(torch.rand(1).item())]
        export_fixtures(directory, compatibility)
        attempt.update(progress={"epoch": 0, "optimizer_steps": 2, "next_position": 32},
                       artifact_sha256={"checkpoint": checkpoint_hash,
                                        "config": compatibility["config_sha256"],
                                        "source": compatibility["code_sha256"]})
        result = {"stage": "offline", "run_id": run_id, "device": str(device),
                  "recovery_equivalent": True, "bundle_roundtrips": 2,
                  "verification_dir": str(directory), "versions": versions()}
        save_json(directory / "summary.json", result)
    merge_accounting(artifact, directory / "accounting")
    return result

def resume_probe(config, root, source_root, probe_directory):
    probe_directory = Path(probe_directory)
    probe = json.loads((probe_directory / "probe-expectation.json").read_text())
    run_id = probe["run_id"]
    configure_cache(config, root)
    device = select_device(config.device)
    order = seed_runtime(config.seed, device)
    artifact = config.paths(root)["artifact_dir"]
    retained_accounting = probe_directory / "accounting"
    if not (retained_accounting / "runs").is_dir():
        raise ValueError("retrieve the probe with its retained accounting records")
    merge_accounting(retained_accounting, artifact)
    guard_budget(artifact)
    # Independently recreate expectations from caller's config/runtime/code.
    scratch = artifact / "resume-checks" / uuid.uuid4().hex
    source = source_snapshot(source_root, scratch / "source.tar")
    resolved, config_hash = save_resolved(scratch / "resolved-config.json", config, versions=versions(),
                                  model_revision="offline-tiny-v1", training_counts=COUNTS, scheduler_total_steps=4)
    expected = {
        "config_sha256": config_hash, "code_sha256": source["code_sha256"],
        "resolved_config": resolved,
        "data_sha256": sha256_bytes(b"synthetic-v1:901:features-55x6:counts-1-through-10"),
        "versions": versions(), "device": str(device), "model_revision": "offline-tiny-v1",
        "scheduler_total_steps": 4,
    }
    if probe["compatibility"] != expected:
        raise ValueError("retrieved probe does not match independently resolved runtime/code/config")
    metadata = {k: v for k, v in expected.items() if k not in ("scheduler_total_steps", "resolved_config")}
    metadata["parent_checkpoint_sha256"] = probe["checkpoint_sha256"]
    with Attempt(artifact, run_id, metadata=metadata) as attempt:
        model = TinyClassifier().to(device)
        optimizer, scheduler = make_optimizer(model, config, 4)
        x, y = synthetic_data(device)
        state = restore_checkpoint(
            probe_directory / "recovery-probe.pt", model=model, optimizer=optimizer,
            scheduler=scheduler, expected=expected, device=device, order_generator=order,
            expected_sha256=probe["checkpoint_sha256"],
        )
        optimizer_window(model, _batches(x, y, state["progress"]["permutation"],
                                        state["progress"]["next_position"], config),
                         class_weights(COUNTS).to(device), optimizer, scheduler)
        reference = probe_directory / "expected-after.pt"
        if sha256_file(reference) != probe["reference_sha256"]:
            raise ValueError("retrieved reference digest mismatch")
        _assert_models(torch.load(reference, map_location="cpu", weights_only=True), model.state_dict())
        assert probe["draws"] == [random.random(), float(np.random.random()), float(torch.rand(1).item())]
        attempt.update(progress={"epoch": 0, "optimizer_steps": 2, "next_position": 32})
        result = {"stage": "retrieved_resume", "run_id": run_id, "device": str(device),
                  "recovery_equivalent": True}
        save_json(scratch / "summary.json", result)
    return result

def run_pretrained(config, root, source_root):
    configure_cache(config, root)
    revision = config.model_revision
    if not revision or len(revision) != 40:
        raise ValueError("model revision did not resolve to an immutable commit")
    run_id = "pretrained-synthetic-" + uuid.uuid4().hex[:12]
    directory, artifact, device, order, compatibility, metadata = _job(
        config, root, source_root, run_id, revision, total_steps=1,
    )
    with Attempt(artifact, run_id, metadata=metadata) as attempt:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=revision, num_labels=10,
            id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
            use_safetensors=True, attn_implementation="eager",
        ).to(device=device, dtype=torch.float32)
        texts = ["Invented example requesting water.",
                 "Synthetic volunteers collect blankets. " * 80]
        lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts]
        encoded = tokenizer(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in encoded.items()}
        optimizer, scheduler = make_optimizer(model, config, 1)
        loss = optimizer_window(model, [Batch(inputs, torch.tensor([7, 8], device=device))],
                                class_weights(COUNTS).to(device), optimizer, scheduler)
        if not np.isfinite(loss):
            raise ValueError("pretrained smoke produced nonfinite loss")
        result = {
            "stage": "pretrained", "run_id": run_id, "device": str(device),
            "model_id": MODEL_ID, "model_revision": revision, "tokenizer_revision": revision,
            "loss_finite": True, "output_shape": [2, 10], "truncated_count": sum(n > 128 for n in lengths),
            "attention_implementation": "eager", "versions": versions(),
        }
        attempt.update(progress={"epoch": 0, "optimizer_steps": 1, "next_position": 2})
        save_json(directory / "summary.json", result)
    return result
