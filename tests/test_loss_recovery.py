import copy
import random
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from disaster_triage.checkpoint import checkpoint_due, restore_checkpoint, save_checkpoint
from disaster_triage.config import RunConfig
from disaster_triage.loss import Batch, backward_window, class_weights, make_optimizer, optimizer_window
from disaster_triage.provenance import sha256_file
from disaster_triage.runtime import seed_runtime
from disaster_triage.smoke import TinyClassifier

@pytest.mark.parametrize("physical", [8, 4, 2])
@pytest.mark.parametrize("n", [16, 13, 1])
def test_accumulation_matches_full_batch_loss_gradients_and_adamw(physical, n):
    torch.manual_seed(42)
    full = TinyClassifier().eval()
    accumulated = copy.deepcopy(full)
    x, y = torch.randn(n, 6), torch.tensor(([0] * 7 + [9] * 4 + [2] * 5)[:n])
    weights = class_weights([1, 2, 3, 5, 7, 11, 13, 17, 19, 31])
    reference = F.cross_entropy(full(x), y, weight=weights)
    reference.backward()
    actual = backward_window(
        accumulated, [Batch({"features": x[i:i+physical]}, y[i:i+physical])
                      for i in range(0, n, physical)], weights)
    assert actual == pytest.approx(reference.item(), rel=1e-5, abs=1e-6)
    for a, b in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
    torch.nn.utils.clip_grad_norm_(full.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(accumulated.parameters(), 1.)
    left, _ = make_optimizer(full, RunConfig(), 4)
    right, _ = make_optimizer(accumulated, RunConfig(), 4)
    left.step()
    right.step()
    for a, b in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

def test_independent_microbatch_means_would_change_objective():
    torch.manual_seed(42)
    model = TinyClassifier().eval()
    x, y = torch.randn(16, 6), torch.tensor([0] * 8 + [9] * 8)
    weights = class_weights(list(range(1, 11)))
    whole = F.cross_entropy(model(x), y, weight=weights)
    wrong = sum(F.cross_entropy(model(x[i:i+8]), y[i:i+8], weight=weights)
                for i in (0, 8)) / 2
    assert not torch.isclose(whole, wrong, rtol=1e-4)

@pytest.mark.parametrize("counts", [[1] * 9, [1] * 9 + [0], [True] * 10, [1.] * 10])
def test_invalid_training_counts(counts):
    with pytest.raises(ValueError):
        class_weights(counts)

def _step(model, optimizer, scheduler, generator, progress, x, y):
    progress = copy.deepcopy(progress)
    if progress["next_position"] == len(progress["permutation"]):
        progress["epoch"] += 1
        progress["permutation"] = torch.randperm(len(y), generator=generator).tolist()
        progress["next_position"] = 0
    start = progress["next_position"]
    ids = progress["permutation"][start:start+16]
    optimizer_window(model, [
        Batch({"features": x[ids[:8]]}, y[ids[:8]]),
        Batch({"features": x[ids[8:]]}, y[ids[8:]]),
    ], class_weights([1] * 10), optimizer, scheduler)
    progress["next_position"] += 16
    progress["optimizer_steps"] += 1
    return progress

def test_dropout_recovery_at_step_100_restores_order_rng_and_scheduler(tmp_path):
    device = torch.device("cpu")
    generator = seed_runtime(42, device)
    x, y = torch.randn(32, 6), torch.arange(32) % 10
    model = TinyClassifier()
    optimizer, scheduler = make_optimizer(model, RunConfig(), 105)
    progress = {"epoch": 0, "optimizer_steps": 0,
                "permutation": torch.randperm(32, generator=generator).tolist(), "next_position": 0}
    compatibility = {"config": "synthetic", "horizon": 105, "device": "cpu"}
    path = tmp_path / "step-100.pt"
    for _ in range(100):
        progress = _step(model, optimizer, scheduler, generator, progress, x, y)
    assert checkpoint_due(100) and checkpoint_due(3, epoch_end=True) and not checkpoint_due(99)
    digest = save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, progress=progress,
        compatibility=compatibility, device=device, order_generator=generator,
        lineage={"run_id": "synthetic", "attempt_id": "first", "active_seconds": 1.},
    )
    wanted_progress = _step(model, optimizer, scheduler, generator, progress, x, y)
    wanted_parameters = copy.deepcopy(model.state_dict())
    wanted_scheduler = copy.deepcopy(scheduler.state_dict())
    wanted_draws = (random.random(), np.random.random(), torch.rand(3))
    restored = TinyClassifier()
    other_optimizer, other_scheduler = make_optimizer(restored, RunConfig(), 105)
    state = restore_checkpoint(
        path, model=restored, optimizer=other_optimizer, scheduler=other_scheduler,
        expected=compatibility, device=device, order_generator=generator, expected_sha256=digest,
    )
    actual_progress = _step(restored, other_optimizer, other_scheduler, generator,
                            state["progress"], x, y)
    assert actual_progress == wanted_progress
    assert wanted_scheduler == other_scheduler.state_dict()
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, wanted_parameters[key], rtol=0, atol=0)
    assert random.random() == wanted_draws[0]
    assert np.random.random() == wanted_draws[1]
    torch.testing.assert_close(torch.rand(3), wanted_draws[2], rtol=0, atol=0)
    with pytest.raises(ValueError, match="mismatch"):
        restore_checkpoint(path, model=restored, optimizer=other_optimizer, scheduler=other_scheduler,
                           expected={"different": True}, device=device, order_generator=generator)
    path.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="digest"):
        restore_checkpoint(path, model=restored, optimizer=other_optimizer, scheduler=other_scheduler,
                           expected=compatibility, device=device, order_generator=generator,
                           expected_sha256=digest)

def test_atomic_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    device = torch.device("cpu")
    generator = seed_runtime(42, device)
    model = TinyClassifier()
    optimizer, scheduler = make_optimizer(model, RunConfig(), 4)
    path = tmp_path / "checkpoint.pt"
    kwargs = dict(model=model, optimizer=optimizer, scheduler=scheduler,
                  progress={"epoch": 0, "optimizer_steps": 0, "permutation": [0], "next_position": 0},
                  compatibility={}, device=device, order_generator=generator, lineage={})
    digest = save_checkpoint(path, **kwargs)
    def fail(state, target):
        with open(target, "wb") as stream:
            stream.write(b"partial")
        raise OSError("synthetic write failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_checkpoint(path, **kwargs)
    assert sha256_file(path) == digest
    assert list(tmp_path.iterdir()) == [path]
