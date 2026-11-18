"""Complete-window weighted CE, independent of model default loss."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class Batch:
    inputs: dict
    labels: torch.Tensor

def class_weights(counts):
    if len(counts) != 10 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("all ten training-only counts must be positive integers")
    return torch.tensor([sum(counts) / (10 * n) for n in counts], dtype=torch.float32)

def backward_window(model, microbatches, weights):
    """Accumulate gradients; caller clips and steps exactly once."""
    if not microbatches:
        raise ValueError("empty accumulation window")
    if weights.shape != (10,) or weights.dtype != torch.float32:
        raise ValueError("expected ten FP32 class weights")
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("weights must be positive finite")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("model parameters must be FP32")
    for batch in microbatches:
        y = batch.labels
        if y.ndim != 1 or y.dtype != torch.int64 or y.numel() == 0:
            raise ValueError("labels must be nonempty int64 vectors")
        if not ((y >= 0) & (y < 10)).all():
            raise ValueError("label index out of range")
    denominator = sum(weights[batch.labels].sum() for batch in microbatches)
    loss_value = 0.
    for batch in microbatches:
        output = model(**batch.inputs)
        logits = output.logits if hasattr(output, "logits") else output
        if logits.shape != (len(batch.labels), 10) or logits.dtype != torch.float32:
            raise ValueError("expected N by 10 FP32 logits")
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite logits")
        loss = F.cross_entropy(logits, batch.labels, weight=weights,
                               reduction="sum", label_smoothing=0.) / denominator
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        loss.backward()
        loss_value += loss.detach().item()
    return loss_value

def optimizer_window(model, microbatches, weights, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    try:
        loss = backward_window(model, microbatches, weights)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.zero_grad(set_to_none=True)
    return loss

def make_optimizer(model, config, total_steps):
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("scheduler horizon must be positive")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        betas=(config.beta1, config.beta2), eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0., 1. - step / total_steps),
    )
    return optimizer, scheduler
