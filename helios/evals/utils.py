"""Utility functions for the evals."""

import math
from collections.abc import Sequence

import torch
from torch.utils.data import default_collate

from helios.train.masking import MaskedHeliosSample


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: float,
    warmup_epochs: int,
    total_epochs: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            )
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def eval_collate_fn(
    batch: Sequence[tuple[MaskedHeliosSample, torch.Tensor]],
) -> tuple[MaskedHeliosSample, torch.Tensor]:
    """Collate function for eval DataLoaders."""
    samples, targets = zip(*batch)
    # we assume that the same values are consistently None
    collated_sample = default_collate([s.as_dict(return_none=False) for s in samples])
    collated_target = default_collate([t for t in targets])
    return MaskedHeliosSample(**collated_sample), collated_target
