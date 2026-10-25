"""WeatherBench-style deterministic metrics with one latitude weight factor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from .data import CHANNELS, LATITUDES, Split


def latitude_weights(latitudes: torch.Tensor) -> torch.Tensor:
    weights = torch.cos(torch.deg2rad(latitudes.to(torch.float64)))
    if torch.any(weights <= 0):
        raise ValueError("latitudes must lie in the open interval (-90, 90)")
    return (weights / weights.mean()).to(torch.float32)


def weighted_rmse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Global time/space RMSE, independently for each channel."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(
            f"pred and target must have matching (N,C,H,W) shapes; "
            f"received {pred.shape} and {target.shape}"
        )
    w = weights.to(pred.device, pred.dtype).view(1, 1, -1, 1)
    return ((pred - target).square() * w).mean(dim=(0, 2, 3)).sqrt()


def weighted_bias(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(pred.device, pred.dtype).view(1, 1, -1, 1)
    return ((pred - target) * w).mean(dim=(0, 2, 3))


def anomaly_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    climatology: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Mean of per-initialization spatial ACC; latitude weights occur once."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("pred and target must have matching (N,C,H,W) shapes")
    if climatology.shape != pred.shape:
        raise ValueError("climatology must align with pred and target at every valid time")
    w = weights.to(pred.device, pred.dtype).view(1, 1, -1, 1)
    pa, ta = pred - climatology, target - climatology
    numerator = (w * pa * ta).sum(dim=(-2, -1))
    denominator = torch.sqrt(
        (w * pa.square()).sum(dim=(-2, -1))
        * (w * ta.square()).sum(dim=(-2, -1))
    )
    if torch.any(denominator <= 0):
        raise ValueError("ACC denominator is zero; a constant value cannot be returned")
    return (numerator / denominator).mean(dim=0)


def denormalize(values: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return values * stds.view(1, -1, 1, 1) + means.view(1, -1, 1, 1)


def persistence_rmse(
    canonical_path: str | Path,
    split: Split,
    *,
    lead_steps: int = 1,
    max_samples: int | None = None,
    block_size: int = 256,
) -> dict[str, float]:
    """Evaluate a physical persistence forecast without loading all data at once."""
    if lead_steps < 1:
        raise ValueError("lead_steps must be at least 1")
    array = zarr.open_group(Path(canonical_path), mode="r")["state"]
    count = split.sample_count(lead_steps)
    if max_samples is not None:
        count = min(count, max_samples)
    if count < 1:
        raise ValueError("the selected split contains no evaluable samples")
    weights = np.cos(np.deg2rad(LATITUDES.astype(np.float64)))
    weights /= weights.mean()
    squared_error = np.zeros(8, dtype=np.float64)
    for offset in range(0, count, block_size):
        size = min(block_size, count - offset)
        start = split.start + offset
        forecast = np.asarray(array[start : start + size], dtype=np.float64)
        truth = np.asarray(array[start + lead_steps : start + lead_steps + size], dtype=np.float64)
        squared_error += (
            np.square(forecast - truth) * weights[None, None, :, None]
        ).sum(axis=(0, 2, 3), dtype=np.float64)
    rmse = np.sqrt(squared_error / (count * len(LATITUDES) * 64))
    return {
        "lead_hours": int(lead_steps * 6),
        "split": split.name,
        "samples": int(count),
        **{name: float(rmse[i]) for i, name in enumerate(CHANNELS)},
    }
