"""True autoregressive model evaluation on held-out calendar blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr

from .data import CHANNELS, LATITUDES, WB2SequenceDataset, climatology_slots, inferred_times, load_normalization, official_splits
from .model import DIBFNO, FixedFNO, StaticGateFNO, effective_modes


class _FrozenMask(torch.nn.Module):
    """Input-independent inference ablation using a train-period mean gate."""

    def __init__(self, mask: np.ndarray):
        super().__init__()
        if mask.shape != (32, 33) or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError("frozen gate must be a finite 32x33 array in [0,1]")
        self.register_buffer("value", torch.from_numpy(mask.astype(np.float32)).view(1, 1, 32, 33))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand(x.shape[0], -1, -1, -1)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    canonical_path: str | Path,
    normalization_path: str | Path,
    climatology_path: str | Path,
    output_path: str | Path,
    *,
    split_name: str = "val",
    max_lead: int = 40,
    batch_size: int = 32,
    max_samples: int | None = None,
    fixed_mask_path: str | Path | None = None,
    gate_variant: str = "checkpoint_gate",
    gate_override: torch.nn.Module | None = None,
) -> dict:
    """Every forecast at lead k is compared with the actual field at t+k.

    ``gate_override`` replaces the spectral gate of a trained checkpoint at
    inference time only (hard top-C, frozen mean gate, thresholded gate).  It is
    recorded in the output so an ablation can never be mistaken for a retrained
    model.
    """
    if split_name not in ("val", "test"):
        raise ValueError("model evaluation is reserved for val or test")
    if max_lead < 1 or max_lead > 40:
        raise ValueError("max_lead must be in 1..40")
    if fixed_mask_path is not None and gate_override is not None:
        raise ValueError("use either fixed_mask_path or gate_override, not both")
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["channels"]) != CHANNELS:
        raise ValueError("checkpoint channels do not match canonical data")
    model_type = checkpoint.get("model_type", "dibfno")
    if model_type not in ("dibfno", "fixed_fno", "static_fno"):
        raise ValueError(f"unsupported checkpoint model_type: {model_type}")
    model_class = {"dibfno": DIBFNO, "fixed_fno": FixedFNO, "static_fno": StaticGateFNO}[model_type]
    model = model_class(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    source_parameter_count = sum(p.numel() for p in model.parameters())
    if fixed_mask_path is not None:
        if model_type != "dibfno":
            raise ValueError("frozen mean gate ablation requires DIB-FNO checkpoint")
        with np.load(fixed_mask_path) as gate_file:
            if "source_checkpoint_sha256" not in gate_file:
                raise ValueError("frozen gate lacks checkpoint provenance")
            import hashlib
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if str(gate_file["source_checkpoint_sha256"]) != checkpoint_hash:
                raise ValueError("frozen gate was calculated from a different checkpoint")
            model.mask = _FrozenMask(np.asarray(gate_file["mean_gate"]))
        gate_variant = "frozen_elementwise_mean_gate"
    elif gate_override is not None:
        if model_type == "fixed_fno":
            raise ValueError("a gate override is meaningless for a gate-free Fixed FNO")
        model.mask = gate_override
    model.eval()
    means_np, stds_np = load_normalization(normalization_path)
    if not np.allclose(checkpoint["normalization"]["mean"], means_np, rtol=1e-6):
        raise ValueError("checkpoint normalization differs from supplied training statistics")
    if not np.allclose(checkpoint["normalization"]["std"], stds_np, rtol=1e-6):
        raise ValueError("checkpoint normalization differs from supplied training statistics")
    means = torch.from_numpy(means_np).view(1, 8, 1, 1)
    stds = torch.from_numpy(stds_np).view(1, 8, 1, 1)
    split = official_splits()[split_name]
    dataset = WB2SequenceDataset(canonical_path, split, max_lead, normalization_path, max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    climate_group = zarr.open_group(Path(climatology_path), mode="r")
    climate = torch.from_numpy(np.asarray(climate_group["mean"][:], dtype=np.float32).reshape(8, 366 * 4, 32, 64))
    weights_np = np.cos(np.deg2rad(LATITUDES.astype(np.float64)))
    weights_np /= weights_np.mean()
    weights = torch.from_numpy(weights_np).view(1, 1, 32, 1)
    mse_sum = torch.zeros((max_lead, 8), dtype=torch.float64)
    bias_sum = torch.zeros((max_lead, 8), dtype=torch.float64)
    acc_sum = torch.zeros((max_lead, 8), dtype=torch.float64)
    sample_mse = np.empty((max_lead, len(dataset), 8), dtype=np.float32)
    sample_acc = np.empty((max_lead, len(dataset), 8), dtype=np.float32)
    mode_count_sum = 0.0
    n = 0
    first_case = {}
    time_axis = inferred_times()
    for x, targets, t0 in loader:
        current = x
        count = x.shape[0]
        for lead in range(1, max_lead + 1):
            current, mask = model(current, return_mask=True)
            if not torch.isfinite(current).all():
                raise ValueError(f"non-finite model forecast at lead {lead}")
            if lead == 1:
                mode_count_sum += float(effective_modes(mask).sum())
            truth = targets[:, lead - 1]
            pred_phys = current * stds + means
            truth_phys = truth * stds + means
            valid_indices = t0.numpy() + lead
            slots = climatology_slots(time_axis[valid_indices])
            cl = climate[:, slots].permute(1, 0, 2, 3)
            delta = (pred_phys - truth_phys).to(torch.float64)
            mse_sum[lead - 1] += (weights * delta.square()).sum(dim=(0, 2, 3))
            sample_mse[lead - 1, n : n + count] = (
                (weights * delta.square()).mean(dim=(2, 3)).numpy().astype(np.float32)
            )
            bias_sum[lead - 1] += (weights * delta).sum(dim=(0, 2, 3))
            pa = (pred_phys - cl).to(torch.float64)
            ta = (truth_phys - cl).to(torch.float64)
            numerator = (weights * pa * ta).sum(dim=(2, 3))
            denominator = torch.sqrt(
                (weights * pa.square()).sum(dim=(2, 3))
                * (weights * ta.square()).sum(dim=(2, 3))
            )
            if torch.any(denominator <= 0):
                raise ValueError("ACC denominator zero during model evaluation")
            acc_sum[lead - 1] += (numerator / denominator).sum(dim=0)
            sample_acc[lead - 1, n : n + count] = (numerator / denominator).numpy().astype(np.float32)
            if n == 0 and lead in (1, 4, 12, 20, 40):
                first_case[f"pred_{lead}"] = pred_phys[0].numpy()
                first_case[f"truth_{lead}"] = truth_phys[0].numpy()
        n += count
    if n != len(dataset):
        raise AssertionError("evaluation sample count mismatch")
    rmse = torch.sqrt(mse_sum / (n * 32 * 64)).numpy()
    bias = (bias_sum / (n * 32 * 64)).numpy()
    acc = (acc_sum / n).numpy()
    records = []
    for lead in range(1, max_lead + 1):
        records.append({
            "lead_hours": lead * 6,
            "channels": {
                channel: {"rmse": float(rmse[lead - 1, i]), "bias": float(bias[lead - 1, i]), "acc": float(acc[lead - 1, i])}
                for i, channel in enumerate(CHANNELS)
            },
        })
    result = {
        "status": "measured_autoregressive_forecast",
        "split": split_name,
        "initializations": n,
        "start_time": str(time_axis[split.start]),
        "last_init_time": str(time_axis[split.start + n - 1]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "seed": int(checkpoint["seed"]),
        "parameter_count": source_parameter_count,
        "model_type": model_type,
        "gate_variant": gate_variant,
        "gate_override_class": type(gate_override).__name__ if gate_override is not None else None,
        "gate_override_top_c": getattr(gate_override, "top_c", None),
        "gate_override_threshold": getattr(gate_override, "threshold", None),
        "mask_override": str(Path(fixed_mask_path).resolve()) if fixed_mask_path is not None else None,
        "mean_effective_modes_at_initial_step": mode_count_sum / n,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path = output_path.with_name(output_path.stem + "_sample_scores.npz")
    np.savez_compressed(
        scores_path,
        model_mse=sample_mse,
        model_acc=sample_acc,
        init_indices=np.arange(split.start, split.start + n, dtype=np.int32),
    )
    result["sample_scores_path"] = str(scores_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if first_case:
        np.savez_compressed(
            output_path.with_name(output_path.stem + "_first_case.npz"),
            init_time=np.asarray(str(time_axis[split.start])),
            **first_case,
        )
    return {"output": str(output_path), "split": split_name, "initializations": n, "max_lead_hours": max_lead * 6}
