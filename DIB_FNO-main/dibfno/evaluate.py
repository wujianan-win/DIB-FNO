"""Leakage-free deterministic validation of physical baseline forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from .data import CHANNELS, LATITUDES, Split, climatology_slots, inferred_times


def evaluate_physical_baselines(
    canonical_path: str | Path,
    climatology_path: str | Path,
    split: Split,
    *,
    max_lead: int = 40,
    max_samples: int | None = None,
    batch_size: int = 128,
    output_path: str | Path | None = None,
) -> dict:
    """Score persistence and calendar climatology on identical initialization dates.

    The sample set is fixed for all leads (last ``max_lead`` starts excluded).
    RMSE is sqrt of the latitude-weighted mean squared error over all samples
    and grid points. ACC is the mean per-sample spatial anomaly correlation.
    Climatology's own ACC is undefined because its forecast anomaly is zero.
    """
    if max_lead < 1 or max_lead > 40:
        raise ValueError("max_lead must be in 1..40")
    canonical = zarr.open_group(Path(canonical_path), mode="r")["state"]
    climate = zarr.open_group(Path(climatology_path), mode="r")["mean"]
    if tuple(canonical.shape) != (93_544, 8, 32, 64):
        raise ValueError("canonical shape mismatch")
    if tuple(climate.shape) != (8, 366, 4, 32, 64):
        raise ValueError("climatology shape mismatch")
    samples = split.sample_count(max_lead)
    if max_samples is not None:
        samples = min(samples, max_samples)
    if samples <= 0:
        raise ValueError("split has no usable initialization dates")
    # Validation spans only a few years; this read is ~300 MiB and avoids 40x I/O.
    fields = np.asarray(canonical[split.start : split.start + samples + max_lead], dtype=np.float32)
    climate_fields = np.asarray(climate[:], dtype=np.float32).reshape(8, 366 * 4, 32, 64)
    slots = climatology_slots(inferred_times()[split.start : split.start + samples + max_lead])
    weights = np.cos(np.deg2rad(LATITUDES.astype(np.float64)))
    weights /= weights.mean()
    weights = weights.reshape(1, 1, 32, 1)
    sample_mse_p = np.empty((max_lead, samples, 8), dtype=np.float32) if output_path is not None else None
    sample_mse_c = np.empty((max_lead, samples, 8), dtype=np.float32) if output_path is not None else None
    sample_acc_p = np.empty((max_lead, samples, 8), dtype=np.float32) if output_path is not None else None
    records = []
    for lead in range(1, max_lead + 1):
        squared = {"persistence": np.zeros(8), "climatology": np.zeros(8)}
        biases = {"persistence": np.zeros(8), "climatology": np.zeros(8)}
        acc_sum = np.zeros(8)
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            init = fields[start:stop].astype(np.float64)
            truth = fields[start + lead : stop + lead].astype(np.float64)
            valid_slots = slots[start + lead : stop + lead]
            cl = np.moveaxis(climate_fields[:, valid_slots], 0, 1).astype(np.float64)
            for name, forecast in (("persistence", init), ("climatology", cl)):
                delta = forecast - truth
                squared[name] += (weights * np.square(delta)).sum(axis=(0, 2, 3))
                biases[name] += (weights * delta).sum(axis=(0, 2, 3))
                if sample_mse_p is not None:
                    scores = (weights * np.square(delta)).mean(axis=(2, 3)).astype(np.float32)
                    if name == "persistence":
                        sample_mse_p[lead - 1, start:stop] = scores
                    else:
                        sample_mse_c[lead - 1, start:stop] = scores
            p_anom, t_anom = init - cl, truth - cl
            numerator = (weights * p_anom * t_anom).sum(axis=(2, 3))
            denominator = np.sqrt(
                (weights * np.square(p_anom)).sum(axis=(2, 3))
                * (weights * np.square(t_anom)).sum(axis=(2, 3))
            )
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise ValueError("ACC denominator is zero or non-finite")
            acc_sum += (numerator / denominator).sum(axis=0)
            if sample_acc_p is not None:
                sample_acc_p[lead - 1, start:stop] = (numerator / denominator).astype(np.float32)
        divisor = samples * 32 * 64
        p_rmse = np.sqrt(squared["persistence"] / divisor)
        c_rmse = np.sqrt(squared["climatology"] / divisor)
        records.append({
            "lead_steps": lead,
            "lead_hours": lead * 6,
            "channels": {
                channel: {
                    "persistence_rmse": float(p_rmse[i]),
                    "climatology_rmse": float(c_rmse[i]),
                    "persistence_acc": float(acc_sum[i] / samples),
                    "climatology_acc": None,
                    "persistence_bias": float(biases["persistence"][i] / divisor),
                    "climatology_bias": float(biases["climatology"][i] / divisor),
                    "persistence_mse_skill_vs_climatology": float(
                        1 - squared["persistence"][i] / squared["climatology"][i]
                    ),
                }
                for i, channel in enumerate(CHANNELS)
            },
        })
    result = {
        "split": split.name,
        "initializations": samples,
        "start_time": str(inferred_times()[split.start]),
        "last_init_time": str(inferred_times()[split.start + samples - 1]),
        "max_lead_hours": 6 * max_lead,
        "canonical_path": str(Path(canonical_path).resolve()),
        "climatology_path": str(Path(climatology_path).resolve()),
        "time_axis_status": "inferred_from_official_WB2_record",
        "protocol": {
            "grid": "32 latitude x 64 longitude; 5.625 degree; no poles",
            "weights": "cos(latitude)/mean(cos(latitude))",
            "rmse": "sqrt(mean_time_space(weight * squared_error))",
            "acc": "mean over initializations of spatial weighted anomaly correlation",
            "climatology": "train-only calendar month-day x UTC hour, no smoothing",
            "climatology_acc": "undefined (zero forecast anomaly)",
        },
        "records": records,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sample_path = output.with_name(output.stem + "_sample_scores.npz")
        np.savez_compressed(
            sample_path,
            persistence_mse=sample_mse_p,
            climatology_mse=sample_mse_c,
            persistence_acc=sample_acc_p,
            init_indices=np.arange(split.start, split.start + samples, dtype=np.int32),
        )
        result["sample_scores_path"] = str(sample_path.resolve())
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
