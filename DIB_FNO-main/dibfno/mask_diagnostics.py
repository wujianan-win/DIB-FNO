"""Measured spectral-gate diagnostics on real validation analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CHANNELS, WB2SequenceDataset, inferred_times, load_normalization, official_splits
from .model import DIBFNO


def diagnose_mask(
    checkpoint_path: str | Path,
    canonical_path: str | Path,
    normalization_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Evaluate 36 monthly 00 UTC dates across the full 2016–2018 validation block."""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type", "dibfno") != "dibfno" or tuple(checkpoint["channels"]) != CHANNELS:
        raise ValueError("mask diagnostics require a DIB-FNO checkpoint with matching channels")
    mean, std = load_normalization(normalization_path)
    if not np.allclose(checkpoint["normalization"]["mean"], mean) or not np.allclose(checkpoint["normalization"]["std"], std):
        raise ValueError("checkpoint normalization mismatch")
    model = DIBFNO(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    split = official_splits()["val"]
    dataset = WB2SequenceDataset(canonical_path, split, 1, normalization_path)
    time_axis = inferred_times()
    requested = np.asarray([
        f"{year:04d}-{month:02d}-15T00" for year in (2016, 2017, 2018) for month in range(1, 13)
    ], dtype="datetime64[h]")
    positions = np.searchsorted(time_axis, requested)
    if not np.array_equal(time_axis[positions], requested):
        raise ValueError("monthly validation dates absent from inferred time axis")
    with torch.no_grad():
        masks = []
        for position in positions:
            x, _, t0 = dataset[int(position) - split.start]
            if int(t0) != int(position):
                raise AssertionError("dataset/index alignment failed")
            _, mask = model(x.unsqueeze(0), return_mask=True)
            masks.append(mask[0, 0].numpy())
    values = np.stack(masks)
    mean_mask = values.mean(axis=0)
    std_mask = values.std(axis=0, ddof=1)
    sums = values.sum(axis=(1, 2))
    signed_lat_frequency = np.abs(np.fft.fftfreq(32) * 32)
    low = (signed_lat_frequency[:, None] <= 4) & (np.arange(33)[None, :] <= 4)
    low[0, 0] = False
    high = np.broadcast_to(np.arange(33)[None, :] >= 24, (32, 33))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "mask_real_validation_monthly_samples.npz"
    np.savez_compressed(npz_path, mask=values, dates=requested.astype(str))
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.5), constrained_layout=True)
    pictures = (mean_mask, std_mask, values[0], values[6])
    titles = ("Mean soft gate", "Across-date gate SD", "2016-01-15 gate", "2016-07-15 gate")
    for ax, field, title in zip(axes.flat, pictures, titles):
        shifted = np.fft.fftshift(field, axes=0)
        image = ax.imshow(shifted, origin="lower", aspect="auto", extent=(-0.5, 32.5, -16.5, 15.5),
                          cmap="viridis", vmin=0, vmax=1 if title != "Across-date gate SD" else None)
        ax.set(title=title, xlabel="Zonal Fourier index", ylabel="Meridional Fourier index")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("DIB-FNO soft spectral gate on 36 real monthly validation analyses", weight="bold")
    fig_path = output_dir / "fig16_real_data_spectral_mask.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    def sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    result = {
        "status": "measured_real_validation_mask_diagnostic",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha(checkpoint_path),
        "sample_count": len(values), "sampling": "15th day 00 UTC of each month, 2016–2018",
        "temperature": float(model.mask.temperature),
        "mean_gate_weight_sum": float(sums.mean()), "sd_gate_weight_sum": float(sums.std(ddof=1)),
        "mean_per_frequency_across_date_sd": float(std_mask.mean()),
        "max_per_frequency_across_date_sd": float(std_mask.max()),
        "mean_low_frequency_gate_excluding_dc": float(values[:, low].mean()),
        "mean_high_zonal_frequency_gate": float(values[:, high].mean()),
        "mean_dc_gate": float(values[:, 0, 0].mean()),
        "caveat": "The learned gate favors high zonal frequencies in this sample; this is descriptive, not proof that the gate causes long-rollout degradation. Gate-weight sums are not independent Fourier degrees.",
        "arrays": str(npz_path), "arrays_sha256": sha(npz_path),
        "figure": str(fig_path), "figure_sha256": sha(fig_path),
        "figure_svg": str(fig_path.with_suffix(".svg")), "figure_svg_sha256": sha(fig_path.with_suffix(".svg")),
    }
    json_path = output_dir / "mask_diagnostic.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(json_path), "figure": str(fig_path), "samples": len(values)}
