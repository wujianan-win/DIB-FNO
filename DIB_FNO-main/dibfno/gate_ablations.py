"""Train-only construction of a frozen mean gate for causal-style inference ablation.

This ablation holds the pretrained DIB-FNO backbone fixed. It estimates the
contribution of input variation at inference, not the effect of retraining a
new static model; the separately trained StaticGateFNO addresses the latter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import CHANNELS, WB2SequenceDataset, inferred_times, load_normalization, official_splits
from .model import DIBFNO


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_train_mean_gate(
    checkpoint_path: str | Path,
    canonical_path: str | Path,
    normalization_path: str | Path,
    output_path: str | Path,
) -> dict:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type", "dibfno") != "dibfno" or tuple(checkpoint["channels"]) != CHANNELS:
        raise ValueError("only a matching DIB-FNO checkpoint may define the mean gate")
    mean, std = load_normalization(normalization_path)
    if not np.allclose(checkpoint["normalization"]["mean"], mean) or not np.allclose(checkpoint["normalization"]["std"], std):
        raise ValueError("checkpoint normalization mismatch")
    model = DIBFNO(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    split = official_splits()["train"]
    dataset = WB2SequenceDataset(canonical_path, split, 1, normalization_path)
    dates = np.asarray([f"{year:04d}-{month:02d}-15T00" for year in range(1959, 2016)
                        for month in range(1, 13)], dtype="datetime64[h]")
    times = inferred_times()
    positions = np.searchsorted(times, dates)
    if not np.array_equal(times[positions], dates):
        raise ValueError("monthly train dates not found on inferred time axis")
    masks = []
    with torch.no_grad():
        for position in positions:
            x, _, t0 = dataset[int(position) - split.start]
            if int(t0) != int(position):
                raise AssertionError("train dataset index alignment failed")
            _, gate = model(x.unsqueeze(0), return_mask=True)
            masks.append(gate[0, 0].numpy())
    values = np.stack(masks)
    mean_gate = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean_gate=mean_gate,
        sampled_train_dates=dates.astype(str),
        source_checkpoint_sha256=np.asarray(_sha256(checkpoint_path)),
        note=np.asarray("mean of one 00 UTC analysis on the 15th of each train month, 1959–2015"),
    )
    metadata = {
        "status": "train_period_only_frozen_mean_gate",
        "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": _sha256(checkpoint_path),
        "sample_count": len(values), "sampling": "15th day 00 UTC of every training month, 1959–2015",
        "mean_gate_weight_sum": float(mean_gate.sum()),
        "mean_across_date_per_frequency_sd": float(values.std(axis=0, ddof=1).mean()),
        "output": str(output_path), "output_sha256": _sha256(output_path),
        "interpretation": "Inference-only static replacement with unchanged trained backbone; not equivalent to retraining a learned static gate.",
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output_path), "metadata": str(json_path), "samples": len(values)}
