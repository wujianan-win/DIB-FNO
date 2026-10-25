"""Portable inference-only frozen-mean-gate DIB-FNO model artifacts.

These are not newly trained checkpoints. They package a trained dynamic
backbone and its train-period mean gate in one file for reproducible inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import DIBFNO
from .model_evaluation import _FrozenMask


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source_checkpoint: Path, train_mean_gate: Path, output: Path) -> dict:
    source_checkpoint = source_checkpoint.resolve()
    train_mean_gate = train_mean_gate.resolve()
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    if source.get("model_type") != "dibfno":
        raise ValueError("source must be a DIB-FNO checkpoint")
    with np.load(train_mean_gate) as data:
        if str(data["source_checkpoint_sha256"]) != sha256(source_checkpoint):
            raise ValueError("mean gate does not match checkpoint")
        mean_gate = np.asarray(data["mean_gate"], dtype=np.float32)
        positions = np.asarray(data["train_positions"], dtype=np.int64)
    if mean_gate.shape != (32, 33) or positions.shape != (384,):
        raise ValueError("unexpected gate or training-sample shape")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_type": "frozen_mean_gate_fno_inference_variant",
        "model_config": source["model_config"],
        "dynamic_checkpoint_state_dict": source["model"],
        "mean_gate": torch.from_numpy(mean_gate),
        "seed": int(source["seed"]),
        "selected_epoch": int(source["epoch"]),
        "train_sample_positions": torch.from_numpy(positions),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "train_mean_gate_sha256": sha256(train_mean_gate),
        "interpretation": "inference-only static replacement; not a retrained model",
    }
    torch.save(payload, output)
    return {"output": str(output), "output_sha256": sha256(output),
            "seed": payload["seed"], "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
            "gate_weight_sum": float(mean_gate.sum())}


def load(path: str | Path) -> tuple[DIBFNO, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("model_type") != "frozen_mean_gate_fno_inference_variant":
        raise ValueError("not a supported frozen-mean-gate artifact")
    model = DIBFNO(**payload["model_config"])
    model.load_state_dict(payload["dynamic_checkpoint_state_dict"])
    model.mask = _FrozenMask(payload["mean_gate"].numpy())
    model.eval()
    return model, payload


def export_all(source: Path, package: Path) -> dict:
    records = []
    for seed in range(3):
        records.append(export(
            source / "checkpoints" / f"dibfno_h16_seed{seed}" / "best.pt",
            package / "mean_gates" / f"seed{seed}_train384_mean_gate.npz",
            package / "frozen_models" / f"frozen_mean_gate_seed{seed}.pt",
        ))
    metadata = {"model_family": "DIB-FNO frozen train-mean gate inference variants",
                "trained_new_weights": False, "seeds": records}
    (package / "frozen_models" / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/training")
    parser.add_argument("--package", default="outputs/frozen_gate")
    args = parser.parse_args()
    print(json.dumps(export_all(Path(args.source).resolve(), Path(args.package).resolve()), ensure_ascii=False))
