"""Train-only mean-gate inference ablation for three saved DIB-FNO seeds.

This keeps each trained backbone fixed. The mean gate is calculated from the
same 384 evenly spaced training initializations for every seed. It is never
estimated from validation or test fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .completion_evaluation import evaluate_with_gate
from .data import WB2SequenceDataset, official_splits
from .gate_topc import load_frozen_backbone
from .model_evaluation import _FrozenMask


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_mean_gate(checkpoint: Path, canonical: Path, normalization: Path,
                    output: Path, *, samples: int = 384) -> dict:
    model, metadata = load_frozen_backbone(checkpoint)
    if metadata.get("model_type") != "dibfno":
        raise ValueError("DIB-FNO checkpoint required")
    split = official_splits()["train"]
    dataset = WB2SequenceDataset(canonical, split, 1, normalization)
    positions = np.linspace(0, len(dataset) - 1, samples).astype(np.int64)
    gates = np.empty((samples, 32, 33), dtype=np.float32)
    with torch.no_grad():
        for row, position in enumerate(positions):
            x, _, t0 = dataset[int(position)]
            if int(t0) != split.start + int(position):
                raise AssertionError("training index mismatch")
            gates[row] = model.mask(x.unsqueeze(0))[0, 0].numpy()
    mean = gates.mean(axis=0, dtype=np.float64).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, mean_gate=mean, train_positions=positions,
                        source_checkpoint_sha256=np.asarray(sha256(checkpoint)))
    result = {"seed": int(metadata["seed"]), "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256(checkpoint), "gate_file": str(output),
              "gate_file_sha256": sha256(output), "train_samples": samples,
              "trained_parameter_count": sum(p.numel() for p in model.parameters()),
              "trained_gate_parameter_count": sum(p.numel() for p in model.mask.parameters()),
              "sampling": "384 evenly spaced train-split initializations, 1959–2015",
              "mean_gate_weight_sum": float(mean.sum()),
              "mean_per_frequency_across_sample_sd": float(gates.std(axis=0, ddof=1).mean()),
              "gate_grid_elements": 1056,
              "gate_temperature": float(model.mask.temperature)}
    return result


def build(source: Path, output: Path, *, threads: int = 12,
          seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    torch.set_num_threads(threads)
    canonical = ROOT / "data/canonical/wb2_8ch_6h.zarr"
    normalization = ROOT / "data/canonical/normalization.json"
    climatology = ROOT / "data/canonical/climatology_v2.zarr"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        checkpoint = source / "checkpoints" / f"dibfno_h16_seed{seed}" / "best.pt"
        gate_path = output / "mean_gates" / f"seed{seed}_train384_mean_gate.npz"
        record = train_mean_gate(checkpoint, canonical, normalization, gate_path)
        if seed == 0:
            with np.load(source / "analyses/gate_families.npz") as previous:
                reference = np.asarray(previous["frequency_mean_gate"])
            with np.load(gate_path) as current:
                np.testing.assert_array_equal(current["mean_gate"], reference)
            record["matches_prior_seed0_train_gate"] = True
            prior_stem = source / "evaluations/dibfno_h16_seed0_test_40steps__frequency_mean_gate"
            stem = output / "evaluations" / "dibfno_h16_seed0_test_40steps__frozen_mean_gate"
            stem.parent.mkdir(parents=True, exist_ok=True)
            for suffix in (".json", "_sample_scores.npz"):
                shutil.copy2(Path(str(prior_stem) + suffix), Path(str(stem) + suffix))
            record["test_scores_reused_after_exact_gate_match"] = True
        else:
            with np.load(gate_path) as gate_file:
                gate = _FrozenMask(np.asarray(gate_file["mean_gate"]))
            stem = output / "evaluations" / f"dibfno_h16_seed{seed}_test_40steps__frozen_mean_gate.json"
            evaluate_with_gate(checkpoint, canonical, normalization, climatology, stem,
                               split_name="test", max_lead=40, batch_size=32,
                               gate=gate, gate_variant="frozen_train384_mean_gate")
        summary_path = stem if seed else Path(str(stem) + ".json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["parameter_count"] = record["trained_parameter_count"]
        summary["active_inference_parameter_count"] = (
            record["trained_parameter_count"] - record["trained_gate_parameter_count"])
        summary["parameter_count_note"] = (
            "trained checkpoint count includes the dynamic gate; frozen inference "
            "replaces it with a parameter-free mean gate"
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    manifest = {"method": "frozen train-mean gate on unchanged trained DIB-FNO backbone",
                "threads": threads, "seeds": records}
    (output / "mean_gates" / "metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/training")
    parser.add_argument("--output", default="outputs/frozen_gate")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    print(json.dumps(build(Path(args.source).resolve(), Path(args.output).resolve(),
                           threads=args.threads, seeds=tuple(args.seeds)), ensure_ascii=False))
