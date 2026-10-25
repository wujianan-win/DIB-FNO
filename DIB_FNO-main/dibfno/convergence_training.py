"""Budgeted training to validation convergence for the 2026-09-23 review round.

Motivation (review request 2): the first experiment round trained six models for
only five epochs, and every model selected its best checkpoint at the final
epoch while the validation metric was still decreasing.  That makes the
fixed-spectral-control comparison a *fixed-compute early-training* comparison
rather than a comparison of trained operators.

This module therefore:

* trains under an explicit epoch ceiling but stops early when the validation
  selection metric has not improved for ``patience`` epochs;
* always keeps the checkpoint with the best validation selection metric;
* records the complete training/validation curve, wall-clock time per epoch and
  the reason training stopped, so convergence can be judged from the artifact
  instead of asserted;
* keeps the sigmoid temperature schedule of the original short runs anchored at
  a fixed epoch budget, so extending the ceiling does not silently change the
  gate sharpening schedule.

It does not modify the existing scientific modules or results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CHANNELS, LATITUDES, ContiguousBatchSampler, WB2SequenceDataset, load_normalization, official_splits
from .metrics import denormalize, latitude_weights, persistence_rmse, weighted_rmse
from .model import DIBFNO, FixedFNO, StaticGateFNO, effective_modes
from .train import seed_everything

MODEL_CLASSES = {"dibfno": DIBFNO, "fixed_fno": FixedFNO, "static_fno": StaticGateFNO}
GATED_TYPES = ("dibfno", "static_fno")


def temperature_at(epoch: int, temperature_epochs: int) -> float:
    """Sigmoid-gate temperature of the original runs, held after the anchor epoch."""
    anchor = max(1, temperature_epochs)
    return max(0.2, 1.0 * (0.2 ** (min(epoch, anchor) / anchor)))


def train_to_convergence(
    canonical_path: str | Path,
    normalization_path: str | Path,
    output_dir: str | Path,
    *,
    model_type: str = "dibfno",
    hidden_dim: int = 16,
    n_blocks: int = 1,
    mask_hidden_dim: int = 8,
    max_epochs: int = 50,
    patience: int = 8,
    min_epochs: int = 5,
    temperature_epochs: int = 5,
    batch_size: int = 32,
    rollout_steps: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    beta: float = 1e-4,
    seed: int = 0,
    threads: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> dict:
    """Train one model/seed pair and persist the full convergence record."""
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"unsupported model_type: {model_type}")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be >= 1")
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be >= 1")
        torch.set_num_threads(int(threads))

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = official_splits()
    train_ds = WB2SequenceDataset(canonical_path, splits["train"], rollout_steps, normalization_path, max_train_samples)
    val_ds = WB2SequenceDataset(canonical_path, splits["val"], rollout_steps, normalization_path, max_val_samples)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=ContiguousBatchSampler(len(train_ds), batch_size, seed),
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model_config = {
        "in_channels": 8, "hidden_dim": hidden_dim, "n_blocks": n_blocks,
        "modes_lat": 8, "modes_lon": 17,
    }
    if model_type == "dibfno":
        model_config["mask_hidden_dim"] = mask_hidden_dim
    model = MODEL_CLASSES[model_type](**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    means_np, stds_np = load_normalization(normalization_path)
    means = torch.from_numpy(means_np).to(device)
    stds = torch.from_numpy(stds_np).to(device)
    lat_w = latitude_weights(torch.from_numpy(LATITUDES)).to(device)
    stds_cpu = torch.from_numpy(stds_np)

    history: list[dict] = []
    best_score = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0
    stop_reason = "reached_max_epochs"
    train_seconds_total = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        temperature = temperature_at(epoch, temperature_epochs)
        model.set_temperature(temperature)
        train_total = 0.0
        n_train = 0
        started = time.perf_counter()
        for x, targets, _ in train_loader:
            x, targets = x.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = []
            masks = []
            current = x
            for _step in range(rollout_steps):
                current, mask = model(current, return_mask=True)
                predictions.append(current)
                masks.append(mask)
            pred_seq = torch.stack(predictions, dim=1)
            area_weights = lat_w.view(1, 1, 1, 32, 1)
            pred_loss = torch.mean((pred_seq - targets).square() * area_weights)
            neff_fraction = torch.stack(
                [effective_modes(m).mean() / m.shape[-2] / m.shape[-1] for m in masks]
            ).mean()
            warmup = min(1.0, epoch / max(1, temperature_epochs))
            loss = pred_loss + (
                beta * warmup * neff_fraction if model_type in GATED_TYPES else 0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * x.shape[0]
            n_train += x.shape[0]
        train_seconds = time.perf_counter() - started
        train_seconds_total += train_seconds

        model.eval()
        rmse_sum = torch.zeros(8, device=device)
        n_val = 0
        mask_variability = []
        gate_sum = 0.0
        with torch.no_grad():
            for x, targets, _ in val_loader:
                x, targets = x.to(device), targets.to(device)
                pred = model(x)
                truth = targets[:, -1]
                pred_phys = denormalize(pred, means, stds)
                truth_phys = denormalize(truth, means, stds)
                batch_rmse = weighted_rmse(pred_phys, truth_phys, lat_w)
                rmse_sum += batch_rmse.square() * x.shape[0]
                _, mask = model(x, return_mask=True)
                centered = mask - mask.mean(dim=0, keepdim=True)
                mask_variability.append(float(centered.square().mean().sqrt()))
                gate_sum += float(effective_modes(mask).sum())
                n_val += x.shape[0]
        rmse = (rmse_sum / max(n_val, 1)).sqrt()
        normalized_score = float((rmse / stds_cpu).mean())
        record = {
            "epoch": epoch,
            "train_loss": train_total / max(n_train, 1),
            "val_normalized_mean_rmse": normalized_score,
            "val_mean_rmse_over_channels": float(rmse.mean()),
            "mask_between_sample_std": float(np.mean(mask_variability)),
            "mean_effective_modes": gate_sum / max(n_val, 1),
            "temperature": temperature,
            "train_seconds": train_seconds,
            **{f"val_rmse_{name}": float(rmse[i]) for i, name in enumerate(CHANNELS)},
        }
        history.append(record)

        if normalized_score < best_score - 1e-9:
            best_score = normalized_score
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "model_config": model_config,
                    "model_type": model_type,
                    "channels": CHANNELS,
                    "normalization": {"mean": means_np.tolist(), "std": stds_np.tolist()},
                },
                output_dir / "best.pt",
            )
        else:
            epochs_since_improvement += 1

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(record["train_loss"], 6),
                    "val_normalized_mean_rmse": round(normalized_score, 6),
                    "best_epoch": best_epoch,
                    "patience_left": patience - epochs_since_improvement,
                    "seconds": round(train_seconds, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if epoch >= min_epochs and epochs_since_improvement >= patience:
            stop_reason = f"validation_plateau_{patience}_epochs"
            break

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    best_record = min(history, key=lambda item: item["val_normalized_mean_rmse"])
    final_record = history[-1]
    persistence = persistence_rmse(
        canonical_path, splits["val"], lead_steps=rollout_steps, max_samples=max_val_samples
    )
    skill_vs_persistence = {
        name: 1.0 - (best_record[f"val_rmse_{name}"] ** 2) / (persistence[name] ** 2)
        for name in CHANNELS
    }
    window = history[-patience:]
    summary = {
        "status": "budgeted_training_to_validation_convergence",
        "data": str(Path(canonical_path).resolve()),
        "normalization": str(Path(normalization_path).resolve()),
        "device": str(device),
        "threads": int(torch.get_num_threads()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "model_config": model_config,
        "model_type": model_type,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "seed": seed,
        "rollout_steps": rollout_steps,
        "optimizer": {"name": "AdamW", "lr": learning_rate, "weight_decay": weight_decay},
        "sparsity_beta": beta if model_type in GATED_TYPES else 0.0,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_epochs": min_epochs,
        "temperature_schedule": (
            f"gate_temperature(epoch) = max(0.2, 0.2 ** (min(epoch, {temperature_epochs}) / "
            f"{temperature_epochs})), i.e. the original 5-epoch schedule held at its floor"
        ),
        "temperature_epochs": temperature_epochs,
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_val_normalized_mean_rmse": best_score,
        "final_epoch": final_record["epoch"],
        "final_val_normalized_mean_rmse": final_record["val_normalized_mean_rmse"],
        "last_window_improvement": float(
            window[0]["val_normalized_mean_rmse"] - window[-1]["val_normalized_mean_rmse"]
        ),
        "last_window_relative_improvement": float(
            (window[0]["val_normalized_mean_rmse"] - window[-1]["val_normalized_mean_rmse"])
            / window[0]["val_normalized_mean_rmse"]
        ),
        "train_seconds_total": train_seconds_total,
        "train_seconds_per_epoch": train_seconds_total / len(history),
        "checkpoint_selection": "lowest validation normalized mean per-channel RMSE",
        "persistence_on_same_validation_samples": persistence,
        "mse_skill_vs_persistence": skill_vs_persistence,
        "train_samples_per_epoch": len(train_ds),
        "val_samples_per_epoch": len(val_ds),
        "scientific_status": (
            "subset_smoke_test_only" if max_train_samples is not None
            else "full_train_single_seed_validation_only"
        ),
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
