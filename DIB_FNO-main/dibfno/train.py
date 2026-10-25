"""Reproducible training and rollout evaluation on canonical WB2 data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import time
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CHANNELS, LATITUDES, ContiguousBatchSampler, WB2SequenceDataset, load_normalization, official_splits
from .metrics import denormalize, latitude_weights, persistence_rmse, weighted_rmse
from .model import DIBFNO, FixedFNO, StaticGateFNO, effective_modes


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def rollout(model: torch.nn.Module, x: torch.Tensor, steps: int) -> torch.Tensor:
    predictions = []
    current = x
    for _ in range(steps):
        current = model(current)
        predictions.append(current)
    return torch.stack(predictions, dim=1)


def train_model(
    canonical_path: str | Path,
    normalization_path: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    rollout_steps: int = 1,
    learning_rate: float = 3e-4,
    beta: float = 1e-4,
    seed: int = 0,
    hidden_dim: int = 64,
    n_blocks: int = 4,
    mask_hidden_dim: int = 32,
    model_type: str = "dibfno",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> dict:
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
    if model_type not in ("dibfno", "fixed_fno", "static_fno"):
        raise ValueError(f"unsupported model_type: {model_type}")
    model_config = {
        "in_channels": 8, "hidden_dim": hidden_dim, "n_blocks": n_blocks,
        "modes_lat": 8, "modes_lon": 17,
    }
    if model_type == "dibfno":
        model_config["mask_hidden_dim"] = mask_hidden_dim
    model_class = {"dibfno": DIBFNO, "fixed_fno": FixedFNO, "static_fno": StaticGateFNO}[model_type]
    model = model_class(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    means_np, stds_np = load_normalization(normalization_path)
    means = torch.from_numpy(means_np).to(device)
    stds = torch.from_numpy(stds_np).to(device)
    lat_w = latitude_weights(torch.from_numpy(LATITUDES)).to(device)
    history = []
    best_score = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        temperature = max(0.2, 1.0 * (0.2 ** (epoch / max(epochs, 1))))
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
            for step in range(rollout_steps):
                current, mask = model(current, return_mask=True)
                predictions.append(current)
                masks.append(mask)
            pred_seq = torch.stack(predictions, dim=1)
            area_weights = lat_w.view(1, 1, 1, 32, 1)
            pred_loss = torch.mean((pred_seq - targets).square() * area_weights)
            neff_fraction = torch.stack([effective_modes(m).mean() / m.shape[-2] / m.shape[-1] for m in masks]).mean()
            warmup = min(1.0, epoch / max(1, epochs // 5))
            loss = pred_loss + (beta * warmup * neff_fraction if model_type in ("dibfno", "static_fno") else 0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * x.shape[0]
            n_train += x.shape[0]

        model.eval()
        rmse_sum = torch.zeros(8, device=device)
        n_val = 0
        mask_variability = []
        with torch.no_grad():
            for x, targets, _ in val_loader:
                x, targets = x.to(device), targets.to(device)
                pred = rollout(model, x, rollout_steps)[:, -1]
                truth = targets[:, -1]
                pred_phys = denormalize(pred, means, stds)
                truth_phys = denormalize(truth, means, stds)
                batch_rmse = weighted_rmse(pred_phys, truth_phys, lat_w)
                rmse_sum += batch_rmse.square() * x.shape[0]
                _, mask = model(x, return_mask=True)
                centered = mask - mask.mean(dim=0, keepdim=True)
                mask_variability.append(float(centered.square().mean().sqrt()))
                n_val += x.shape[0]
        rmse = (rmse_sum / max(n_val, 1)).sqrt()
        normalized_score = float((rmse / stds).mean())
        record = {
            "epoch": epoch,
            "train_loss": train_total / max(n_train, 1),
            "normalized_mean_rmse": normalized_score,
            "mask_between_sample_std": float(np.mean(mask_variability)),
            "temperature": temperature,
            "seconds": time.perf_counter() - started,
            **{f"rmse_{name}": float(rmse[i]) for i, name in enumerate(CHANNELS)},
        }
        history.append(record)
        if normalized_score < best_score:
            best_score = normalized_score
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

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best_record = min(history, key=lambda item: item["normalized_mean_rmse"])
    persistence = persistence_rmse(
        canonical_path,
        splits["val"],
        lead_steps=rollout_steps,
        max_samples=max_val_samples,
    )
    skill_vs_persistence = {
        name: 1.0 - (best_record[f"rmse_{name}"] ** 2) / (persistence[name] ** 2)
        for name in CHANNELS
    }
    summary = {
        "data": str(Path(canonical_path).resolve()),
        "normalization": str(Path(normalization_path).resolve()),
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "model_config": model_config,
        "model_type": model_type,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "seed": seed,
        "rollout_steps": rollout_steps,
        "best_normalized_mean_rmse": best_score,
        "best_epoch": int(np.argmin([r["normalized_mean_rmse"] for r in history]) + 1),
        "persistence_on_same_validation_samples": persistence,
        "mse_skill_vs_persistence": skill_vs_persistence,
        "scientific_status": (
            "subset_smoke_test_only" if max_train_samples is not None
            else "full_train_single_seed_validation_only"
        ),
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
