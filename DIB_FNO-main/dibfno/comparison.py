"""Paired, dependence-aware comparisons of measured validation forecasts.

The intervals here describe variation across initialization dates conditional on
the trained checkpoint.  They are not an estimate of random-seed variability.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from .data import CHANNELS, UNITS, official_splits


def _load_scores(path: str | Path, key: str, *, nonnegative: bool = True) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        scores = np.asarray(data[key], dtype=np.float64)
        indices = np.asarray(data["init_indices"], dtype=np.int32)
    if scores.ndim != 3 or scores.shape[2] != len(CHANNELS):
        raise ValueError(f"invalid sample score dimensions: {path}")
    if scores.shape[1] != len(indices) or not np.isfinite(scores).all() or (nonnegative and np.any(scores < 0)):
        raise ValueError(f"invalid or non-finite sample scores: {path}")
    if not nonnegative and np.any(np.abs(scores) > 1.0001):
        raise ValueError(f"ACC outside [-1,1]: {path}")
    return scores, indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _block_bootstrap_skill(
    model_mse: np.ndarray,
    reference_mse: np.ndarray,
    *,
    block_size: int,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Circular moving-block percentile CI for 1 - mean(model MSE)/mean(ref MSE)."""
    if model_mse.shape != reference_mse.shape or model_mse.ndim != 2:
        raise ValueError("paired MSE arrays must have shape (initializations, channels)")
    n = model_mse.shape[0]
    if not 1 <= block_size <= n or replicates < 100:
        raise ValueError("invalid bootstrap configuration")
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(replicates, n_blocks), dtype=np.int32)
    # The final block is truncated so each draw contains exactly n dates.
    block_lengths = np.full(n_blocks, block_size, dtype=np.int32)
    block_lengths[-1] = n - block_size * (n_blocks - 1)
    model_extended = np.concatenate((model_mse, model_mse[:block_size]), axis=0)
    ref_extended = np.concatenate((reference_mse, reference_mse[:block_size]), axis=0)
    model_cum = np.vstack((np.zeros((1, model_mse.shape[1])), np.cumsum(model_extended, axis=0)))
    ref_cum = np.vstack((np.zeros((1, model_mse.shape[1])), np.cumsum(ref_extended, axis=0)))
    model_draw = np.zeros((replicates, model_mse.shape[1]))
    ref_draw = np.zeros_like(model_draw)
    for j, length in enumerate(block_lengths):
        model_draw += model_cum[starts[:, j] + length] - model_cum[starts[:, j]]
        ref_draw += ref_cum[starts[:, j] + length] - ref_cum[starts[:, j]]
    if np.any(ref_draw <= 0):
        raise ValueError("reference MSE must be positive")
    skill = 1.0 - model_draw / ref_draw
    return np.quantile(skill, 0.025, axis=0), np.quantile(skill, 0.975, axis=0)


def compare_validation(
    baseline_scores: str | Path,
    dib_scores: str | Path,
    fno_scores: str | Path,
    output_dir: str | Path,
    *,
    canonical_path: str | Path | None = None,
    dib_first_case: str | Path | None = None,
    fno_first_case: str | Path | None = None,
    split_name: str = "val",
    block_size: int = 60,
    replicates: int = 1000,
    seed: int = 20260921,
) -> dict:
    """Compare identical initialization dates, full 40-step curves, selected-lead CIs."""
    paths = [Path(p).resolve() for p in (baseline_scores, dib_scores, fno_scores)]
    if split_name not in ("val", "test"):
        raise ValueError("comparison split must be val or test")
    with np.load(paths[0]) as baseline:
        indices = np.asarray(baseline["init_indices"], dtype=np.int32)
        persistence = np.asarray(baseline["persistence_mse"], dtype=np.float64)
        climatology = np.asarray(baseline["climatology_mse"], dtype=np.float64)
        persistence_acc = np.asarray(baseline["persistence_acc"], dtype=np.float64)
    dib, dib_indices = _load_scores(paths[1], "model_mse")
    fno, fno_indices = _load_scores(paths[2], "model_mse")
    dib_acc, dib_acc_indices = _load_scores(paths[1], "model_acc", nonnegative=False)
    fno_acc, fno_acc_indices = _load_scores(paths[2], "model_acc", nonnegative=False)
    if not np.array_equal(indices, dib_indices) or not np.array_equal(indices, fno_indices):
        raise ValueError("model and baseline initializations differ")
    if dib.shape != fno.shape or dib.shape != persistence.shape or dib.shape != climatology.shape:
        raise ValueError("model and baseline leads/channels differ")
    if not np.array_equal(indices, dib_acc_indices) or not np.array_equal(indices, fno_acc_indices):
        raise ValueError("ACC initialization dates differ")
    if dib_acc.shape != dib.shape or fno_acc.shape != dib.shape or persistence_acc.shape != dib.shape:
        raise ValueError("ACC array dimensions differ")
    if not np.isfinite(persistence_acc).all() or np.any(np.abs(persistence_acc) > 1.0001):
        raise ValueError("invalid persistence ACC")
    if dib.shape[0] != 40:
        raise ValueError("full 40-step scores are required")
    split = official_splits()[split_name]
    if len(indices) != split.sample_count(40) or not np.array_equal(
        indices, np.arange(split.start, split.start + len(indices), dtype=np.int32)
    ):
        raise ValueError("score initialization indices do not match requested split")
    if not np.isfinite(persistence).all() or not np.isfinite(climatology).all():
        raise ValueError("non-finite baseline scores")
    score_sets = {"DIB-FNO": dib, "Fixed FNO": fno, "Persistence": persistence, "Climatology": climatology}
    means = {name: values.mean(axis=1) for name, values in score_sets.items()}
    acc_means = {"DIB-FNO": dib_acc.mean(axis=1), "Fixed FNO": fno_acc.mean(axis=1),
                 "Persistence": persistence_acc.mean(axis=1)}
    selected = (1, 4, 12, 20, 40)
    ci = {}
    for lead in selected:
        d = dib[lead - 1]
        for reference_name, reference in (("Persistence", persistence), ("Fixed FNO", fno), ("Climatology", climatology)):
            low, high = _block_bootstrap_skill(
                d, reference[lead - 1], block_size=block_size,
                replicates=replicates, seed=seed + lead,
            )
            ci[(lead, reference_name)] = (low, high)
    rows = []
    for lead in range(1, 41):
        for i, channel in enumerate(CHANNELS):
            row = {"lead_hours": 6 * lead, "channel": channel, "unit": UNITS[i], "initializations": len(indices)}
            for name, value in means.items():
                row[f"{name}_rmse"] = float(np.sqrt(value[lead - 1, i]))
            for name, value in acc_means.items():
                row[f"{name}_acc"] = float(value[lead - 1, i])
            for name in ("Persistence", "Fixed FNO", "Climatology"):
                row[f"DIB_skill_vs_{name}"] = float(1 - means["DIB-FNO"][lead - 1, i] / means[name][lead - 1, i])
                if lead in selected:
                    low, high = ci[(lead, name)]
                    row[f"DIB_skill_vs_{name}_ci_low"] = float(low[i])
                    row[f"DIB_skill_vs_{name}_ci_high"] = float(high[i])
            rows.append(row)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / ("table03_model_comparison_all_leads.csv" if split_name == "val" else "table07_test_comparison_all_leads.csv")
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path = output_dir / ("table04_model_selected_leads.md" if split_name == "val" else "table08_test_selected_leads.md")
    period = "2016–2018 validation" if split_name == "val" else "2019–2023-01-10 test"
    lines = [
        f"# Measured {split_name} model comparison", "",
        f"Same {len(indices)} forecast initialization times at every lead; 6-hourly {period} subset.",
        f"Positive MSE skill means DIB-FNO is better than the named reference. The bracketed range is a descriptive 95% circular moving-block bootstrap interval over initialization dates ({block_size / 4:g}-day blocks, {replicates} draws); it does not cover retraining/seed uncertainty and is not multiple-comparison adjusted.",
        "", "| Lead | Channel | Unit | DIB-FNO RMSE | Fixed FNO RMSE | Persistence RMSE | Climatology RMSE | DIB MSE skill vs Fixed FNO [95% CI] |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for lead in selected:
        for channel in CHANNELS:
            row = rows[(lead - 1) * 8 + CHANNELS.index(channel)]
            lines.append(
                f"| {6 * lead} h | {channel} | {row['unit']} | {row['DIB-FNO_rmse']:.3f} | {row['Fixed FNO_rmse']:.3f} | "
                f"{row['Persistence_rmse']:.3f} | {row['Climatology_rmse']:.3f} | "
                f"{row['DIB_skill_vs_Fixed FNO']:+.3f} [{row['DIB_skill_vs_Fixed FNO_ci_low']:+.3f}, {row['DIB_skill_vs_Fixed FNO_ci_high']:+.3f}] |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.alpha": 0.18})
    colors = {"DIB-FNO": "#225e72", "Fixed FNO": "#8667a0", "Persistence": "#bc7642", "Climatology": "#67756d"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    for ax, channel in zip(axes.flat, ("Z500", "T850", "U850", "V850")):
        i = CHANNELS.index(channel)
        for name, value in means.items():
            ax.plot(np.arange(1, 41) / 4, np.sqrt(value[:, i]), label=name, color=colors[name], lw=1.8)
        ax.set(title=channel, xlabel="Lead time (days)", ylabel=f"Area-weighted RMSE ({UNITS[i]})", xlim=(0, 10))
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Measured 40-step autoregressive forecasts — {period}, seed 0", weight="bold")
    figure_path = output_dir / ("fig07_model_rmse_comparison.png" if split_name == "val" else "fig11_test_rmse_comparison.png")
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    for ax, channel in zip(axes.flat, ("Z500", "T850", "U850", "V850")):
        i = CHANNELS.index(channel)
        for name in ("Persistence", "Fixed FNO", "Climatology"):
            skill = 1 - means["DIB-FNO"][:, i] / means[name][:, i]
            ax.plot(np.arange(1, 41) / 4, skill, label=f"vs {name}", color=colors[name], lw=1.8)
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set(title=channel, xlabel="Lead time (days)", ylabel="DIB-FNO MSE skill", xlim=(0, 10))
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Negative values indicate DIB-FNO is worse than the reference", weight="bold")
    skill_path = output_dir / ("fig08_model_skill_comparison.png" if split_name == "val" else "fig12_test_skill_comparison.png")
    fig.savefig(skill_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(skill_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    for ax, channel in zip(axes.flat, ("Z500", "T850", "U850", "V850")):
        i = CHANNELS.index(channel)
        for name, value in acc_means.items():
            ax.plot(np.arange(1, 41) / 4, value[:, i], label=name, color=colors[name], lw=1.8)
        ax.axhline(0.6, color="0.4", ls="--", lw=0.8)
        ax.set(title=channel, xlabel="Lead time (days)", ylabel="Mean spatial ACC", xlim=(0, 10), ylim=(-0.45, 1.02))
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Anomaly correlation vs train-only climatology — {period}; climatology ACC undefined", weight="bold")
    acc_path = output_dir / ("fig14_model_acc_comparison.png" if split_name == "val" else "fig15_test_acc_comparison.png")
    fig.savefig(acc_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(acc_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    spatial_outputs = []
    spatial_inputs = []
    if canonical_path is not None and dib_first_case is not None and fno_first_case is not None:
        canonical_path = Path(canonical_path).resolve()
        dib_first_case = Path(dib_first_case).resolve()
        fno_first_case = Path(fno_first_case).resolve()
        with np.load(dib_first_case) as dib_case, np.load(fno_first_case) as fno_case:
            for lead in (4, 40):
                if not np.array_equal(dib_case[f"truth_{lead}"], fno_case[f"truth_{lead}"]):
                    raise ValueError("model first-case truths differ")
            if str(dib_case["init_time"]) != str(fno_case["init_time"]):
                raise ValueError("model first-case initialization dates differ")
            init = np.asarray(zarr.open_group(canonical_path, mode="r")["state"][int(indices[0])], dtype=np.float32)
            fig, axes = plt.subplots(2, 3, figsize=(12.1, 6.4), constrained_layout=True)
            for row, lead in enumerate((4, 40)):
                truth = dib_case[f"truth_{lead}"][0]
                fields = (dib_case[f"pred_{lead}"][0], fno_case[f"pred_{lead}"][0], init[0])
                errors = [field - truth for field in fields]
                limit = max(float(np.quantile(np.abs(np.stack(errors)), 0.995)), 1.0)
                for ax, name, error in zip(axes[row], ("DIB-FNO", "Fixed FNO", "Persistence"), errors):
                    im = ax.imshow(error, origin="lower", extent=(-2.8125, 357.1875, -90, 90),
                                   aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
                    ax.set(title=f"{name}: {lead * 6} h error", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
                    ax.set_xticks(np.arange(0, 361, 90))
                    ax.set_yticks(np.arange(-90, 91, 45))
                fig.colorbar(im, ax=axes[row].tolist(), shrink=0.75, label="Z500 forecast − analysis (gpm)")
            fig.suptitle(f"Single {split_name} initialization {str(dib_case['init_time'])}; illustrative, not an average", weight="bold")
            case_path = output_dir / ("fig09_z500_single_case_errors.png" if split_name == "val" else "fig13_test_z500_single_case_errors.png")
            fig.savefig(case_path, dpi=220, bbox_inches="tight", facecolor="white")
            fig.savefig(case_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
            plt.close(fig)
            spatial_outputs = [case_path, case_path.with_suffix(".svg")]
            spatial_inputs = [canonical_path, dib_first_case, fno_first_case]

    manifest = {
        "status": f"measured_{split_name}_comparison_single_seed",
        "scientific_caveat": "Time-block intervals are conditional on one trained seed, not a multi-seed significance result.",
        "initializations": len(indices), "lead_steps": 40,
        "bootstrap": {"method": "paired circular moving block", "block_size_6h_steps": block_size,
                      "replicates": replicates, "seed": seed, "interval": "percentile 95%", "selected_leads": selected},
        "score_inputs": [{"path": str(p), "sha256": _sha256(p)} for p in paths],
        "spatial_inputs": [{"path": str(p), "sha256": _sha256(p) if p.is_file() else None} for p in spatial_inputs],
        "outputs": [{"path": str(p), "sha256": _sha256(p)} for p in
                    (csv_path, md_path, figure_path, figure_path.with_suffix(".svg"), skill_path, skill_path.with_suffix(".svg"), acc_path, acc_path.with_suffix(".svg"), *spatial_outputs)],
    }
    manifest_path = output_dir / ("model_comparison_manifest.json" if split_name == "val" else "test_comparison_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "figures": 3 + bool(spatial_outputs), "tables": 2, "initializations": len(indices)}
