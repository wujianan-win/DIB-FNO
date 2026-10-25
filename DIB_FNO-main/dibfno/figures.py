"""Publication-oriented figures and tables from measured WB2 artifacts only."""

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

from .data import CHANNELS, LATITUDES, LONGITUDES, UNITS, climatology_slots, inferred_times


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(fig, target: Path, outputs: list[Path]) -> None:
    fig.savefig(target.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    outputs.extend((target.with_suffix(".png"), target.with_suffix(".svg")))
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18,
        "figure.dpi": 120, "savefig.dpi": 220,
    })


def make_figures(
    canonical_path: str | Path,
    climatology_path: str | Path,
    baseline_json: str | Path,
    output_dir: str | Path,
) -> dict:
    """Render maps, diagnostics, baseline curves, and CSV/Markdown tables."""
    _style()
    canonical_path = Path(canonical_path).resolve()
    climatology_path = Path(climatology_path).resolve()
    baseline_json = Path(baseline_json).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_result = json.loads(baseline_json.read_text(encoding="utf-8"))
    if raw_result["split"] != "val" or len(raw_result["records"]) != 40:
        raise ValueError("expected full 40-step validation baseline artifact")
    group = zarr.open_group(canonical_path, mode="r")
    climate_group = zarr.open_group(climatology_path, mode="r")
    climate = np.asarray(climate_group["mean"][:], dtype=np.float32)
    if tuple(climate.shape) != (8, 366, 4, 32, 64):
        raise ValueError("climatology shape mismatch")
    outputs: list[Path] = []
    channels = ("Z500", "T850", "U850", "V850")
    colors = {"persistence": "#1d5c63", "climatology": "#c26b35"}
    hours = np.asarray([r["lead_hours"] for r in raw_result["records"]])

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    for ax, channel in zip(axes.flat, channels):
        i = CHANNELS.index(channel)
        for name in ("persistence", "climatology"):
            values = [r["channels"][channel][f"{name}_rmse"] for r in raw_result["records"]]
            ax.plot(hours / 24, values, color=colors[name], lw=2, label=name.capitalize())
        ax.set(title=channel, xlabel="Lead time (days)", ylabel=f"Area-weighted RMSE ({UNITS[i]})", xlim=(0, 10))
        ax.set_xticks(np.arange(0, 11, 2))
        ax.legend(frameon=False, loc="best")
    fig.suptitle("WeatherBench2 subset: physical baseline skill (validation 2016–2018)", weight="bold")
    _save(fig, output_dir / "fig01_baseline_rmse", outputs)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    for ax, channel in zip(axes.flat, channels):
        acc = [r["channels"][channel]["persistence_acc"] for r in raw_result["records"]]
        ax.plot(hours / 24, acc, color=colors["persistence"], lw=2)
        ax.axhline(0.6, color="#8b4b42", ls="--", lw=1, label="ACC = 0.6")
        ax.set(title=channel, xlabel="Lead time (days)", ylabel="Persistence ACC", xlim=(0, 10), ylim=(-0.25, 1.02))
        ax.set_xticks(np.arange(0, 11, 2))
        ax.legend(frameon=False)
    fig.suptitle("Anomaly correlation against train-only calendar climatology", weight="bold")
    _save(fig, output_dir / "fig02_persistence_acc", outputs)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    for ax, channel in zip(axes.flat, channels):
        skill = [r["channels"][channel]["persistence_mse_skill_vs_climatology"] for r in raw_result["records"]]
        ax.plot(hours / 24, skill, color="#37558e", lw=2)
        ax.axhline(0, color="#3a3f45", lw=1)
        ax.set(title=channel, xlabel="Lead time (days)", ylabel="Persistence MSE skill vs climatology", xlim=(0, 10))
        ax.set_xticks(np.arange(0, 11, 2))
    fig.suptitle("Negative values indicate climatology is better", weight="bold")
    _save(fig, output_dir / "fig03_persistence_skill", outputs)

    # A single real reanalysis state; no simulated fields or hand-tuned patterns.
    sample_index = 83_276  # 2016-01-01T00, first validation analysis.
    state = np.asarray(group["state"][sample_index], dtype=np.float32)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), constrained_layout=True)
    for ax, channel, cmap in zip(axes.flat, channels, ("viridis", "coolwarm", "RdBu_r", "RdBu_r")):
        idx = CHANNELS.index(channel)
        field = state[idx]
        image = ax.imshow(field, origin="lower", extent=(-2.8125, 357.1875, -90, 90), aspect="auto", cmap=cmap)
        ax.set(title=f"{channel} ({UNITS[idx]})", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
        ax.set_xticks(np.arange(0, 361, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        fig.colorbar(image, ax=ax, shrink=0.80)
    fig.suptitle("ERA5 analysis fields, inferred valid time 2016-01-01 00 UTC; 5.625° grid", weight="bold")
    _save(fig, output_dir / "fig04_real_analysis_maps", outputs)

    # Zonal means from 57 years of train-only calendar climatology.
    zonal = climate.mean(axis=(1, 2, 4))
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0), constrained_layout=True)
    for ax, channel in zip(axes, ("Z500", "T850", "U850")):
        idx = CHANNELS.index(channel)
        ax.plot(zonal[idx], LATITUDES, color="#205b68", lw=2)
        ax.set(title=channel, xlabel=f"Zonal mean ({UNITS[idx]})", ylabel="Latitude (°N)", ylim=(-90, 90))
        ax.axhline(0, color="0.6", lw=0.8)
    fig.suptitle("Train-period zonal climatology (1959–2015)", weight="bold")
    _save(fig, output_dir / "fig05_zonal_profiles", outputs)

    # Calendar cycle of area-weighted global-mean T850 and NH midlatitude U500.
    lat_w = np.cos(np.deg2rad(LATITUDES.astype(np.float64)))
    lat_w /= lat_w.sum()
    t850_daily = climate[3].mean(axis=(1, 3)) @ lat_w
    northern_mid = (LATITUDES >= 30) & (LATITUDES <= 60)
    u500_daily = climate[4][:, :, northern_mid, :].mean(axis=(1, 2, 3))
    day = np.arange(1, 367)
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(day, t850_daily, color="#bc6934", lw=1.8)
    axes[0].set(ylabel="Global T850 (K)", title="Area-weighted global mean")
    axes[1].plot(day, u500_daily, color="#275e83", lw=1.8)
    axes[1].set(xlabel="Calendar day on leap-year template", ylabel="30–60°N U500 (m/s)", title="Northern midlatitude zonal wind")
    for ax in axes:
        ax.set_xlim(1, 366)
        ax.set_xticks([1, 60, 121, 182, 244, 305, 366])
    fig.suptitle("Annual cycle derived only from 1959–2015 observations", weight="bold")
    _save(fig, output_dir / "fig06_annual_cycle", outputs)

    # All leads / channels in machine-readable CSV and a compact paper table.
    csv_path = output_dir / "table01_baseline_all_leads.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "initializations", "lead_hours", "channel", "unit", "persistence_rmse", "climatology_rmse", "persistence_acc", "persistence_bias", "climatology_bias", "persistence_mse_skill_vs_climatology"))
        for record in raw_result["records"]:
            for i, channel in enumerate(CHANNELS):
                values = record["channels"][channel]
                writer.writerow((raw_result["split"], raw_result["initializations"], record["lead_hours"], channel, UNITS[i], values["persistence_rmse"], values["climatology_rmse"], values["persistence_acc"], values["persistence_bias"], values["climatology_bias"], values["persistence_mse_skill_vs_climatology"]))
    outputs.append(csv_path)
    md_path = output_dir / "table02_selected_leads.md"
    lines = [
        "# Validation baseline results (2016–2018)", "",
        f"Identical {raw_result['initializations']} initialization times at every lead; time axis inferred from official WB2 provenance.",
        "", "| Lead | Channel | Unit | Persistence RMSE | Climatology RMSE | Persistence ACC |", "|---:|---|---|---:|---:|---:|",
    ]
    for lead in (1, 4, 12, 20, 40):
        record = raw_result["records"][lead - 1]
        for i, channel in enumerate(CHANNELS):
            item = record["channels"][channel]
            lines.append(f"| {record['lead_hours']} h | {channel} | {UNITS[i]} | {item['persistence_rmse']:.3f} | {item['climatology_rmse']:.3f} | {item['persistence_acc']:.3f} |")
    lines.extend(("", "ACC of the climatology forecast is undefined (zero anomaly) and is intentionally not reported.", ""))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    outputs.append(md_path)

    manifest = {
        "status": "measured_validation_baselines_only",
        "script": str(Path(__file__).resolve()),
        "canonical": str(canonical_path),
        "climatology": str(climatology_path),
        "baseline_artifact": str(baseline_json),
        "baseline_artifact_sha256": _sha256(baseline_json),
        "no_model_skill_claim": True,
        "outputs": [{"file": str(path), "sha256": _sha256(path)} for path in outputs],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "figures": 6, "tables": 2, "files": len(outputs)}
