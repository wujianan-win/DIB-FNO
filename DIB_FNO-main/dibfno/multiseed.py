"""Descriptive seed-to-seed summaries of independently trained forecast checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CHANNELS, UNITS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_model(paths: list[Path], expected_type: str) -> np.ndarray:
    all_rmse = []
    expected_parameters = 3106 if expected_type == "dibfno" else 3116
    for expected_seed, path in enumerate(paths):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["split"] != "val" or result["initializations"] != 4344 or len(result["records"]) != 40:
            raise ValueError(f"different validation protocol: {path}")
        if result["seed"] != expected_seed or result.get("model_type", expected_type) != expected_type:
            raise ValueError(f"wrong seed or model type: {path}")
        if result["parameter_count"] != expected_parameters or not 1 <= result["checkpoint_epoch"] <= 5:
            raise ValueError(f"different architecture or training horizon: {path}")
        if result["start_time"] != "2016-01-01T00" or result["last_init_time"] != "2018-12-21T18":
            raise ValueError(f"different validation dates: {path}")
        all_rmse.append(np.array([
            [record["channels"][channel]["rmse"] for channel in CHANNELS]
            for record in result["records"]
        ], dtype=np.float64))
    return np.stack(all_rmse)


def summarize_multiseed(
    dib_eval_paths: list[str | Path],
    fno_eval_paths: list[str | Path],
    baseline_json: str | Path,
    output_dir: str | Path,
) -> dict:
    """Require seeds 0/1/2, identical 40-step validation protocol and parameters."""
    if len(dib_eval_paths) != 3 or len(fno_eval_paths) != 3:
        raise ValueError("exactly seeds 0, 1 and 2 are required for each model")
    dib_paths = [Path(p).resolve() for p in dib_eval_paths]
    fno_paths = [Path(p).resolve() for p in fno_eval_paths]
    baseline_json = Path(baseline_json).resolve()
    dib = _read_model(dib_paths, "dibfno")
    fno = _read_model(fno_paths, "fixed_fno")
    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    if baseline["split"] != "val" or baseline["initializations"] != 4344 or len(baseline["records"]) != 40:
        raise ValueError("baseline uses a different validation protocol")
    persistence = np.array([[r["channels"][c]["persistence_rmse"] for c in CHANNELS] for r in baseline["records"]])
    climate = np.array([[r["channels"][c]["climatology_rmse"] for c in CHANNELS] for r in baseline["records"]])
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lead in range(40):
        for i, channel in enumerate(CHANNELS):
            rows.append({
                "lead_hours": (lead + 1) * 6, "channel": channel, "unit": UNITS[i],
                "DIB_RMSE_mean": dib[:, lead, i].mean(), "DIB_RMSE_sample_sd": dib[:, lead, i].std(ddof=1),
                "Fixed_FNO_RMSE_mean": fno[:, lead, i].mean(), "Fixed_FNO_RMSE_sample_sd": fno[:, lead, i].std(ddof=1),
                "persistence_RMSE": persistence[lead, i], "climatology_RMSE": climate[lead, i],
                "DIB_seeds_better_than_FNO": int(np.sum(dib[:, lead, i] < fno[:, lead, i])),
                "DIB_seeds_better_than_persistence": int(np.sum(dib[:, lead, i] < persistence[lead, i])),
            })
    csv_path = output_dir / "table05_multiseed_all_leads.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = output_dir / "table06_multiseed_selected_leads.md"
    lines = [
        "# Independent seed 0/1/2 validation summary", "",
        "Values are mean ± sample SD of three independently trained checkpoints' RMSE. The SD describes seed variability, not a confidence interval; the 4,344 validation initialization times overlap across leads.",
        "", "| Lead | Channel | DIB-FNO RMSE | Fixed FNO RMSE | Persistence | Climatology | DIB seeds better than Fixed FNO |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for lead in (1, 4, 12, 20, 40):
        for channel in ("Z500", "T850", "U850", "V850"):
            row = rows[(lead - 1) * 8 + CHANNELS.index(channel)]
            lines.append(
                f"| {row['lead_hours']} h | {channel} ({row['unit']}) | {row['DIB_RMSE_mean']:.3f} ± {row['DIB_RMSE_sample_sd']:.3f} | "
                f"{row['Fixed_FNO_RMSE_mean']:.3f} ± {row['Fixed_FNO_RMSE_sample_sd']:.3f} | "
                f"{row['persistence_RMSE']:.3f} | {row['climatology_RMSE']:.3f} | "
                f"{row['DIB_seeds_better_than_FNO']}/3 |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.alpha": 0.18})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    days = np.arange(1, 41) / 4
    for ax, channel in zip(axes.flat, ("Z500", "T850", "U850", "V850")):
        i = CHANNELS.index(channel)
        for values, label, color in ((dib, "DIB-FNO", "#225e72"), (fno, "Fixed FNO", "#8667a0")):
            mean = values[:, :, i].mean(axis=0)
            sd = values[:, :, i].std(axis=0, ddof=1)
            ax.plot(days, mean, lw=1.8, color=color, label=f"{label} mean")
            ax.fill_between(days, mean - sd, mean + sd, color=color, alpha=0.13)
        ax.plot(days, persistence[:, i], lw=1.3, color="#bc7642", label="Persistence")
        ax.plot(days, climate[:, i], lw=1.3, color="#67756d", label="Climatology")
        ax.set(title=channel, xlabel="Lead time (days)", ylabel=f"RMSE ({UNITS[i]})", xlim=(0, 10))
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Validation 2016–2018: three independent seeds per model; bands = ±1 sample SD", weight="bold")
    figure_path = output_dir / "fig10_multiseed_rmse.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    inputs = dib_paths + fno_paths + [baseline_json]
    outputs = [csv_path, md_path, figure_path, figure_path.with_suffix(".svg")]
    manifest = {
        "status": "measured_validation_three_seeds_per_architecture",
        "seed_set": [0, 1, 2], "initializations": 4344, "lead_steps": 40,
        "uncertainty": "sample SD across three independently trained seeds, not CI",
        "inputs": [{"path": str(p), "sha256": _sha256(p)} for p in inputs],
        "outputs": [{"path": str(p), "sha256": _sha256(p)} for p in outputs],
    }
    manifest_path = output_dir / "multiseed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "figures": 1, "tables": 2}
