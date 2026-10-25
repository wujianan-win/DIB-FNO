"""Command-line interface for the public DIB-FNO WeatherBench2 workflow."""

from __future__ import annotations

import argparse
import json

import numpy as np

from dibfno.climatology import build_climatology_v2
from dibfno.comparison import compare_validation
from dibfno.convergence_training import train_to_convergence
from dibfno.data import WB2RawStore, build_canonical, official_splits, validate_physics
from dibfno.evaluate import evaluate_physical_baselines
from dibfno.gate_ablations import build_train_mean_gate
from dibfno.mask_diagnostics import diagnose_mask
from dibfno.metrics import persistence_rmse
from dibfno.model_evaluation import evaluate_checkpoint
from dibfno.multiseed import summarize_multiseed
from dibfno.train import train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate DIB-FNO on the audited 5.625-degree WeatherBench2 subset."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Audit the read-only source Zarr stores.")
    audit.add_argument("--raw", required=True)

    build = sub.add_parser("build", help="Build the canonical eight-channel Zarr store.")
    build.add_argument("--raw", required=True)
    build.add_argument("--output", default="data/canonical/wb2_8ch_6h.zarr")
    build.add_argument("--accept-inferred-time-axis", action="store_true", required=True)
    build.add_argument("--overwrite", action="store_true")

    climatology = sub.add_parser("climatology", help="Build train-only calendar climatology.")
    climatology.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    climatology.add_argument("--output", default="data/canonical/climatology_v2.zarr")
    climatology.add_argument("--overwrite", action="store_true")

    train = sub.add_parser("train", help="Train one model with validation-based checkpoint selection.")
    train.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    train.add_argument("--normalization", default="data/canonical/normalization.json")
    train.add_argument("--output", required=True)
    train.add_argument("--model", choices=("dibfno", "fixed_fno", "static_fno"), default="dibfno")
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--rollout-steps", type=int, default=1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--hidden-dim", type=int, default=16)
    train.add_argument("--blocks", type=int, default=1)
    train.add_argument("--mask-hidden-dim", type=int, default=8)
    train.add_argument("--max-train-samples", type=int)
    train.add_argument("--max-val-samples", type=int)

    converge = sub.add_parser("train-converge", help="Train with an epoch budget and patience.")
    converge.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    converge.add_argument("--normalization", default="data/canonical/normalization.json")
    converge.add_argument("--output", required=True)
    converge.add_argument("--model", choices=("dibfno", "fixed_fno", "static_fno"), default="dibfno")
    converge.add_argument("--hidden-dim", type=int, required=True)
    converge.add_argument("--mask-hidden-dim", type=int, default=8)
    converge.add_argument("--blocks", type=int, default=1)
    converge.add_argument("--max-epochs", type=int, default=50)
    converge.add_argument("--patience", type=int, default=8)
    converge.add_argument("--min-epochs", type=int, default=5)
    converge.add_argument("--temperature-epochs", type=int, default=5)
    converge.add_argument("--batch-size", type=int, default=32)
    converge.add_argument("--rollout-steps", type=int, default=1)
    converge.add_argument("--seed", type=int, default=0)
    converge.add_argument("--threads", type=int, default=12)

    baseline = sub.add_parser("baseline", help="Evaluate persistence at one lead time.")
    baseline.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    baseline.add_argument("--split", choices=("train", "val", "test"), default="val")
    baseline.add_argument("--lead-steps", type=int, default=1)
    baseline.add_argument("--max-samples", type=int)

    curves = sub.add_parser("baseline-curves", help="Evaluate persistence and climatology to 10 days.")
    curves.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    curves.add_argument("--climatology", default="data/canonical/climatology_v2.zarr")
    curves.add_argument("--split", choices=("train", "val", "test"), default="val")
    curves.add_argument("--max-lead", type=int, default=40)
    curves.add_argument("--max-samples", type=int)
    curves.add_argument("--output", required=True)

    evaluate = sub.add_parser("evaluate", help="Evaluate a checkpoint with aligned rollout targets.")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    evaluate.add_argument("--normalization", default="data/canonical/normalization.json")
    evaluate.add_argument("--climatology", default="data/canonical/climatology_v2.zarr")
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--max-lead", type=int, default=40)
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--fixed-mask")
    evaluate.add_argument("--output", required=True)

    compare = sub.add_parser("compare", help="Create paired comparison tables and plots.")
    compare.add_argument("--baseline-scores", required=True)
    compare.add_argument("--dib-scores", required=True)
    compare.add_argument("--fno-scores", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    compare.add_argument("--dib-first-case")
    compare.add_argument("--fno-first-case")
    compare.add_argument("--split", choices=("val", "test"), default="val")
    compare.add_argument("--block-size", type=int, default=60)
    compare.add_argument("--replicates", type=int, default=1000)

    seeds = sub.add_parser("summarize-seeds", help="Summarize three-seed evaluation JSON files.")
    seeds.add_argument("--dib", nargs=3, required=True)
    seeds.add_argument("--fixed", nargs=3, required=True)
    seeds.add_argument("--baseline", required=True)
    seeds.add_argument("--output", required=True)

    mask = sub.add_parser("mask-diagnostics", help="Measure input-dependent gate variability.")
    mask.add_argument("--checkpoint", required=True)
    mask.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    mask.add_argument("--normalization", default="data/canonical/normalization.json")
    mask.add_argument("--output", required=True)

    mean_gate = sub.add_parser("build-mean-gate", help="Build a train-only frozen-mean-gate ablation.")
    mean_gate.add_argument("--checkpoint", required=True)
    mean_gate.add_argument("--canonical", default="data/canonical/wb2_8ch_6h.zarr")
    mean_gate.add_argument("--normalization", default="data/canonical/normalization.json")
    mean_gate.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit":
        raw = WB2RawStore(args.raw)
        result = {
            **raw.audit,
            "physics": validate_physics(raw, np.linspace(0, 93_543, 12, dtype=int).tolist()),
        }
    elif args.command == "build":
        result = build_canonical(
            args.raw,
            args.output,
            accept_inferred_time_axis=args.accept_inferred_time_axis,
            overwrite=args.overwrite,
        )
    elif args.command == "climatology":
        result = build_climatology_v2(args.canonical, args.output, overwrite=args.overwrite)
    elif args.command == "train":
        result = train_model(
            args.canonical,
            args.normalization,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            n_blocks=args.blocks,
            mask_hidden_dim=args.mask_hidden_dim,
            model_type=args.model,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
    elif args.command == "train-converge":
        result = train_to_convergence(
            args.canonical,
            args.normalization,
            args.output,
            model_type=args.model,
            hidden_dim=args.hidden_dim,
            mask_hidden_dim=args.mask_hidden_dim,
            n_blocks=args.blocks,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_epochs=args.min_epochs,
            temperature_epochs=args.temperature_epochs,
            batch_size=args.batch_size,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
            threads=args.threads,
        )
    elif args.command == "baseline":
        result = persistence_rmse(
            args.canonical,
            official_splits()[args.split],
            lead_steps=args.lead_steps,
            max_samples=args.max_samples,
        )
    elif args.command == "baseline-curves":
        result = evaluate_physical_baselines(
            args.canonical,
            args.climatology,
            official_splits()[args.split],
            max_lead=args.max_lead,
            max_samples=args.max_samples,
            output_path=args.output,
        )
    elif args.command == "evaluate":
        result = evaluate_checkpoint(
            args.checkpoint,
            args.canonical,
            args.normalization,
            args.climatology,
            args.output,
            split_name=args.split,
            max_lead=args.max_lead,
            max_samples=args.max_samples,
            fixed_mask_path=args.fixed_mask,
        )
    elif args.command == "compare":
        result = compare_validation(
            args.baseline_scores,
            args.dib_scores,
            args.fno_scores,
            args.output,
            canonical_path=args.canonical,
            dib_first_case=args.dib_first_case,
            fno_first_case=args.fno_first_case,
            split_name=args.split,
            block_size=args.block_size,
            replicates=args.replicates,
        )
    elif args.command == "summarize-seeds":
        result = summarize_multiseed(args.dib, args.fixed, args.baseline, args.output)
    elif args.command == "mask-diagnostics":
        result = diagnose_mask(args.checkpoint, args.canonical, args.normalization, args.output)
    else:
        result = build_train_mean_gate(
            args.checkpoint,
            args.canonical,
            args.normalization,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
