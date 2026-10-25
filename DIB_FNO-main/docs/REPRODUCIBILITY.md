# Reproducibility protocol

## Environment

Use Python 3.10 or newer and install the project with `python -m pip install -e .`. The verified local environment used Zarr 3.2.1. Run `pytest` before a new experiment.

## Data preparation

Follow `WeatherBench2_Preprocessing_Guide.md`. Do not change the source arrays to accommodate code assumptions. The `audit` command must pass before building canonical data.

## Reported training matrix

Run seeds 0, 1, and 2 for each configuration:

| Model | CLI options | Parameters |
| --- | --- | ---: |
| DIB-FNO | `--model dibfno --hidden-dim 16 --mask-hidden-dim 8 --blocks 1` | 3,106 |
| Static gate | `--model static_fno --hidden-dim 17 --blocks 1` | 3,173 |
| Fixed FNO | `--model fixed_fno --hidden-dim 16 --blocks 1` | 1,896 |

Common options are `--epochs 15 --batch-size 32 --rollout-steps 1`. Change only the seed and output path between repeated runs.

## Evaluation

Evaluate the same saved checkpoints on the same 5,844 test initializations and all 40 lead steps. Keep per-initialization score arrays until paired statistics have been generated.

For interval estimation, use 2,000 paired circular 15-day block replicates. Resample dates jointly across all compared models and resample the three seed identifiers jointly. Report pointwise percentile 95% intervals and explicitly state that no multiplicity correction was applied.

## Result audit

The checked-in CSV files under `results/` are immutable publication-facing summaries. New experiments must write to `outputs/` and must not overwrite those tables. Record the Git commit, dependency versions, data-manifest hashes, checkpoint hashes, seed, and command line for every new run.
