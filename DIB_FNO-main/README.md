# DIB-FNO

Dynamic Information Bottleneck Fourier Neural Operator for adaptive spectral learning in global weather forecasting.

DIB-FNO applies a sample-conditioned soft gate to the Fourier coefficients of an eight-channel atmospheric state before a residual spectral backbone. The repository contains the model, audited WeatherBench2 data contract, training and evaluation code, compact result tables, and Git-LFS checkpoints used in the reported three-seed experiment.

## Scope

- Grid: 32 × 64 (5.625°).
- Time step: 6 hours; autoregressive evaluation to 40 steps (10 days).
- Channels: Z500, Z850, T500, T850, U500, U850, V500, and V850.
- Models: DIB-FNO (3,106 parameters), learned static-gate FNO (3,173), and same-width fixed FNO (1,896).
- Training seeds: 0, 1, and 2; 15 epochs per reported run.
- Test starts: 5,844 inferred six-hourly initializations from 2019-01-01 00 UTC through 2022-12-31 18 UTC.

This is a focused local-subset study, not an official WeatherBench2 leaderboard submission. The coarse grid supports conclusions about large-scale circulation, not tropical-cyclone intensity or mesoscale extremes.

## Repository layout

```text
dibfno/                     model, data contract, training, metrics, evaluation
data/README.md              data availability and local directory policy
docs/                       data, model, and reproducibility documentation
examples/                   data-free synthetic API quickstart
results/                    publication-facing CSV result tables
tests/                      lightweight model and metric tests
weights/                    Git-LFS checkpoints and checkpoint documentation
run_experiment.py           public command-line entry point
WeatherBench2_Preprocessing_Guide.md
LICENSE                     Apache License 2.0 terms
```

Generated figures, review documents, office files, caches, local Zarr arrays, and duplicated delivery archives are intentionally excluded from the source tree.

## Installation

Python 3.10 or newer is required. Install Git LFS before cloning if checkpoints are needed.

```bash
git lfs install
git clone https://github.com/haohandongming-ship-it/DIB_FNO.git
cd DIB_FNO
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Data-free quickstart

Run the synthetic example to verify installation, tensor shapes, and the
input-dependent gate without downloading WeatherBench2 data:

```bash
python examples/synthetic_quickstart.py
```

The command prints a JSON summary for a `(2, 8, 32, 64)` synthetic batch and
the corresponding `(2, 1, 32, 33)` gates. This is an API smoke test only; its
random, untrained predictions are not scientific results.

## Computational requirements

- The test suite and synthetic quickstart run on a CPU with Python 3.10 or
  newer.
- Training and evaluation automatically use CUDA when available and otherwise
  fall back to the CPU. A CUDA-capable GPU is strongly recommended for the
  complete three-model, three-seed experiment.
- Full reproduction requires local storage for the four source Zarr stores,
  the generated canonical store, climatology, checkpoints, and evaluation
  arrays. Exact storage, memory, and runtime depend on Zarr chunking, hardware,
  and the chosen sample limits.
- Every training run records the selected device, PyTorch and CUDA versions,
  seed, command configuration, and data metadata in its output summary.

## Data

The WeatherBench2 arrays are not redistributed in this repository. Prepare the four local source stores described in [WeatherBench2_Preprocessing_Guide.md](WeatherBench2_Preprocessing_Guide.md), then audit them before building the canonical store:

```bash
python run_experiment.py audit --raw /path/to/wb2_subset
python run_experiment.py build \
  --raw /path/to/wb2_subset \
  --output data/canonical/wb2_8ch_6h.zarr \
  --accept-inferred-time-axis
python run_experiment.py climatology
```

The explicit `--accept-inferred-time-axis` flag is required because the local source stores do not contain a time coordinate. See [docs/DATA.md](docs/DATA.md) for the complete contract.

## Training

The reported DIB-FNO configuration is:

```bash
python run_experiment.py train \
  --model dibfno \
  --hidden-dim 16 \
  --mask-hidden-dim 8 \
  --blocks 1 \
  --epochs 15 \
  --batch-size 32 \
  --seed 0 \
  --output outputs/dibfno_seed0
```

Repeat with seeds 1 and 2. The parameter-near-matched static-gate control uses `--model static_fno --hidden-dim 17`; the strict same-width gate-removal control uses `--model fixed_fno --hidden-dim 16`.

## Evaluation

```bash
python run_experiment.py baseline-curves \
  --split test \
  --max-lead 40 \
  --output outputs/baselines_test.json

python run_experiment.py evaluate \
  --checkpoint weights/checkpoints/dibfno_h16_seed0.pt \
  --split test \
  --max-lead 40 \
  --output outputs/dibfno_seed0_test.json
```

Metrics are reported independently in physical units with cosine-latitude weighting applied once. Autoregressive step `k` is scored against the aligned `t+k` target. Anomaly correlation uses train-only calendar climatology.

## Reported evidence

The `results/` directory contains the publication-facing CSV tables. Paired intervals use 2,000 circular 15-day block-bootstrap replicates, joint date/seed resampling, and pointwise percentile 95% intervals without multiplicity correction.

For Z500, DIB-FNO minus the same-width fixed FNO is −5.799 gpm at 24 h (95% interval −6.396 to −5.311) and −25.812 gpm at 240 h (−40.671 to −12.818). Long-lead differences must be interpreted with the absolute-RMSE tables because part of the contrast is control-model rollout divergence. Wind and temperature differences at 240 h are not uniformly distinguishable from zero.

## Checkpoints

All 12 checkpoint files under `weights/` are tracked with Git LFS. After
cloning, run `git lfs pull` before evaluation; GitHub-generated source archives
may contain only LFS pointer files. Their roles, configurations, and
limitations are documented in [weights/README.md](weights/README.md).
Checkpoints store channel order, normalization, seed, model type, and model
configuration.

## Reproducibility and limitations

- The source Zarr time axis is absent; dates are a high-confidence inference from record length and six-hour spacing, not stored metadata.
- The held-out test set had been inspected during earlier development. The final tables remain descriptive evidence rather than a pristine confirmatory benchmark.
- All nine reported training runs reached the 15-epoch budget; the repository does not claim full optimization convergence.
- The frozen-mean-gate and top-C analyses are post hoc diagnostic ablations.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md) before reusing results.

## Citation

Until the archival paper citation is available, cite the repository and the exact commit used:

```bibtex
@software{dibfno2026,
  title   = {DIB-FNO: Dynamic Information Bottleneck Fourier Neural Operator},
  author  = {{DIB-FNO Authors}},
  year    = {2026},
  url     = {https://github.com/haohandongming-ship-it/DIB_FNO}
}
```

## License

The source code and documentation are licensed under the
[Apache License 2.0](LICENSE). Data and third-party materials remain subject to
their respective terms.
