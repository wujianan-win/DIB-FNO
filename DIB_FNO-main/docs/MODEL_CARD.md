# Model card

## Model description

DIB-FNO is a residual six-hour forecast operator in per-channel standardized space. A convolutional gate generator reads the log-amplitude input spectrum and produces a soft mask over the 32 × 33 real-FFT grid. The filtered state is passed to a complex-valued spectral channel mixer and a residual output projection.

The repository also provides two controls:

- `FixedFNO`: the identical hidden width and spectral backbone without a learned gate.
- `StaticGateFNO`: an input-independent learned spectral gate with a near-matched parameter count.

## Intended use

The released configuration is intended for research on adaptive spectral filtering and large-scale, coarse-grid weather prediction. It is not suitable for operational forecasting, emergency decisions, local extremes, tropical-cyclone intensity, or safety-critical use.

## Training evidence

The reported experiment uses three seeds and 15 epochs per model. All nine training runs reached the epoch budget. Checkpoint selection uses validation normalized RMSE; this is not evidence of full convergence.

## Evaluation caveats

- The 5.625° grid omits important mesoscale processes.
- The source time coordinate is inferred because it is absent from the local Zarr stores.
- The test set was inspected during earlier development, so the final test analysis is descriptive rather than pristine confirmatory evidence.
- Long-lead differences must be read together with absolute RMSE because some controls become unstable under autoregressive rollout.
- Bootstrap intervals are pointwise and are not adjusted for multiple comparisons.

## Checkpoint integrity

Checkpoint files store the model state, model type, model configuration, seed, channel order, normalization parameters, and selected epoch. Git LFS is required to retrieve the binary files.
