# Data contract

The model consumes float32 tensors with shape `(batch, 8, 32, 64)` and channel order `Z500, Z850, T500, T850, U500, U850, V500, V850`.

- Latitude increases from −87.1875° to +87.1875°.
- Longitude spans 0° to 354.375° in 5.625° increments.
- Z channels are geopotential height in gpm; T channels are K; U/V channels are m s⁻¹.
- Training-only per-channel mean and standard deviation are applied before fitting.
- Calendar climatology is indexed by month-day and UTC hour on a leap-year template. February 29 has fewer training samples and is not duplicated or overweighted.
- Forecast step `k` is always paired with target `t+k`.

The local source stores do not contain time coordinates. All calendar labels are explicitly marked as inferred. See [`../WeatherBench2_Preprocessing_Guide.md`](../WeatherBench2_Preprocessing_Guide.md) for preparation commands and split boundaries.
