# WeatherBench2 preprocessing guide

## Data source and scientific boundary

The project uses a local 5.625° subset derived from WeatherBench2 ERA5 data. The repository does not redistribute the source or canonical arrays. The local source consists of four Zarr stores:

| Store | Variable | Required pressure levels |
| --- | --- | --- |
| `geopotential_500_850.zarr` | geopotential | 500 and 850 hPa |
| `temperature_500_850.zarr` | temperature | 500 and 850 hPa |
| `u_component_of_wind_500_850.zarr` | zonal wind | 500 and 850 hPa |
| `v_component_of_wind_500_850.zarr` | meridional wind | 500 and 850 hPa |

Each audited source array has shape `(93,544, 2, 64, 32)` and physical axis order `(time, level, longitude, latitude)`. The canonical builder transposes the spatial axes and writes `(time, channel, latitude, longitude)` with channel order:

```text
Z500, Z850, T500, T850, U500, U850, V500, V850
```

Geopotential is divided by standard gravity, 9.80665 m s⁻², to obtain geopotential height in gpm. Temperature is stored in K and wind components in m s⁻¹.

## Time-coordinate caveat

The local source stores contain no explicit time coordinate. The sequence is treated as six-hourly data from 1959-01-01 00 UTC through 2023-01-10 18 UTC because the 93,544 records match that interval. This is a high-confidence inference, not source metadata. The builder therefore requires explicit acknowledgement through `--accept-inferred-time-axis`.

## Splits

| Split | Inferred interval | Use |
| --- | --- | --- |
| Train | 1959-01-01 to 2015-12-31 | normalization, climatology, fitting |
| Validation | 2016-01-01 to 2018-12-31 | checkpoint selection and exploratory ablations |
| Test | starts from 2019-01-01 through 2022-12-31 18 UTC | final descriptive evaluation |

The final 240 h test target is inferred as 2023-01-10 18 UTC. Normalization and calendar climatology are computed from the training split only.

## Directory layout

Place the source stores outside the repository or under an ignored local directory:

```text
/path/to/wb2_subset/
  geopotential_500_850.zarr/
  temperature_500_850.zarr/
  u_component_of_wind_500_850.zarr/
  v_component_of_wind_500_850.zarr/
```

## Audit and build

```bash
python run_experiment.py audit --raw /path/to/wb2_subset

python run_experiment.py build \
  --raw /path/to/wb2_subset \
  --output data/canonical/wb2_8ch_6h.zarr \
  --accept-inferred-time-axis

python run_experiment.py climatology \
  --canonical data/canonical/wb2_8ch_6h.zarr \
  --output data/canonical/climatology_v2.zarr
```

The audit checks store names, shapes, levels, dtype, spatial orientation, and basic physical ranges. It never modifies the source stores. The canonical builder records metadata hashes and the inferred-time status in its manifest.

## Data policy

Do not commit raw or canonical Zarr arrays to Git. Retain the generated build manifest, normalization metadata, and experiment configuration with each local run. The `data/canonical/` and `*.zarr/` paths are ignored by default.
