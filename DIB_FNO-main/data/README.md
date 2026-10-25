# Data directory

WeatherBench2 arrays are not versioned in Git. This directory is reserved for locally generated canonical data:

```text
data/canonical/wb2_8ch_6h.zarr
data/canonical/climatology_v2.zarr
data/canonical/normalization.json
```

Follow [`../WeatherBench2_Preprocessing_Guide.md`](../WeatherBench2_Preprocessing_Guide.md). Source stores must remain read-only, and all preprocessing must be reproducible from the public command-line workflow.
