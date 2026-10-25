"""Train-only calendar climatology rebuilt without changing source data."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import zarr

from .data import CHANNELS, climatology_slots, inferred_times, official_splits


def build_climatology_v2(
    canonical_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    block_size: int = 512,
) -> dict:
    """Compute an unsmoothed month-day/hour climatology using train only."""
    canonical = zarr.open_group(Path(canonical_path), mode="r")["state"]
    if tuple(canonical.shape) != (93_544, 8, 32, 64):
        raise ValueError(f"unexpected canonical shape: {canonical.shape}")
    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    train_stop = official_splits()["train"].stop
    slots = climatology_slots(inferred_times()[:train_stop])
    counts = np.bincount(slots, minlength=366 * 4).astype(np.int16)
    if counts.min() != 14 or counts.max() != 57:
        raise ValueError(f"unexpected climatology group counts: {counts.min()}..{counts.max()}")
    group = zarr.open_group(output, mode="w")
    mean_array = group.create_array(
        "mean", shape=(8, 366, 4, 32, 64), chunks=(1, 1, 4, 32, 64),
        dtype="float32", fill_value=np.nan,
        dimension_names=("channel", "calendar_day", "hour_index", "latitude", "longitude"),
    )
    group.create_array("count", data=counts.reshape(366, 4), chunks=(366, 4))
    for channel in range(8):
        totals = np.zeros((366 * 4, 32, 64), dtype=np.float64)
        for start in range(0, train_stop, block_size):
            stop = min(train_stop, start + block_size)
            values = np.asarray(canonical[start:stop, channel], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite canonical values in channel {channel}")
            local_slots = slots[start:stop]
            for slot in np.unique(local_slots):
                totals[slot] += values[local_slots == slot].sum(axis=0)
        mean_array[channel] = (totals / counts[:, None, None]).reshape(366, 4, 32, 64).astype(np.float32)
    group.attrs.update({
        "source": str(Path(canonical_path).resolve()),
        "train_interval": "1959-01-01T00 to 2015-12-31T18 inclusive",
        "grouping": "calendar month-day on leap-year template x UTC hour",
        "smoothing_days": 0,
        "leap_day_count": 14,
        "ordinary_day_count": 57,
        "channels": list(CHANNELS),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    result = {
        "path": str(output), "shape": list(mean_array.shape),
        "count_min": int(counts.min()), "count_max": int(counts.max()),
        "feb29_count": int(counts[59 * 4]),
        "mar1_count": int(counts[60 * 4]),
        "grouping": group.attrs["grouping"],
    }
    (output.parent / "climatology_v2_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
