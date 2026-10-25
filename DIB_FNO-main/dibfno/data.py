"""Strict data access for the local WeatherBench2 subset.

The source stores contain valid fields but invalid spatial dimension labels and
no time coordinate.  This module corrects those issues in a derived canonical
store; it never modifies the source stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

GRAVITY = np.float32(9.80665)
EXPECTED_RAW_SHAPE = (93_544, 2, 64, 32)  # time, level, longitude, latitude
LEVELS = (500, 850)
LATITUDES = np.arange(-87.1875, 90.0, 5.625, dtype=np.float32)
LONGITUDES = np.arange(0.0, 360.0, 5.625, dtype=np.float32)
INFERRED_START = np.datetime64("1959-01-01T00", "h")
TIME_STEP_HOURS = 6

CHANNELS = (
    "Z500", "Z850", "T500", "T850", "U500", "U850", "V500", "V850"
)
UNITS = ("gpm", "gpm", "K", "K", "m s-1", "m s-1", "m s-1", "m s-1")
RAW_VARIABLES = (
    ("geopotential", "geopotential_500_850.zarr"),
    ("temperature", "temperature_500_850.zarr"),
    ("u_component_of_wind", "u_component_of_wind_500_850.zarr"),
    ("v_component_of_wind", "v_component_of_wind_500_850.zarr"),
)


def inferred_times() -> np.ndarray:
    """Return the inferred official WB2 time axis, explicitly not source metadata."""
    return INFERRED_START + np.arange(EXPECTED_RAW_SHAPE[0]) * np.timedelta64(
        TIME_STEP_HOURS, "h"
    )


def climatology_slots(times: np.ndarray) -> np.ndarray:
    """Map calendar month/day and UTC hour to a 366 x 4 leap-template grid.

    Using raw day-of-year would mix March 1 in common years with February 29
    in leap years.  The leap template keeps the same calendar date aligned.
    """
    days = times.astype("datetime64[D]")
    months = times.astype("datetime64[M]")
    month_number = months.astype(np.int64) % 12
    day_number = (days - months).astype(np.int16)
    leap_month_starts = np.asarray(
        [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
        dtype=np.int16,
    )
    day_index = leap_month_starts[month_number] + day_number
    hour_index = ((times - days).astype("timedelta64[h]").astype(np.int16) // 6)
    if np.any(hour_index < 0) or np.any(hour_index > 3):
        raise ValueError("time axis contains values outside 00/06/12/18 UTC")
    return day_index.astype(np.int32) * 4 + hour_index.astype(np.int32)


def resolve_raw_root(path: str | Path) -> Path:
    """Resolve the directory containing exactly the four required stores."""
    root = Path(path).resolve()
    missing = [filename for _, filename in RAW_VARIABLES if not (root / filename).is_dir()]
    if missing:
        raise FileNotFoundError(f"{root} is missing WeatherBench2 stores: {missing}")
    return root


def _sha256_metadata(store: Path) -> str:
    """Hash metadata only; hashing 5 GB payload is an explicit separate audit task."""
    digest = hashlib.sha256()
    for item in sorted(store.rglob("zarr.json")):
        digest.update(item.relative_to(store).as_posix().encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Split:
    name: str
    start: int
    stop: int

    def sample_count(self, rollout_steps: int) -> int:
        return max(0, self.stop - self.start - rollout_steps)


def official_splits() -> dict[str, Split]:
    """Continuous, non-overlapping splits; targets never cross a boundary."""
    times = inferred_times()
    i2016 = int(np.searchsorted(times, np.datetime64("2016-01-01T00", "h")))
    i2019 = int(np.searchsorted(times, np.datetime64("2019-01-01T00", "h")))
    return {
        "train": Split("train", 0, i2016),
        "val": Split("val", i2016, i2019),
        "test": Split("test", i2019, len(times)),
    }


class WB2RawStore:
    """Read-only view that repairs the raw axis order on access."""

    def __init__(self, root: str | Path):
        self.root = resolve_raw_root(root)
        self.arrays: list[zarr.Array] = []
        self.audit = self._open_and_validate()

    def _open_and_validate(self) -> dict:
        records = []
        for variable, filename in RAW_VARIABLES:
            group = zarr.open_group(self.root / filename, mode="r")
            if variable not in group:
                raise ValueError(f"{filename} does not contain variable {variable}")
            array = group[variable]
            if tuple(array.shape) != EXPECTED_RAW_SHAPE:
                raise ValueError(
                    f"{filename} has shape {array.shape}; expected {EXPECTED_RAW_SHAPE}"
                )
            if np.dtype(array.dtype) != np.dtype("float32"):
                raise TypeError(f"{filename} has dtype {array.dtype}; expected float32")
            levels = np.asarray(group["level"][:])
            if not np.array_equal(levels, LEVELS):
                raise ValueError(
                    f"{filename} has levels {levels.tolist()}; expected {LEVELS}"
                )
            # These coordinate arrays are known preprocessing indices, not degrees.
            if not np.array_equal(group["latitude"][:], np.arange(64)):
                raise ValueError(
                    f"{filename} first spatial axis is not the expected 0..63 index"
                )
            if not np.array_equal(group["longitude"][:], np.arange(32)):
                raise ValueError(
                    f"{filename} second spatial axis is not the expected 0..31 index"
                )
            self.arrays.append(array)
            records.append(
                {
                    "variable": variable,
                    "store": filename,
                    "shape": list(array.shape),
                    "chunks": list(array.chunks),
                    "dtype": str(array.dtype),
                    "metadata_sha256": _sha256_metadata(self.root / filename),
                }
            )
        return {
            "raw_root": str(self.root),
            "stores": records,
            "axis_contract": "source=(time,level,longitude,latitude); canonical=(time,channel,latitude,longitude)",
            "time_axis_status": "inferred_not_present_in_source",
            "time_axis_basis": (
                "93,544 six-hourly steps match the official WeatherBench2 ERA5 "
                "1959-01-01T00 through 2023-01-10T18 record"
            ),
        }

    def read_channel(self, channel: int, selection=slice(None)) -> np.ndarray:
        """Read one canonical channel as (..., latitude, longitude), float32."""
        if not 0 <= channel < len(CHANNELS):
            raise IndexError(channel)
        var_idx, level_idx = divmod(channel, 2)
        raw = np.asarray(self.arrays[var_idx][selection, level_idx, :, :], dtype=np.float32)
        canonical = np.swapaxes(raw, -2, -1)
        if var_idx == 0:
            canonical = canonical / GRAVITY
        return np.ascontiguousarray(canonical, dtype=np.float32)

    def read_state(self, index: int) -> np.ndarray:
        return np.stack([self.read_channel(c, index) for c in range(8)], axis=0)


def validate_physics(raw: WB2RawStore, sample_indices: Sequence[int]) -> dict[str, float]:
    """Run axis and broad physical-range checks on real source fields."""
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")
    t850 = np.stack([raw.read_channel(3, i) for i in sample_indices])
    z500 = np.stack([raw.read_channel(0, i) for i in sample_indices])
    u850 = np.stack([raw.read_channel(5, i) for i in sample_indices])
    equator = np.abs(LATITUDES) < 10
    polar = np.abs(LATITUDES) > 75
    t_contrast = float(t850[:, equator, :].mean() - t850[:, polar, :].mean())
    z_contrast = float(z500[:, equator, :].mean() - z500[:, polar, :].mean())
    if t_contrast < 25:
        raise ValueError(
            f"latitude-axis physics check failed: T850 equator-to-pole "
            f"contrast is only {t_contrast:.2f} K"
        )
    if z_contrast < 600:
        raise ValueError(
            f"latitude-axis physics check failed: Z500 equator-to-pole "
            f"contrast is only {z_contrast:.2f} gpm"
        )
    if not (200 <= float(t850.min()) and float(t850.max()) <= 320):
        raise ValueError("T850 lies outside the broad physical range [200, 320] K")
    if float(np.abs(u850).max()) >= 120:
        raise ValueError("U850 lies outside the broad physical range |U| < 120 m/s")
    # This is the crucial north/south test: the first spatial row is warm in
    # austral summer and cold in austral winter; the last row has the inverse
    # boreal seasonal phase. A meridional gradient alone cannot establish sign.
    times = inferred_times()
    jan_first, jul_first, jan_last, jul_last = [], [], [], []
    for year in (1979, 1989, 1999, 2009, 2014):
        jan_idx = int(np.searchsorted(times, np.datetime64(f"{year}-01-15T00", "h")))
        jul_idx = int(np.searchsorted(times, np.datetime64(f"{year}-07-15T00", "h")))
        jan = raw.read_channel(3, jan_idx)
        jul = raw.read_channel(3, jul_idx)
        jan_first.append(float(jan[:5].mean()))
        jul_first.append(float(jul[:5].mean()))
        jan_last.append(float(jan[-5:].mean()))
        jul_last.append(float(jul[-5:].mean()))
    first_jul_minus_jan = float(np.mean(jul_first) - np.mean(jan_first))
    last_jul_minus_jan = float(np.mean(jul_last) - np.mean(jan_last))
    if not (first_jul_minus_jan < -5 and last_jul_minus_jan > 5):
        raise ValueError(
            "north-south seasonal-phase check failed: the first endpoint should be "
            "Southern Hemisphere and the last endpoint Northern Hemisphere; "
            f"observed {first_jul_minus_jan:.2f}/{last_jul_minus_jan:.2f} K"
        )
    return {
        "t850_equator_minus_polar_K": t_contrast,
        "z500_equator_minus_polar_gpm": z_contrast,
        "t850_min_K": float(t850.min()),
        "t850_max_K": float(t850.max()),
        "u850_abs_max_m_s": float(np.abs(u850).max()),
        "first_polar_rows_july_minus_january_K": first_jul_minus_jan,
        "last_polar_rows_july_minus_january_K": last_jul_minus_jan,
        "latitude_orientation": "ascending_south_to_north",
    }


def repair_derived_latitude(canonical_path: str | Path) -> dict:
    """Repair only the derived latitude coordinate; the 5 GB fields stay intact."""
    canonical_path = Path(canonical_path).resolve()
    group = zarr.open_group(canonical_path, mode="a")
    state = group["state"]
    if tuple(state.shape) != (93_544, 8, 32, 64):
        raise ValueError("unexpected derived state shape")
    previous = np.asarray(group["latitude"][:], dtype=np.float32)
    if np.array_equal(previous, LATITUDES):
        status = "already_correct"
    elif np.array_equal(previous, LATITUDES[::-1]):
        group["latitude"][:] = LATITUDES
        status = "corrected_coordinate_only"
    else:
        raise ValueError("derived latitude coordinate has an unknown layout")
    group.attrs["latitude_orientation_basis"] = (
        "T850 high-latitude seasonal phase from five train years: first row "
        "austral summer warm, last row boreal summer warm"
    )
    group.attrs["latitude_orientation"] = "ascending_south_to_north"
    result = {
        "status": status,
        "canonical": str(canonical_path),
        "previous_latitude_first_last": [float(previous[0]), float(previous[-1])],
        "corrected_latitude_first_last": [float(LATITUDES[0]), float(LATITUDES[-1])],
        "state_array_modified": False,
        "raw_source_modified": False,
    }
    (canonical_path.parent / "latitude_repair_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_canonical(
    raw_root: str | Path,
    output: str | Path,
    *,
    accept_inferred_time_axis: bool,
    overwrite: bool = False,
) -> dict:
    """Build the immutable-source-derived canonical store and train statistics."""
    if not accept_inferred_time_axis:
        raise ValueError(
            "source files have no time coordinate; pass "
            "accept_inferred_time_axis=True explicitly to acknowledge the time axis "
            "inferred from the equal-length official WeatherBench2 record"
        )
    raw = WB2RawStore(raw_root)
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    group = zarr.open_group(output, mode="w")
    state = group.create_array(
        "state",
        shape=(EXPECTED_RAW_SHAPE[0], 8, 32, 64),
        chunks=(32, 1, 32, 64),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=("time", "channel", "latitude", "longitude"),
    )
    hours = np.arange(EXPECTED_RAW_SHAPE[0], dtype=np.int64) * TIME_STEP_HOURS
    group.create_array("time_hours_since_1959_01_01", data=hours, chunks=(4096,))
    group.create_array("latitude", data=LATITUDES, chunks=(32,))
    group.create_array("longitude", data=LONGITUDES, chunks=(64,))
    group.create_array("level", data=np.asarray(LEVELS, dtype=np.int16), chunks=(2,))
    climatology = group.create_array(
        "climatology",
        shape=(8, 366, 4, 32, 64),
        chunks=(1, 1, 4, 32, 64),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=("channel", "dayofyear", "hour_index", "latitude", "longitude"),
    )
    group.attrs.update(
        {
            "channels": list(CHANNELS),
            "units": list(UNITS),
            "source_manifest": raw.audit,
            "time_units": "hours since 1959-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
            "time_axis_status": "inferred_from_official_WB2_record_not_local_metadata",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    train_stop = official_splits()["train"].stop
    train_times = inferred_times()[:train_stop]
    climate_slots = climatology_slots(train_times)
    sums = np.zeros(8, dtype=np.float64)
    sumsq = np.zeros(8, dtype=np.float64)
    counts = np.zeros(8, dtype=np.int64)
    source_chunk = EXPECTED_RAW_SHAPE[0] // 8  # 11,693, matches source chunking.
    for c in range(8):
        climate_sum = np.zeros((366 * 4, 32, 64), dtype=np.float64)
        climate_count = np.zeros(366 * 4, dtype=np.int32)
        for start in range(0, EXPECTED_RAW_SHAPE[0], source_chunk):
            stop = min(start + source_chunk, EXPECTED_RAW_SHAPE[0])
            values = raw.read_channel(c, slice(start, stop))
            if not np.isfinite(values).all():
                bad = int(values.size - np.isfinite(values).sum())
                raise ValueError(
                    f"{CHANNELS[c]} contains {bad} non-finite values in [{start}:{stop}]"
                )
            state[start:stop, c, :, :] = values
            stat_stop = min(stop, train_stop)
            if start < stat_stop:
                train_values = values[: stat_stop - start].astype(np.float64, copy=False)
                sums[c] += train_values.sum(dtype=np.float64)
                sumsq[c] += np.square(train_values).sum(dtype=np.float64)
                counts[c] += train_values.size
                slots = climate_slots[start:stat_stop]
                for slot in np.unique(slots):
                    selected = train_values[slots == slot]
                    climate_sum[slot] += selected.sum(axis=0, dtype=np.float64)
                    climate_count[slot] += selected.shape[0]
        if np.any(climate_count == 0):
            raise ValueError(
                f"{CHANNELS[c]} climatology has an empty day-of-year/hour group"
            )
        channel_climate = climate_sum / climate_count[:, None, None]
        climatology[c] = channel_climate.reshape(366, 4, 32, 64).astype(np.float32)

    means = sums / counts
    variances = sumsq / counts - means**2
    if np.any(variances <= 0):
        raise ValueError(f"invalid training-set variances: {variances.tolist()}")
    stds = np.sqrt(variances)
    stats = {
        "computed_from": "train: 1959-01-01T00 <= t < 2016-01-01T00",
        "ddof": 0,
        "channels": {
            name: {
                "mean": float(means[i]),
                "std": float(stds[i]),
                "unit": UNITS[i],
                "count": int(counts[i]),
            }
            for i, name in enumerate(CHANNELS)
        },
        "climatology": {
            "source": "train only",
            "grouping": "calendar month-day on leap template x UTC hour (00,06,12,18)",
            "smoothing_days": 0,
            "samples_per_group_min": int(climate_count.min()),
            "samples_per_group_max": int(climate_count.max()),
            "note": "Feb-29 groups have fewer samples; counts are not uniformly 57.",
        },
    }
    stats_path = output.parent / "normalization.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        **raw.audit,
        "canonical": str(output),
        "canonical_shape": list(state.shape),
        "physics": validate_physics(raw, np.linspace(0, 93_543, 12, dtype=int).tolist()),
        "normalization": str(stats_path),
    }
    (output.parent / "build_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def load_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    means = np.asarray([payload["channels"][c]["mean"] for c in CHANNELS], dtype=np.float32)
    stds = np.asarray([payload["channels"][c]["std"] for c in CHANNELS], dtype=np.float32)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("normalization.json contains invalid statistics")
    return means, stds


class WB2SequenceDataset(Dataset):
    """Canonical time-series samples with explicit multi-step targets."""

    def __init__(
        self,
        canonical_path: str | Path,
        split: Split,
        rollout_steps: int,
        normalization_path: str | Path,
        max_samples: int | None = None,
    ):
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        self.path = str(Path(canonical_path).resolve())
        self.split = split
        self.rollout_steps = rollout_steps
        self.means, self.stds = load_normalization(normalization_path)
        available = split.sample_count(rollout_steps)
        self.length = min(available, max_samples) if max_samples else available
        self._array = None

    @property
    def array(self):
        if self._array is None:
            group = zarr.open_group(self.path, mode="r")
            self._array = group["state"]
            if tuple(self._array.shape) != (93_544, 8, 32, 64):
                raise ValueError(f"invalid canonical shape: {self._array.shape}")
        return self._array

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int):
        if item < 0 or item >= self.length:
            raise IndexError(item)
        t0 = self.split.start + item
        x = np.asarray(self.array[t0], dtype=np.float32)
        y = np.asarray(self.array[t0 + 1 : t0 + self.rollout_steps + 1], dtype=np.float32)
        means = self.means[:, None, None]
        stds = self.stds[:, None, None]
        x = (x - means) / stds
        y = (y - means[None, ...]) / stds[None, ...]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), t0

    def __getitems__(self, items: list[int]):
        """Vectorized contiguous reads used by PyTorch's batch fetcher."""
        if not items:
            return []
        if items != list(range(items[0], items[0] + len(items))):
            return [self[item] for item in items]
        if items[0] < 0 or items[-1] >= self.length:
            raise IndexError(items)
        t0 = self.split.start + items[0]
        block = np.asarray(
            self.array[t0 : t0 + len(items) + self.rollout_steps], dtype=np.float32
        )
        means = self.means[None, :, None, None]
        stds = self.stds[None, :, None, None]
        block = (block - means) / stds
        result = []
        for offset, item in enumerate(items):
            x = torch.from_numpy(block[offset].copy())
            y = torch.from_numpy(block[offset + 1 : offset + self.rollout_steps + 1].copy())
            result.append((x, y, self.split.start + item))
        return result


def contiguous_batch_sampler(length: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    """Shuffle contiguous blocks to exploit time-chunk locality without leakage."""
    if length < 1 or batch_size < 1:
        return
    blocks = [list(range(i, min(i + batch_size, length))) for i in range(0, length, batch_size)]
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    yield from blocks


class ContiguousBatchSampler:
    """Epoch-aware shuffled block sampler retaining efficient contiguous Zarr reads."""

    def __init__(self, length: int, batch_size: int, seed: int):
        self.length = length
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return (self.length + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        blocks = list(contiguous_batch_sampler(self.length, self.batch_size, self.seed + self.epoch))
        self.epoch += 1
        yield from blocks
