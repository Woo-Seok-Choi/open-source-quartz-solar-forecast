"""Sliding-window PyTorch Dataset for nwp_transformer training.

Builds per-site time-aligned arrays from the three Quartz training parquet
files (PV 30-min, NWP hourly, site metadata), excludes the 50 testset sites,
and yields ``(x_nwp, kwp, y_power)`` windows.

Data conventions (verified — see project root CLAUDE.md):
    - NWP ``(latitude, longitude)`` rows match site ``(latitude_rounded,
      longitude_rounded)`` exactly. No nearest-neighbor lookup needed.
    - PV is already 30-min aligned. NWP is hourly and is upsampled to 30-min
      via linear interpolation, **except** ``wind_direction_10m`` which is a
      circular variable and uses forward-fill instead.
    - All timestamps are normalized to tz-naive UTC.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NWP_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]
N_NWP_VARS = len(NWP_VARS)


def _to_naive_utc(ts: pd.Series) -> pd.Series:
    """Convert tz-aware timestamps to tz-naive UTC; pass tz-naive through."""
    if pd.api.types.is_datetime64_any_dtype(ts.dtype) and ts.dt.tz is not None:
        return ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts


def load_site_data(
    data_dir: str | os.PathLike[str],
    testset_csv: str | os.PathLike[str],
    num_sites: int,
    seed: int = 42,
    min_valid_steps: int | None = None,
) -> dict[int, tuple[np.ndarray, float]]:
    """Public helper: select sites and load PV+NWP arrays once.

    The returned mapping can be passed via ``site_data=`` to
    :class:`QuartzWindowDataset` so train/val/test splits share memory.

    Args:
        data_dir: Path containing the three training parquet files.
        testset_csv: Path to ``dataset/testset.csv`` (50 pv_ids excluded).
        num_sites: Number of sites to sample (after testset exclusion).
        seed: RNG seed for site sampling.
        min_valid_steps: Minimum non-null PV samples a candidate site must
            have. Defaults to ``192`` (one window) if ``None``.
    """
    if min_valid_steps is None:
        min_valid_steps = 192
    sites = _select_sites(
        Path(data_dir), Path(testset_csv), num_sites, seed, min_valid_steps,
    )
    return _load_site_arrays(Path(data_dir), sites)


def _select_sites(
    data_dir: Path,
    testset_csv: Path,
    num_sites: int,
    seed: int,
    min_valid_steps: int,
) -> pd.DataFrame:
    """Pick ``num_sites`` sites with sufficient valid PV measurements.

    Excludes the 50 testset pv_ids and any pv_id whose non-null power_kw
    count is below ``min_valid_steps``. Sampling is deterministic for a
    given ``seed``.

    Returns:
        DataFrame with columns ``pv_id``, ``latitude_rounded``,
        ``longitude_rounded``, ``kWp``.
    """
    metadata = pd.read_parquet(data_dir / "site_metadata.parquet")
    testset = pd.read_csv(testset_csv)
    testset_ids = set(testset["pv_id"].unique())

    candidates = metadata[~metadata["pv_id"].isin(testset_ids)].copy()
    candidates = candidates.dropna(
        subset=["latitude_rounded", "longitude_rounded", "kWp"]
    )

    pv_counts = pd.read_parquet(
        data_dir / "pv_30min_2018_2020.parquet",
        columns=["pv_id", "power_kw"],
    )
    valid = (
        pv_counts.dropna(subset=["power_kw"]).groupby("pv_id").size().rename("n_valid")
    )
    candidates = candidates.merge(valid, left_on="pv_id", right_index=True, how="inner")
    candidates = candidates[candidates["n_valid"] >= min_valid_steps]

    if len(candidates) < num_sites:
        raise ValueError(
            f"Only {len(candidates)} eligible sites (need {num_sites}). "
            f"Lower min_valid_steps or num_sites."
        )

    sampled = candidates.sample(n=num_sites, random_state=seed).reset_index(drop=True)
    return sampled[["pv_id", "latitude_rounded", "longitude_rounded", "kWp"]]


def _load_site_arrays(
    data_dir: Path,
    sites: pd.DataFrame,
) -> dict[int, tuple[np.ndarray, float]]:
    """Load PV + NWP, align to a 30-min index per site.

    Returns:
        ``{pv_id: (xy_array, kwp)}`` where ``xy_array`` has shape
        ``[T_site, N_NWP_VARS + 1]`` with NWP variables in the first
        ``N_NWP_VARS`` columns and ``power_kw`` in the last column.
    """
    site_ids = sites["pv_id"].tolist()

    pv = pd.read_parquet(
        data_dir / "pv_30min_2018_2020.parquet",
        columns=["pv_id", "timestamp", "power_kw"],
        filters=[("pv_id", "in", site_ids)],
    )
    pv["timestamp"] = _to_naive_utc(pv["timestamp"])

    # Build the exact (lat, lon) pair set so we don't pull the Cartesian
    # product of unique lats × unique lons. Use a range pre-filter (cheap
    # I/O) to avoid any float-equality issue in arrow predicate pushdown,
    # then reduce to exact pairs via inner merge.
    pair_df = (
        sites[["latitude_rounded", "longitude_rounded"]]
        .drop_duplicates()
        .rename(columns={"latitude_rounded": "latitude",
                         "longitude_rounded": "longitude"})
        .reset_index(drop=True)
    )
    lat_min = float(pair_df["latitude"].min())
    lat_max = float(pair_df["latitude"].max())
    lon_min = float(pair_df["longitude"].min())
    lon_max = float(pair_df["longitude"].max())

    nwp_cols = ["time", "latitude", "longitude", *NWP_VARS]
    nwp = pd.read_parquet(
        data_dir / "nwp_hourly_2018_2020.parquet",
        columns=nwp_cols,
        filters=[
            ("latitude", ">=", lat_min),
            ("latitude", "<=", lat_max),
            ("longitude", ">=", lon_min),
            ("longitude", "<=", lon_max),
        ],
    )
    nwp = nwp.merge(pair_df, on=["latitude", "longitude"], how="inner")
    nwp["time"] = _to_naive_utc(nwp["time"])

    nwp_by_coord: dict[tuple[float, float], pd.DataFrame] = {}
    for (lat, lon), grp in nwp.groupby(["latitude", "longitude"], sort=False):
        df = grp.set_index("time")[NWP_VARS].sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df_30 = df.resample("30min").interpolate(method="linear")
        # wind_direction_10m is circular; linear interp is invalid (e.g. 350°
        # and 10° interpolate to 180° instead of 0°). Reindex on the 30-min
        # grid and forward-fill so each HH:30 inherits the prior hour's
        # value with explicit alignment to df_30.index.
        df_30["wind_direction_10m"] = (
            df["wind_direction_10m"].reindex(df_30.index).ffill()
        )
        nwp_by_coord[(float(lat), float(lon))] = df_30

    pv_by_id: dict[int, pd.Series] = {}
    for pv_id, grp in pv.groupby("pv_id", sort=False):
        s = grp.set_index("timestamp")["power_kw"].sort_index()
        s = s[~s.index.duplicated(keep="first")]
        pv_by_id[int(pv_id)] = s

    out: dict[int, tuple[np.ndarray, float]] = {}
    for row in sites.itertuples(index=False):
        pv_id = int(row.pv_id)
        coord = (float(row.latitude_rounded), float(row.longitude_rounded))
        if pv_id not in pv_by_id or coord not in nwp_by_coord:
            continue

        pv_s = pv_by_id[pv_id]
        nwp_df = nwp_by_coord[coord]
        common_idx = pv_s.index.intersection(nwp_df.index).sort_values()
        if len(common_idx) == 0:
            continue

        nwp_arr = nwp_df.reindex(common_idx)[NWP_VARS].to_numpy(dtype=np.float32)
        pv_arr = pv_s.reindex(common_idx).to_numpy(dtype=np.float32)
        xy = np.concatenate([nwp_arr, pv_arr[:, None]], axis=1)
        out[pv_id] = (xy, float(row.kWp))

    return out


class QuartzWindowDataset(Dataset):
    """Sliding-window dataset for nwp_transformer.

    Each ``__getitem__`` returns ``(x_nwp, kwp, y_power)``:

    - ``x_nwp``: ``[seq_len, N_NWP_VARS]`` float32
    - ``kwp``: scalar float32 (per-site capacity in kW)
    - ``y_power``: ``[pred_len]`` float32 (raw kW)

    Args:
        data_dir: Path containing the three training parquet files.
        testset_csv: Path to ``dataset/testset.csv`` (50 pv_ids to exclude).
        split: ``"train"``, ``"val"`` or ``"test"``.
        num_sites: Number of sites to sample after exclusion.
        seq_len: Lookback window length in 30-min steps.
        pred_len: Forecast horizon in 30-min steps.
        stride: Step between consecutive window starts. ``1`` enumerates
            every 30-min step.
        seed: RNG seed for site sampling.
        train_frac, val_frac: Time-axis split fractions per site.
        min_valid_density: Minimum non-null PV samples a candidate site
            must have to be eligible (default ``seq_len + pred_len``).
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str] | None = None,
        testset_csv: str | os.PathLike[str] | None = None,
        split: str = "train",
        num_sites: int = 100,
        seq_len: int = 96,
        pred_len: int = 96,
        stride: int = 1,
        seed: int = 42,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        min_valid_density: int | None = None,
        site_data: dict[int, tuple[np.ndarray, float]] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if not 0 < train_frac < 1 or not 0 < val_frac < 1 - train_frac:
            raise ValueError("train_frac/val_frac out of range")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.split = split
        window_len = seq_len + pred_len
        if min_valid_density is None:
            min_valid_density = window_len

        # Reuse pre-loaded site_data when provided so train/val/test splits
        # share the heavy NWP+PV arrays without doubling RAM.
        if site_data is None:
            if data_dir is None or testset_csv is None:
                raise ValueError(
                    "Either pass site_data, or provide data_dir + testset_csv "
                    "to load it here."
                )
            sites = _select_sites(
                Path(data_dir), Path(testset_csv),
                num_sites, seed, min_valid_density,
            )
            site_data = _load_site_arrays(Path(data_dir), sites)

        # Pack into parallel lists indexed by site_idx.
        # Per-site time split: train [0, n_train); val [n_train, n_train+n_val);
        # test [n_train+n_val, T). Val and test windows let their lookback x
        # extend back into the prior segment by up to seq_len steps so that
        # boundary windows are usable. Targets y always remain inside the
        # split's segment, preventing label leakage.
        self._site_xy: list[np.ndarray] = []
        self._site_kwp: list[float] = []
        self._windows: list[tuple[int, int]] = []

        n_dropped_sites = 0
        n_dropped_nan = 0
        n_total_candidate_windows = 0

        for pv_id, (xy, kwp) in site_data.items():
            t = xy.shape[0]
            n_train = int(t * train_frac)
            n_val = int(t * val_frac)
            # Per-site time split. Val/test starts shifted by -seq_len so
            # boundary windows can use lookback inside the previous segment.
            if split == "train":
                lo, hi = 0, n_train
            elif split == "val":
                lo, hi = n_train - seq_len, n_train + n_val
            else:
                lo, hi = n_train + n_val - seq_len, t
            lo = max(lo, 0)
            last_start = hi - window_len
            if last_start < lo:
                n_dropped_sites += 1
                continue

            site_idx = len(self._site_xy)
            self._site_xy.append(xy)
            self._site_kwp.append(kwp)

            for ws in range(lo, last_start + 1, stride):
                n_total_candidate_windows += 1
                # NaN filter: drop any window with NaN in NWP-input or PV-target.
                x_block = xy[ws : ws + seq_len, :N_NWP_VARS]
                y_block = xy[ws + seq_len : ws + window_len, N_NWP_VARS]
                if not (np.isfinite(x_block).all() and np.isfinite(y_block).all()):
                    n_dropped_nan += 1
                    continue
                self._windows.append((site_idx, ws))

        if not self._windows:
            raise ValueError(
                f"No valid windows for split={split!r} "
                f"(candidates: {n_total_candidate_windows}, dropped sites: "
                f"{n_dropped_sites}, dropped nan: {n_dropped_nan})."
            )

        self.stats = {
            "n_sites": len(self._site_xy),
            "n_dropped_sites": n_dropped_sites,
            "n_candidate_windows": n_total_candidate_windows,
            "n_dropped_nan": n_dropped_nan,
            "n_valid_windows": len(self._windows),
        }

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        site_idx, ws = self._windows[idx]
        xy = self._site_xy[site_idx]
        kwp = self._site_kwp[site_idx]

        x = xy[ws : ws + self.seq_len, :N_NWP_VARS]
        y = xy[ws + self.seq_len : ws + self.seq_len + self.pred_len, N_NWP_VARS]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor(kwp, dtype=torch.float32),
            torch.from_numpy(np.ascontiguousarray(y)),
        )
