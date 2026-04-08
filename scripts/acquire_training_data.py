"""Download and save all data needed for training and evaluation.

This script acquires:
    1. Site metadata from HuggingFace openclimatefix/uk_pv
    2. Bad data periods from HuggingFace openclimatefix/uk_pv
    3. 30-minute PV training data (2018-2020) with cleaning applied
    4. Test ground truth (2021) for evaluation
    5. NWP hourly data (2018-2020) from Open-Meteo Archive API

Usage:
    python scripts/acquire_training_data.py
"""

import os
import tempfile
import time
import warnings

import numpy as np
import openmeteo_requests
import pandas as pd
import pvlib
import requests_cache
from huggingface_hub import hf_hub_download
from retry_requests import retry

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO = "openclimatefix/uk_pv"
TRAINING_DIR = "data/training"
PV_CACHE_DIR = "data/pv/parquet_cache_30min"
TRAINING_YEARS = range(2018, 2021)  # 2018, 2019, 2020

NWP_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]

NWP_START = "2018-01-01"
NWP_END = "2020-12-31"
NWP_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
NWP_CHECKPOINT_EVERY = 100
NWP_SLEEP_BETWEEN_CALLS = 0.5
NWP_RETRY_RATE_LIMIT_SLEEP = 120
NWP_RETRY_OTHER_ERROR_SLEEP = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(PV_CACHE_DIR, exist_ok=True)


def _load_30min_month(year: int, month: int) -> pd.DataFrame:
    """Download (or load from cache) one month of 30-minute PV data.

    Args:
        year: Calendar year.
        month: Calendar month (1-12).

    Returns:
        DataFrame with columns ss_id, datetime_GMT, generation_Wh.
    """
    cache_file = os.path.join(PV_CACHE_DIR, f"pv_{year}_{month:02d}.parquet")
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    print(f"  Downloading 30min PV data for {year}-{month:02d} ...")
    hf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"30_minutely/year={year}/month={month:02d}/data.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(hf_path)
    df.to_parquet(cache_file, index=False)
    print(f"    Cached {len(df):,} rows -> {cache_file}")
    return df


# ---------------------------------------------------------------------------
# Step 1: Metadata
# ---------------------------------------------------------------------------


def step1_download_metadata() -> pd.DataFrame:
    """Download site metadata and save to parquet.

    Returns:
        DataFrame with pv_id and site attributes.
    """
    output_file = os.path.join(TRAINING_DIR, "site_metadata.parquet")
    if os.path.exists(output_file):
        print(f"[Step 1] Skipping — {output_file} already exists.")
        return pd.read_parquet(output_file)

    print("[Step 1] Downloading metadata.csv from HuggingFace ...")
    hf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="metadata.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(hf_path)
    df = df.rename(columns={"ss_id": "pv_id"})
    df.to_parquet(output_file, index=False)
    print(f"  Saved {len(df):,} sites -> {output_file}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Bad data
# ---------------------------------------------------------------------------


def step2_download_bad_data() -> pd.DataFrame:
    """Download bad_data.csv and save to parquet.

    Returns:
        DataFrame with pv_id, start_datetime_GMT, end_datetime_GMT, reason.
    """
    output_file = os.path.join(TRAINING_DIR, "bad_data.parquet")
    if os.path.exists(output_file):
        print(f"[Step 2] Skipping — {output_file} already exists.")
        return pd.read_parquet(output_file)

    print("[Step 2] Downloading bad_data.csv from HuggingFace ...")
    hf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="bad_data.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(hf_path)
    df = df.rename(columns={"ss_id": "pv_id"})
    df.to_parquet(output_file, index=False)
    print(f"  Saved {len(df):,} bad-period rows -> {output_file}")
    return df


# ---------------------------------------------------------------------------
# Step 3: 30-minute PV training data (2018-2020)
# ---------------------------------------------------------------------------


def _apply_bad_data_mask(
    pv_df: pd.DataFrame,
    bad_data: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove rows that fall within known bad periods.

    Iterates over bad_data rows (typically a few hundred) rather than over
    pv_ids (~30,000) for better performance. NaN end_datetime_GMT values are
    treated as infinity (remove all records from start onwards).

    Args:
        pv_df: PV DataFrame with pv_id and timestamp columns (UTC-aware).
        bad_data: Bad-period DataFrame with pv_id, start_datetime_GMT,
            end_datetime_GMT columns.

    Returns:
        Tuple of (cleaned DataFrame, number of rows removed).
    """
    bad = bad_data.copy()
    bad["start"] = pd.to_datetime(bad["start_datetime_GMT"], utc=True)
    bad["end"] = pd.to_datetime(bad["end_datetime_GMT"], utc=True, errors="coerce")

    mask_drop = pd.Series(False, index=pv_df.index)
    for _, bad_row in bad.iterrows():
        pid = bad_row["pv_id"]
        start = bad_row["start"]
        end = bad_row["end"]

        pid_mask = pv_df["pv_id"] == pid
        ts_ge_start = pv_df["timestamp"] >= start
        if pd.isna(end):
            in_period = pid_mask & ts_ge_start
        else:
            in_period = pid_mask & ts_ge_start & (pv_df["timestamp"] <= end)
        mask_drop |= in_period

    pv_clean = pv_df[~mask_drop]
    return pv_clean, int(mask_drop.sum())


def _compute_solar_elevation_map(
    unique_combos: pd.DataFrame,
) -> dict[tuple, float]:
    """Compute solar elevation for each unique (lat, lon, date, hour) combo.

    Groups by (latitude_rounded, longitude_rounded) and calls pvlib once per
    location with all its timestamps at once, reducing pvlib calls from
    millions to ~7,032 (one per unique location).

    Args:
        unique_combos: DataFrame with columns latitude_rounded,
            longitude_rounded, date, hour.

    Returns:
        Dict mapping (lat, lon, date, hour) -> solar elevation in degrees.
    """
    elevations: dict[tuple, float] = {}
    total_locs = unique_combos.groupby(
        ["latitude_rounded", "longitude_rounded"]
    ).ngroups
    done = 0
    for (lat, lon), group in unique_combos.groupby(
        ["latitude_rounded", "longitude_rounded"]
    ):
        dt_strings = (
            group["date"].astype(str)
            + " "
            + group["hour"].astype(str).str.zfill(2)
            + ":00:00"
        )
        ts_index = pd.DatetimeIndex(dt_strings, tz="UTC")
        sol = pvlib.solarposition.get_solarposition(ts_index, lat, lon)
        for i, (_, row) in enumerate(group.iterrows()):
            elevations[(lat, lon, row["date"], int(row["hour"]))] = float(
                sol["elevation"].iloc[i]
            )
        done += 1
        if done % 500 == 0:
            print(f"    Solar position progress: {done}/{total_locs} locations ...")
    return elevations


def _clean_month_step1(
    df: pd.DataFrame,
    bad_data: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Apply cleaning step 1 (bad_data filtering) to a single month.

    Args:
        df: Raw monthly PV DataFrame with pv_id, timestamp, generation_Wh.
        bad_data: Bad-period DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, number of rows removed).
    """
    return _apply_bad_data_mask(df, bad_data)


def step3_download_pv_training(
    metadata: pd.DataFrame,
    bad_data: pd.DataFrame,
) -> None:
    """Download 30-minute PV data (2018-2020), clean it, and save to parquet.

    Cleaning steps applied in order:
        1. Remove rows in bad_data periods.
        2. Remove entire days where any row has negative generation_Wh.
        3. Remove entire days where any row has generation_Wh > kWp * 750.
        4. Remove entire days where any nighttime row has power_kw > 0.
        5. Remove entire days where all daytime rows have power_kw == 0.

    Step 1 is applied per-month to limit peak memory usage. Steps 2-5
    require day-level grouping and are applied after loading all monthly
    chunks to avoid issues at month boundaries.

    Args:
        metadata: Site metadata DataFrame (from Step 1).
        bad_data: Bad-period DataFrame (from Step 2).
    """
    output_file = os.path.join(TRAINING_DIR, "pv_30min_2018_2020.parquet")
    if os.path.exists(output_file):
        print(f"[Step 3] Skipping — {output_file} already exists.")
        return

    print("[Step 3] Downloading and cleaning 30-minute PV data (2018-2020) ...")

    # --- Step 1 (bad_data filtering): per-month processing ---
    tmp_dir = tempfile.mkdtemp(prefix="pv_training_tmp_")
    print(f"  Temporary directory for monthly chunks: {tmp_dir}")

    rows_removed: dict[str, int] = {
        "bad_data_periods": 0,
        "negative_generation": 0,
        "exceeds_capacity": 0,
    }
    tmp_files: list[str] = []

    for year in TRAINING_YEARS:
        for month in range(1, 13):
            raw = _load_30min_month(year, month)

            # Standardize column names
            raw = raw.rename(
                columns={
                    "ss_id": "pv_id",
                    "datetime_GMT": "timestamp",
                }
            )
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

            if "generation_Wh" not in raw.columns:
                raise ValueError(
                    "Expected column 'generation_Wh' not found. "
                    f"Columns present: {list(raw.columns)}"
                )

            cleaned, n_bad = _clean_month_step1(raw, bad_data)
            rows_removed["bad_data_periods"] += n_bad

            tmp_path = os.path.join(tmp_dir, f"pv_{year}_{month:02d}_clean.parquet")
            cleaned.to_parquet(tmp_path, index=False)
            tmp_files.append(tmp_path)
            print(
                f"  {year}-{month:02d}: {len(raw):,} raw -> {len(cleaned):,} clean "
                f"(removed bad={n_bad:,})"
            )

    # --- Load all monthly chunks for steps 2-5 ---
    print("  Loading all monthly chunks for steps 2-5 ...")
    pv = pd.concat(
        [pd.read_parquet(f) for f in tmp_files],
        ignore_index=True,
    )
    print(
        f"  After step 1: {pv.shape[0]:,} rows, "
        f"{pv['pv_id'].nunique():,} unique sites"
    )
    print(f"  Date range: {pv['timestamp'].min()} to {pv['timestamp'].max()}")

    # --- Cleaning step 2: days with negative generation_Wh ---
    pv["_date"] = pv["timestamp"].dt.date
    neg_keys = pv.loc[pv["generation_Wh"] < 0, ["pv_id", "_date"]].drop_duplicates()
    if len(neg_keys) > 0:
        idx2 = pv.set_index(["pv_id", "_date"]).index
        mask_neg = idx2.isin(neg_keys.set_index(["pv_id", "_date"]).index)
        rows_removed["negative_generation"] = int(mask_neg.sum())
        pv = pv[~mask_neg].copy()
    print(
        f"  [Clean 2/5] Removed {rows_removed['negative_generation']:,} rows "
        f"from days with negative generation."
    )

    # --- Cleaning step 3: days where generation_Wh > kWp * 750 ---
    kwp_map = metadata.set_index("pv_id")["kWp"]
    pv["_kwp"] = pv["pv_id"].map(kwp_map)
    has_kwp = pv["_kwp"].notna()
    over_keys = pv.loc[
        has_kwp & (pv["generation_Wh"] > pv["_kwp"] * 750),
        ["pv_id", "_date"],
    ].drop_duplicates()
    if len(over_keys) > 0:
        idx3 = pv.set_index(["pv_id", "_date"]).index
        mask_over = idx3.isin(over_keys.set_index(["pv_id", "_date"]).index)
        rows_removed["exceeds_capacity"] = int(mask_over.sum())
        pv = pv[~mask_over].copy()
    pv = pv.drop(columns=["_kwp", "_date"])
    print(
        f"  [Clean 3/5] Removed {rows_removed['exceeds_capacity']:,} rows "
        f"from days exceeding capacity."
    )

    # --- Cleaning step 4: days with nonzero generation at night ---
    print(
        "  [Clean 4/5] Computing solar position for nighttime filter "
        "(this may take several minutes) ..."
    )

    loc_map = (
        metadata[["pv_id", "latitude_rounded", "longitude_rounded"]]
        .drop_duplicates("pv_id")
        .set_index("pv_id")
    )
    pv["latitude_rounded"] = pv["pv_id"].map(loc_map["latitude_rounded"])
    pv["longitude_rounded"] = pv["pv_id"].map(loc_map["longitude_rounded"])
    pv["_date"] = pv["timestamp"].dt.date
    pv["_hour"] = pv["timestamp"].dt.hour
    pv["_power_kw"] = pv["generation_Wh"] * 2 / 1000

    unique_combos = (
        pv[["latitude_rounded", "longitude_rounded", "_date", "_hour"]]
        .rename(columns={"_date": "date", "_hour": "hour"})
        .dropna(subset=["latitude_rounded", "longitude_rounded"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"    Computing elevation for {len(unique_combos):,} unique combos ...")
    elevation_map = _compute_solar_elevation_map(unique_combos)

    pv["_elevation"] = pv.apply(
        lambda r: elevation_map.get(
            (r["latitude_rounded"], r["longitude_rounded"], r["_date"], r["_hour"]),
            np.nan,
        ),
        axis=1,
    )
    pv["_is_night"] = pv["_elevation"] < 0

    # Identify (pv_id, date) pairs with any nighttime power > 0
    night_gen = (
        pv[pv["_is_night"] & (pv["_power_kw"] > 0)][["pv_id", "_date"]]
        .drop_duplicates()
    )
    night_gen["_bad_night_day"] = True
    pv = pv.merge(night_gen, on=["pv_id", "_date"], how="left")
    mask_night_day = pv["_bad_night_day"] == True  # noqa: E712
    n_night_day = int(mask_night_day.sum())
    pv = pv[~mask_night_day].copy()
    rows_removed["night_generation_days"] = n_night_day
    print(f"    Removed {n_night_day:,} rows from days with nighttime generation.")
    pv = pv.drop(columns=["_bad_night_day"], errors="ignore")

    # --- Cleaning step 5: days with zero generation all daytime hours ---
    daytime_pv = pv[~pv["_is_night"]]
    daytime_max = daytime_pv.groupby(["pv_id", "_date"])["_power_kw"].max()
    zero_days = daytime_max[daytime_max == 0].reset_index()[["pv_id", "_date"]]
    zero_days["_bad_zero_day"] = True
    pv = pv.merge(zero_days, on=["pv_id", "_date"], how="left")
    mask_zero_day = pv["_bad_zero_day"] == True  # noqa: E712
    n_zero_day = int(mask_zero_day.sum())
    pv = pv[~mask_zero_day].copy()
    rows_removed["zero_daytime_generation_days"] = n_zero_day
    print(
        f"  [Clean 5/5] Removed {n_zero_day:,} rows from days "
        f"with zero daytime generation."
    )

    # Drop helper columns
    pv = pv.drop(
        columns=[
            "_date",
            "_hour",
            "_power_kw",
            "_elevation",
            "_is_night",
            "_bad_zero_day",
            "latitude_rounded",
            "longitude_rounded",
        ],
        errors="ignore",
    )

    # Convert to kW and drop raw Wh column
    pv["power_kw"] = pv["generation_Wh"] * 2 / 1000
    pv = pv.drop(columns=["generation_Wh"])

    # Final stats
    print("\n  --- Final dataset stats ---")
    print(f"  Shape: {pv.shape}")
    print(f"  Unique sites: {pv['pv_id'].nunique():,}")
    print(f"  Date range: {pv['timestamp'].min()} to {pv['timestamp'].max()}")
    print("  Rows removed per cleaning step:")
    for step_name, count in rows_removed.items():
        print(f"    {step_name}: {count:,}")

    pv.to_parquet(output_file, index=False)
    print(f"\n  Saved -> {output_file}")

    # Clean up temp files
    for f in tmp_files:
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Step 4: Test ground truth
# ---------------------------------------------------------------------------


def step4_build_test_ground_truth() -> None:
    """Build test ground truth from 2021 30-minute PV data.

    Loads testset.csv, fetches 49 hourly values (horizon 0-48h) for each
    (pv_id, timestamp) pair using 30-minute data aggregated to hourly means,
    and saves the result as a parquet file.

    The 30-minute data uses end-of-interval timestamps, so the record at
    HH:30 covers HH:00-HH:30 and the record at HH+1:00 covers HH:30-HH+1:00.
    To compute the hourly average kW for the hour starting at HH:00 we
    average those two slots.
    """
    output_file = os.path.join(TRAINING_DIR, "test_ground_truth.parquet")
    if os.path.exists(output_file):
        print(f"[Step 4] Skipping — {output_file} already exists.")
        return

    print("[Step 4] Building test ground truth ...")

    testset_path = os.path.join("quartz_solar_forecast", "dataset", "testset.csv")
    testset = pd.read_csv(testset_path)
    testset["timestamp"] = pd.to_datetime(testset["timestamp"], utc=True)

    # Determine which year-month parquet files to load
    timestamps = testset["timestamp"]
    end_timestamps = timestamps + pd.Timedelta(hours=48)
    all_ts = pd.concat([timestamps, end_timestamps])
    year_months: set[tuple[int, int]] = set()
    for ts in all_ts:
        year_months.add((ts.year, ts.month))

    print(f"  Loading {len(year_months)} month(s) of 30-min PV data ...")
    all_frames: list[pd.DataFrame] = []
    for year, month in sorted(year_months):
        df = _load_30min_month(year, month)
        all_frames.append(df)

    pv_data = pd.concat(all_frames, ignore_index=True)
    pv_data = pv_data.rename(
        columns={
            "ss_id": "pv_id",
            "datetime_GMT": "timestamp",
        }
    )
    pv_data["timestamp"] = pd.to_datetime(pv_data["timestamp"], utc=True)
    pv_data["power_kw"] = pv_data["generation_Wh"] * 2 / 1000

    # Filter to testset pv_ids for performance
    testset_pv_ids = testset["pv_id"].unique()
    pv_data = pv_data[pv_data["pv_id"].isin(testset_pv_ids)].copy()
    pv_data = pv_data.set_index(["pv_id", "timestamp"]).sort_index()

    records: list[dict] = []
    total = len(testset)
    for idx, row in testset.iterrows():
        if idx % 100 == 0:
            print(f"  Processing sample {idx}/{total} ...")
        pv_id = int(row["pv_id"])
        base_ts = row["timestamp"]

        for horizon in range(49):
            target_ts = base_ts + pd.Timedelta(hours=horizon)
            # datetime_GMT is end-of-interval. For the hour starting at HH:00:
            #   slot1 = HH:30  (covers HH:00 - HH:30)
            #   slot2 = HH+1:00  (covers HH:30 - HH+1:00)
            slot1_ts = target_ts + pd.Timedelta(minutes=30)
            slot2_ts = target_ts + pd.Timedelta(hours=1)

            v1 = np.nan
            v2 = np.nan
            try:
                raw = pv_data.loc[(pv_id, slot1_ts), "power_kw"]
                v1 = float(raw) if np.isscalar(raw) else float(raw.iloc[0])
            except KeyError:
                pass
            try:
                raw = pv_data.loc[(pv_id, slot2_ts), "power_kw"]
                v2 = float(raw) if np.isscalar(raw) else float(raw.iloc[0])
            except KeyError:
                pass

            available = [v for v in (v1, v2) if not np.isnan(v)]
            value = float(np.mean(available)) if available else np.nan

            records.append(
                {
                    "pv_id": pv_id,
                    "timestamp": base_ts,
                    "value": value,
                    "horizon_hour": horizon,
                }
            )

    gt = pd.DataFrame(records)
    gt.to_parquet(output_file, index=False)
    print(f"  Saved {len(gt):,} rows -> {output_file}")


# ---------------------------------------------------------------------------
# Step 5: NWP data from Open-Meteo Archive API
# ---------------------------------------------------------------------------


def _build_openmeteo_client() -> openmeteo_requests.Client:
    """Build an Open-Meteo client with a cache session.

    Retries are handled manually inside the fetch loop, so retry_requests
    is configured with zero automatic retries.

    Returns:
        Configured openmeteo_requests.Client.
    """
    cache_session = requests_cache.CachedSession(
        ".cache_openmeteo_archive",
        expire_after=-1,
    )
    retry_session = retry(cache_session, retries=0)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_nwp_for_location(
    client: openmeteo_requests.Client,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Fetch NWP archive data for one location with infinite retry logic.

    Retries indefinitely: waits NWP_RETRY_RATE_LIMIT_SLEEP seconds on rate
    limit errors (HTTP 429) and NWP_RETRY_OTHER_ERROR_SLEEP seconds for all
    other errors.

    Args:
        client: Open-Meteo client.
        latitude: Site latitude (rounded to ~1 km privacy grid).
        longitude: Site longitude (rounded to ~1 km privacy grid).

    Returns:
        DataFrame with columns: latitude, longitude, time, and one column
        per NWP variable listed in NWP_VARIABLES.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": NWP_START,
        "end_date": NWP_END,
        "hourly": NWP_VARIABLES,
    }

    while True:
        try:
            responses = client.weather_api(NWP_ARCHIVE_URL, params=params)
            break
        except Exception as exc:
            exc_str = str(exc).lower()
            if "rate" in exc_str or "429" in exc_str or "too many" in exc_str:
                print(
                    f"    Rate limit hit for ({latitude}, {longitude}). "
                    f"Sleeping {NWP_RETRY_RATE_LIMIT_SLEEP}s ..."
                )
                time.sleep(NWP_RETRY_RATE_LIMIT_SLEEP)
            else:
                print(
                    f"    Error for ({latitude}, {longitude}): {exc}. "
                    f"Sleeping {NWP_RETRY_OTHER_ERROR_SLEEP}s ..."
                )
                time.sleep(NWP_RETRY_OTHER_ERROR_SLEEP)

    response = responses[0]
    hourly = response.Hourly()

    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data: dict = {"time": time_index}
    for idx, var in enumerate(NWP_VARIABLES):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["latitude"] = latitude
    df["longitude"] = longitude
    return df


def step5_download_nwp(metadata: pd.DataFrame) -> None:
    """Download NWP archive data for all unique site locations (2018-2020).

    Supports resuming via a checkpoint file saved every
    NWP_CHECKPOINT_EVERY locations.

    Args:
        metadata: Site metadata DataFrame containing latitude_rounded and
            longitude_rounded columns.
    """
    output_file = os.path.join(TRAINING_DIR, "nwp_hourly_2018_2020.parquet")
    checkpoint_file = os.path.join(TRAINING_DIR, ".nwp_progress.parquet")

    if os.path.exists(output_file):
        print(f"[Step 5] Skipping — {output_file} already exists.")
        return

    print("[Step 5] Downloading NWP data from Open-Meteo Archive API ...")

    locations = (
        metadata[["latitude_rounded", "longitude_rounded"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    total_locs = len(locations)
    print(f"  {total_locs:,} unique locations to fetch.")

    # Resume support
    done_frames: list[pd.DataFrame] = []
    done_coords: set[tuple[float, float]] = set()
    if os.path.exists(checkpoint_file):
        print(f"  Loading checkpoint from {checkpoint_file} ...")
        progress_df = pd.read_parquet(checkpoint_file)
        done_frames.append(progress_df)
        for _, loc_row in (
            progress_df[["latitude", "longitude"]].drop_duplicates().iterrows()
        ):
            done_coords.add(
                (float(loc_row["latitude"]), float(loc_row["longitude"]))
            )
        print(f"  Resuming: {len(done_coords):,} locations already done.")

    client = _build_openmeteo_client()
    new_frames: list[pd.DataFrame] = []
    processed_since_checkpoint = 0

    for i, loc_row in locations.iterrows():
        lat = float(loc_row["latitude_rounded"])
        lon = float(loc_row["longitude_rounded"])

        if (lat, lon) in done_coords:
            continue

        print(f"  [{i + 1}/{total_locs}] Fetching NWP for lat={lat}, lon={lon} ...")
        loc_df = _fetch_nwp_for_location(client, lat, lon)
        new_frames.append(loc_df)
        done_coords.add((lat, lon))
        processed_since_checkpoint += 1

        time.sleep(NWP_SLEEP_BETWEEN_CALLS)

        if processed_since_checkpoint >= NWP_CHECKPOINT_EVERY:
            print(
                f"  Saving checkpoint ({len(done_coords):,}/{total_locs} done) ..."
            )
            all_so_far = done_frames + new_frames
            pd.concat(all_so_far, ignore_index=True).to_parquet(
                checkpoint_file, index=False
            )
            processed_since_checkpoint = 0

    all_frames = done_frames + new_frames
    if not all_frames:
        print("  No NWP data fetched — check that metadata has valid coordinates.")
        return

    print("  Concatenating all NWP data ...")
    nwp = pd.concat(all_frames, ignore_index=True)
    nwp.to_parquet(output_file, index=False)
    print(f"  Saved {len(nwp):,} rows -> {output_file}")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("  Removed checkpoint file.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all data acquisition steps in sequence."""
    _ensure_dirs()

    metadata = step1_download_metadata()
    bad_data = step2_download_bad_data()
    step3_download_pv_training(metadata, bad_data)
    step4_build_test_ground_truth()
    step5_download_nwp(metadata)

    print("\nAll steps complete.")


if __name__ == "__main__":
    main()
