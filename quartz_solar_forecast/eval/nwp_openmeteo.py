"""Open-Meteo Archive NWP fetcher for ``nwp_transformer`` evaluation.

Mirrors the variable set and API source used by
``scripts/acquire_training_data.py`` so the train and eval NWP distributions
match. The companion ICON-based ``eval/nwp.py`` is unchanged and continues
to serve the v1/v2 evaluation paths.

Output convention:
    Returns one DataFrame with all (pv_id, timestamp) lookback NWP blocks
    concatenated. Each (pv_id, timestamp) sample contributes a 49-row block
    (hourly, ``[t-48h, t]`` inclusive) covering the lookback window of the
    nwp_transformer model.

Cache:
    Per-(lat, lon) hourly NWP for the date range needed by the testset is
    saved to ``data/nwp_openmeteo_eval/`` as parquet. Re-runs reuse cache.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

NWP_VARIABLES: list[str] = [
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

ARCHIVE_URL = "https://customer-archive-api.open-meteo.com/v1/archive"
LOOKBACK_HOURS = 48
SLEEP_BETWEEN_CALLS = 0.5
RATE_LIMIT_SLEEP = 120.0
OTHER_ERROR_SLEEP = 30.0
CACHE_DIR = Path("data/nwp_openmeteo_eval")


def _build_client() -> openmeteo_requests.Client:
    """Open-Meteo client with HTTP-level disk cache (long-lived)."""
    cache_session = requests_cache.CachedSession(
        ".cache_openmeteo_eval", expire_after=-1,
    )
    retry_session = retry(cache_session, retries=0)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_one_location(
    client: openmeteo_requests.Client,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch hourly NWP for one ``(lat, lon)`` over ``[start_date, end_date]``.

    Both date arguments are ``YYYY-MM-DD`` UTC strings. Retries forever on
    rate-limit and other transient errors (matches
    ``scripts/acquire_training_data.py`` behavior).
    """
    params: dict[str, object] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": NWP_VARIABLES,
    }
    api_key = os.environ.get("OPEN_METEO_API_KEY", "")
    if api_key:
        params["apikey"] = api_key

    while True:
        try:
            responses = client.weather_api(ARCHIVE_URL, params=params)
            break
        except Exception as exc:
            text = str(exc).lower()
            if "rate" in text or "429" in text or "too many" in text:
                print(
                    f"    Rate limit at ({latitude}, {longitude}); "
                    f"sleeping {RATE_LIMIT_SLEEP:.0f}s ..."
                )
                time.sleep(RATE_LIMIT_SLEEP)
            else:
                print(
                    f"    Error at ({latitude}, {longitude}): {exc}; "
                    f"sleeping {OTHER_ERROR_SLEEP:.0f}s ..."
                )
                time.sleep(OTHER_ERROR_SLEEP)

    response = responses[0]
    hourly = response.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    df = pd.DataFrame({"time": times})
    for idx, var in enumerate(NWP_VARIABLES):
        df[var] = hourly.Variables(idx).ValuesAsNumpy()
    df["latitude"] = latitude
    df["longitude"] = longitude
    return df


def get_nwp_openmeteo(time_locations: pd.DataFrame) -> pd.DataFrame:
    """Fetch lookback NWP for each ``(pv_id, timestamp)`` row.

    For each unique ``(latitude, longitude)`` in ``time_locations``, hourly
    NWP is fetched once over a date range that covers every timestamp's
    ``[t - 48h, t]`` lookback window. The function then slices the
    per-sample lookback block out of the cached frame.

    Args:
        time_locations: DataFrame with columns ``pv_id``, ``timestamp``,
            ``latitude``, ``longitude``. ``timestamp`` may be tz-aware or
            tz-naive (treated as UTC if naive).

    Returns:
        Concatenated DataFrame with columns ``pv_id``, ``timestamp``,
        ``time``, ``latitude``, ``longitude``, and one column per variable
        in :data:`NWP_VARIABLES`. Each sample contributes 49 rows (the
        lookback hourly window from ``t-48h`` to ``t``, inclusive).
    """
    if "OPEN_METEO_API_KEY" not in os.environ:
        # Try .env file via python-dotenv if installed
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Normalize timestamps to tz-aware UTC.
    locs = time_locations.copy()
    locs["timestamp"] = pd.to_datetime(locs["timestamp"], utc=True)

    # Per-(lat, lon) date range: cover the union of every sample's lookback.
    range_df = (
        locs.groupby(["latitude", "longitude"], as_index=False)
        .agg(t_min=("timestamp", "min"), t_max=("timestamp", "max"))
    )

    client = _build_client()
    fetched: dict[tuple[float, float], pd.DataFrame] = {}

    for i, row in range_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        start_dt = (row["t_min"] - pd.Timedelta(hours=LOOKBACK_HOURS + 1)).floor("D")
        end_dt = row["t_max"].ceil("D")
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        cache_file = CACHE_DIR / (
            f"lat={lat}_lon={lon}_{start_date}_{end_date}.parquet"
        )
        if cache_file.exists():
            print(f"[{i + 1}/{len(range_df)}] cached: {cache_file.name}")
            df = pd.read_parquet(cache_file)
        else:
            print(
                f"[{i + 1}/{len(range_df)}] fetching {lat},{lon} "
                f"{start_date}..{end_date}"
            )
            df = _fetch_one_location(client, lat, lon, start_date, end_date)
            df.to_parquet(cache_file, index=False)
            time.sleep(SLEEP_BETWEEN_CALLS)

        fetched[(lat, lon)] = df

    # Slice each sample's lookback window out of the cached frame.
    # Range is widened by 1 h on both sides so that the 30-min upsampler
    # in the adapter has bracketing hourly anchors at every 30-min target
    # step (especially for testset ``ts`` on 15-min boundaries).
    out_blocks: list[pd.DataFrame] = []
    for _, row in locs.iterrows():
        coord = (float(row["latitude"]), float(row["longitude"]))
        ts: pd.Timestamp = row["timestamp"]
        full = fetched[coord]
        lookback_start = ts - pd.Timedelta(hours=LOOKBACK_HOURS + 1)
        lookback_end = ts + pd.Timedelta(hours=1)
        mask = (full["time"] >= lookback_start) & (full["time"] <= lookback_end)
        block = full.loc[mask].copy()
        block["pv_id"] = row["pv_id"]
        block["timestamp"] = ts
        out_blocks.append(block)

    return pd.concat(out_blocks, ignore_index=True)
