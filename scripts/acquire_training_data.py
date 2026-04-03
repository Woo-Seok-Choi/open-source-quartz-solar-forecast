"""Download and save all data needed for training and evaluation.

Steps:
1. Metadata (from HuggingFace)
2. Ground truth for testset (from HuggingFace parquet)
3. Training PV data 2018-2020 (from HuggingFace parquet)
4. Training NWP data 2018-2020 (from Open-Meteo archive API)

Usage:
    python scripts/acquire_training_data.py
"""

import os
import sys
import time

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from huggingface_hub import hf_hub_download
from retry_requests import retry

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quartz_solar_forecast.eval.pv import get_pv_metadata, get_pv_truth

TRAINING_DIR = "data/training"
PV_CACHE_DIR = "data/pv/parquet_cache"


def step1_metadata():
    """Download and save full metadata with orientation/tilt."""
    output_file = f"{TRAINING_DIR}/site_metadata.parquet"
    if os.path.exists(output_file):
        print(f"[Step 1] Already exists: {output_file}")
        return pd.read_parquet(output_file)

    print("[Step 1] Downloading metadata...")
    cache_dir = "data/pv"
    metadata_file = f"{cache_dir}/metadata.csv"

    if not os.path.exists(metadata_file):
        os.makedirs(cache_dir, exist_ok=True)
        hf_hub_download(
            repo_id="openclimatefix/uk_pv",
            filename="metadata.csv",
            repo_type="dataset",
            local_dir=cache_dir,
        )

    meta = pd.read_csv(metadata_file)
    meta = meta.rename(columns={"ss_id": "pv_id"})
    meta.to_parquet(output_file, index=False)
    print(f"[Step 1] Saved: {output_file} ({meta.shape})")
    return meta


def step2_ground_truth():
    """Download and save ground truth for the testset."""
    output_file = f"{TRAINING_DIR}/test_ground_truth.parquet"
    if os.path.exists(output_file):
        print(f"[Step 2] Already exists: {output_file}")
        return

    print("[Step 2] Building ground truth for testset...")
    testset = pd.read_csv("quartz_solar_forecast/dataset/testset.csv")
    truth = get_pv_truth(testset)
    truth.to_parquet(output_file, index=False)
    print(f"[Step 2] Saved: {output_file} ({truth.shape})")
    nan_count = truth["value"].isna().sum()
    print(f"[Step 2] NaN count: {nan_count} / {len(truth)}")


def step3_training_pv():
    """Download 2018-2020 PV data from HuggingFace parquet files."""
    output_file = f"{TRAINING_DIR}/pv_5min_2018_2020.parquet"
    if os.path.exists(output_file):
        print(f"[Step 3] Already exists: {output_file}")
        return

    print("[Step 3] Downloading training PV data (2018-2020)...")
    os.makedirs(PV_CACHE_DIR, exist_ok=True)

    all_data = []
    for year in range(2018, 2021):
        for month in range(1, 13):
            cache_file = f"{PV_CACHE_DIR}/pv_{year}_{month:02d}.parquet"

            if not os.path.exists(cache_file):
                print(f"  Downloading {year}-{month:02d}...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id="openclimatefix/uk_pv",
                        filename=f"5_minutely/year={year}/month={month:02d}/data.parquet",
                        repo_type="dataset",
                    )
                    df = pd.read_parquet(downloaded_path)
                    df.to_parquet(cache_file)
                except Exception as e:
                    print(f"  ERROR downloading {year}-{month:02d}: {e}")
                    continue
            else:
                print(f"  Cached: {year}-{month:02d}")
                df = pd.read_parquet(cache_file)

            print(f"    {len(df)} records")
            all_data.append(df)

    print("  Combining all PV data...")
    pv_data = pd.concat(all_data, ignore_index=True)
    pv_data = pv_data.rename(columns={
        "ss_id": "pv_id",
        "datetime_GMT": "timestamp",
        "generation_Wh": "generation_wh",
    })
    pv_data["timestamp"] = pd.to_datetime(pv_data["timestamp"], utc=True)
    pv_data["power_kw"] = pv_data["generation_wh"] * 12 / 1000  # Wh per 5min -> kW

    # Drop raw column, keep converted
    pv_data = pv_data.drop(columns=["generation_wh"])

    pv_data.to_parquet(output_file, index=False)
    print(f"[Step 3] Saved: {output_file} ({pv_data.shape})")
    print(f"[Step 3] Unique sites: {pv_data['pv_id'].nunique()}")
    print(f"[Step 3] Date range: {pv_data['timestamp'].min()} ~ {pv_data['timestamp'].max()}")


def step4_training_nwp(metadata: pd.DataFrame):
    """Download 2018-2020 NWP data from Open-Meteo archive API."""
    output_file = f"{TRAINING_DIR}/nwp_hourly_2018_2020.parquet"
    if os.path.exists(output_file):
        print(f"[Step 4] Already exists: {output_file}")
        return

    print("[Step 4] Downloading training NWP data (2018-2020)...")

    # Get unique locations
    locations = metadata.groupby(
        ["latitude_rounded", "longitude_rounded"]
    ).size().reset_index()[["latitude_rounded", "longitude_rounded"]]
    print(f"  Unique locations: {len(locations)}")

    # Setup Open-Meteo client with cache
    cache_session = requests_cache.CachedSession(
        f"{TRAINING_DIR}/.nwp_cache", expire_after=-1
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.5)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    variables = [
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

    # Progress tracking for resume support
    progress_file = f"{TRAINING_DIR}/.nwp_progress.parquet"
    if os.path.exists(progress_file):
        print("  Resuming from previous progress...")
        all_nwp = pd.read_parquet(progress_file)
        done_locations = set(
            zip(all_nwp["latitude"].unique(), all_nwp["longitude"].unique())
        )
        # Deduplicate - some might have partial data
        done_lats = all_nwp.groupby(["latitude", "longitude"]).size().reset_index()
        done_locations = set(zip(done_lats["latitude"], done_lats["longitude"]))
        all_nwp_list = [all_nwp]
    else:
        done_locations = set()
        all_nwp_list = []

    total = len(locations)
    skipped = 0
    errors = 0
    api_calls = 0

    for idx, row in locations.iterrows():
        lat = row["latitude_rounded"]
        lon = row["longitude_rounded"]

        if (lat, lon) in done_locations:
            skipped += 1
            continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": "2018-01-01",
                    "end_date": "2020-12-31",
                    "hourly": variables,
                }
                response = openmeteo.weather_api(
                    "https://archive-api.open-meteo.com/v1/archive", params=params
                )
                hourly = response[0].Hourly()

                times = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                )

                hourly_data = {"timestamp": times}
                for var_idx, var_name in enumerate(variables):
                    hourly_data[var_name] = hourly.Variables(var_idx).ValuesAsNumpy()

                df = pd.DataFrame(hourly_data)
                df["latitude"] = lat
                df["longitude"] = lon
                all_nwp_list.append(df)
                api_calls += 1

                # Rate limiting: stay well under 600/min
                time.sleep(1.0)

                # Progress update and save every 100 locations
                done_count = api_calls + skipped
                if api_calls % 100 == 0:
                    print(
                        f"  Progress: {done_count}/{total} "
                        f"(API calls: {api_calls}, skipped: {skipped}, errors: {errors})"
                    )
                if api_calls % 500 == 0:
                    print("  Saving progress checkpoint...")
                    progress_df = pd.concat(all_nwp_list, ignore_index=True)
                    progress_df.to_parquet(progress_file, index=False)

                break  # Success, exit retry loop

            except Exception as e:
                is_rate_limit = "limit" in str(e).lower() or "429" in str(e)
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)
                    print(f"  Rate limited at ({lat}, {lon}), waiting {wait_time}s (attempt {attempt + 1})...")
                    time.sleep(wait_time)
                    continue
                errors += 1
                if errors <= 10:
                    print(f"  ERROR at ({lat}, {lon}): {e}")
                elif errors == 11:
                    print("  (suppressing further error messages)")
                break

    print(f"  Done! API calls: {api_calls}, skipped: {skipped}, errors: {errors}")
    print("  Combining all NWP data...")
    nwp_data = pd.concat(all_nwp_list, ignore_index=True)
    nwp_data.to_parquet(output_file, index=False)
    print(f"[Step 4] Saved: {output_file} ({nwp_data.shape})")

    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)


def main():
    os.makedirs(TRAINING_DIR, exist_ok=True)

    print("=" * 60)
    print("Data Acquisition Pipeline")
    print("=" * 60)

    # Step 1
    metadata = step1_metadata()

    # Step 2
    step2_ground_truth()

    # Step 3
    step3_training_pv()

    # Step 4
    step4_training_nwp(metadata)

    print()
    print("=" * 60)
    print("All done! Saved files:")
    for f in os.listdir(TRAINING_DIR):
        if f.endswith(".parquet"):
            size_mb = os.path.getsize(f"{TRAINING_DIR}/{f}") / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
