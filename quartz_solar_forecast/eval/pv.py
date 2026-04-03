import os

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


def get_pv_metadata(testset: pd.DataFrame):
    """Get PV metadata for the sites in the testset from Hugging Face."""
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

    # Load in the dataset
    metadata_df = pd.read_csv(metadata_file)
    metadata_df = metadata_df.rename(columns={"ss_id": "pv_id"})

    # join metadata with testset
    combined_data = testset.merge(metadata_df, on="pv_id", how="left")

    # only keep the columns we need
    combined_data = combined_data[
        ["pv_id", "timestamp", "latitude_rounded", "longitude_rounded", "kWp"]
    ]

    # rename columns
    combined_data = combined_data.rename(
        columns={
            "latitude_rounded": "latitude",
            "longitude_rounded": "longitude",
            "kWp": "capacity",
        }
    )

    # format datetime
    combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])

    return combined_data


def get_pv_truth(testset: pd.DataFrame):
    """Load PV ground truth data from Hugging Face.

    The dataset has been migrated from pv.netcdf to parquet format:
    5_minutely/year=YYYY/month=MM/data.parquet
    """
    print("Loading PV data")

    cache_dir = "data/pv/parquet_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Get unique year-month combinations needed from testset
    # Each sample needs up to 48 hours ahead, so include the next month too
    timestamps = pd.to_datetime(testset["timestamp"])
    end_timestamps = timestamps + pd.Timedelta(hours=48)
    all_timestamps = pd.concat([timestamps, end_timestamps])

    year_months = set()
    for ts in all_timestamps:
        year_months.add((ts.year, ts.month))

    # Download and load parquet files for each year-month
    all_pv_data = []
    for year, month in sorted(year_months):
        cache_file = f"{cache_dir}/pv_{year}_{month:02d}.parquet"

        if not os.path.exists(cache_file):
            print(f"Downloading PV data for {year}-{month:02d}")
            downloaded_path = hf_hub_download(
                repo_id="openclimatefix/uk_pv",
                filename=f"5_minutely/year={year}/month={month:02d}/data.parquet",
                repo_type="dataset",
            )
            # Copy to our cache location
            df = pd.read_parquet(downloaded_path)
            df.to_parquet(cache_file)
        else:
            print(f"Loading cached PV data for {year}-{month:02d}")
            df = pd.read_parquet(cache_file)

        all_pv_data.append(df)
        print(f"  {len(df)} records loaded")

    # Combine all data
    pv_data = pd.concat(all_pv_data, ignore_index=True)
    pv_data = pv_data.rename(columns={
        "ss_id": "pv_id",
        "datetime_GMT": "timestamp",
        "generation_Wh": "generation_wh",
    })
    pv_data["timestamp"] = pd.to_datetime(pv_data["timestamp"], utc=True)
    pv_data["value"] = pv_data["generation_wh"] * 12 / 1000  # Wh per 5min -> kW

    # Filter to only the pv_ids in the testset for performance
    testset_pv_ids = testset["pv_id"].unique()
    pv_data = pv_data[pv_data["pv_id"].isin(testset_pv_ids)]

    # Index by (pv_id, timestamp) for fast lookup
    pv_data = pv_data.set_index(["pv_id", "timestamp"]).sort_index()

    # Build ground truth for each testset sample
    combined_data = []
    for index, row in testset.iterrows():
        if index % 100 == 0:
            print(f"Processing {index} of {len(testset)}")

        pv_id = row["pv_id"]
        base_datetime = pd.to_datetime(row["timestamp"])
        if base_datetime.tz is None:
            base_datetime = base_datetime.tz_localize("UTC")

        for i in range(49):  # 0 to 48 hours
            future_datetime = base_datetime + pd.Timedelta(hours=i)

            try:
                value = pv_data.loc[(pv_id, future_datetime), "value"]
            except KeyError:
                print(f"WARNING: No data for pv_id={pv_id}, timestamp={future_datetime}")
                value = np.nan

            combined_data.append({
                "pv_id": pv_id,
                "timestamp": future_datetime,
                "value": value,
                "horizon_hour": i,
            })

    return pd.DataFrame(combined_data)
