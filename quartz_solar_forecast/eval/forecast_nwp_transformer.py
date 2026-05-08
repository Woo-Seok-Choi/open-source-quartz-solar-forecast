"""Adapter: run ``nwp_transformer`` over the evaluation testset.

The adapter iterates each ``(pv_id, timestamp)`` sample produced by
``get_pv_metadata``, slices the corresponding lookback NWP window out of
the Open-Meteo frame produced by :func:`get_nwp_openmeteo`, and calls the
nwp_transformer inference function. Output is shaped to be drop-in
compatible with :func:`eval.utils.combine_forecast_ground_truth` (columns
``timestamp``, ``pv_id``, ``horizon_hour``, ``power_kw``).

The 1-hour Open-Meteo NWP is upsampled to 30 min via linear interpolation
(``wind_direction_10m`` uses forward-fill because it is a circular
variable; matches training-time preprocessing).
"""

from __future__ import annotations

import pandas as pd
import torch

from quartz_solar_forecast.forecasts.nwp_transformer.dataset import NWP_VARS
from quartz_solar_forecast.forecasts.nwp_transformer.inference import (
    forecast_nwp_transformer,
    load_checkpoint,
)


def run_forecast_nwp_transformer(
    pv_df: pd.DataFrame,
    nwp_df: pd.DataFrame,
    model_path: str,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Run nwp_transformer on each testset sample.

    Args:
        pv_df: Columns ``pv_id``, ``timestamp``, ``latitude``, ``longitude``,
            ``capacity`` (output of :func:`eval.pv.get_pv_metadata`).
        nwp_df: Hourly Open-Meteo NWP block (output of
            :func:`eval.nwp_openmeteo.get_nwp_openmeteo`). Must contain
            ``pv_id``, ``timestamp``, ``time``, and the 14 NWP variables.
        model_path: Path to a ``best.pt`` checkpoint produced by
            ``forecasts/nwp_transformer/train.py``.
        device: Torch device. Defaults to CUDA if available.

    Returns:
        DataFrame with columns ``timestamp``, ``pv_id``, ``horizon_hour``,
        ``power_kw``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(model_path, device=device)
    seq_len = model.seq_len

    pv_df = pv_df.copy()
    pv_df["timestamp"] = pd.to_datetime(pv_df["timestamp"], utc=True)
    nwp_df = nwp_df.copy()
    nwp_df["timestamp"] = pd.to_datetime(nwp_df["timestamp"], utc=True)
    nwp_df["time"] = pd.to_datetime(nwp_df["time"], utc=True)
    nwp_groups = nwp_df.groupby(["pv_id", "timestamp"])

    n_skipped = 0
    all_preds: list[pd.DataFrame] = []
    for i, row in pv_df.iterrows():
        pv_id = int(row["pv_id"])
        ts = pd.Timestamp(row["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        kwp = float(row["capacity"])

        try:
            block = nwp_groups.get_group((pv_id, ts))
        except KeyError:
            print(f"  [{i + 1}/{len(pv_df)}] no NWP for ({pv_id}, {ts})")
            n_skipped += 1
            continue

        block = block.sort_values("time").set_index("time")

        # Build a 30-min lookback grid anchored exactly at ``ts`` (regardless
        # of whether ``ts`` lands on the hour, the half-hour, or a 15-min
        # offset). Time-interpolate the hourly NWP onto this grid by first
        # reindexing onto the union of (hourly index, target index) and then
        # restricting back to the target index.
        target_idx = pd.date_range(end=ts, periods=seq_len, freq="30min")
        # Drop columns we will replace, to avoid linear-interpolating a
        # circular variable. Linear interp on the remaining 13 vars only.
        linear_cols = [v for v in NWP_VARS if v != "wind_direction_10m"]
        combined_idx = block.index.union(target_idx).sort_values()
        nwp_dense = (
            block[linear_cols].reindex(combined_idx).interpolate(method="time")
        )
        nwp_30 = nwp_dense.reindex(target_idx)
        nwp_30["wind_direction_10m"] = block["wind_direction_10m"].reindex(
            target_idx, method="ffill"
        )
        nwp_30 = nwp_30[NWP_VARS]  # restore canonical column order
        if not nwp_30.notna().all().all():
            print(
                f"  [{i + 1}/{len(pv_df)}] NaN after interpolation for "
                f"({pv_id}, {ts}); skipping"
            )
            n_skipped += 1
            continue

        pred_df = forecast_nwp_transformer(
            nwp_window=nwp_30[NWP_VARS],
            ts=ts,
            kwp=kwp,
            model=model,
            device=device,
        )
        pred_df = pred_df.rename(columns={"forecast_kw": "power_kw"})
        pred_df["pv_id"] = pv_id
        all_preds.append(pred_df)

        if (i + 1) % 100 == 0:
            print(f"  forecast {i + 1}/{len(pv_df)}")

    if not all_preds:
        raise RuntimeError("No forecasts produced for any testset sample.")
    print(f"Forecast complete: {len(all_preds)} samples, {n_skipped} skipped.")
    return pd.concat(all_preds, ignore_index=True)
