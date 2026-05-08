"""Inference API for nwp_transformer.

Entry points:
    - ``load_checkpoint(path)``: hydrate an ``NWPTransformer`` from a
      ``best.pt`` produced by ``train.py``.
    - ``forecast_nwp_transformer(...)``: predict 48 h PV power and return
      an hourly-aligned DataFrame compatible with ``eval/utils.py``'s
      merge on ``["timestamp", "pv_id", "horizon_hour"]``.

Output convention:
    - Columns: ``timestamp``, ``horizon_hour``, ``forecast_kw``.
    - The 30-min model output is downsampled to hourly by taking
      on-the-hour samples ``[0, 2, 4, ..., 94]`` of the 96-step output,
      giving 48 rows for horizon hours ``0..47``. ``run_evaluation.py``
      builds 49-row ground truth for ``0..48``; horizon 48 will simply
      be missing from the predictions and the merge keeps the row with
      NaN forecast_kw.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch

from quartz_solar_forecast.forecasts.nwp_transformer.dataset import NWP_VARS
from quartz_solar_forecast.forecasts.nwp_transformer.model import NWPTransformer

NWPInput = Union[np.ndarray, torch.Tensor, pd.DataFrame]


def load_checkpoint(
    model_path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
) -> NWPTransformer:
    """Load a ``train.py`` checkpoint and return the model in eval mode."""
    ckpt = torch.load(Path(model_path), map_location=device, weights_only=False)
    model_config = ckpt.get("model_config")
    if model_config is None:
        raise ValueError(
            f"Checkpoint at {model_path} is missing 'model_config'. "
            "Re-train with the current train.py to regenerate it."
        )
    model = NWPTransformer(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _coerce_nwp(nwp_window: NWPInput) -> np.ndarray:
    """Validate and coerce NWP input to a ``[seq_len, 14]`` float32 array."""
    if isinstance(nwp_window, pd.DataFrame):
        missing = [c for c in NWP_VARS if c not in nwp_window.columns]
        if missing:
            raise ValueError(f"NWP DataFrame missing columns: {missing}")
        arr = nwp_window[NWP_VARS].to_numpy(dtype=np.float32)
    elif isinstance(nwp_window, np.ndarray):
        arr = nwp_window.astype(np.float32, copy=False)
    elif isinstance(nwp_window, torch.Tensor):
        arr = nwp_window.detach().cpu().to(torch.float32).numpy()
    else:
        raise TypeError(
            f"Unsupported nwp_window type: {type(nwp_window).__name__}"
        )
    if arr.ndim != 2 or arr.shape[1] != len(NWP_VARS):
        raise ValueError(
            f"nwp_window must be [seq_len, {len(NWP_VARS)}], got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("nwp_window contains NaN/inf")
    return arr


def forecast_nwp_transformer(
    nwp_window: NWPInput,
    ts: pd.Timestamp,
    kwp: float,
    model: NWPTransformer | None = None,
    model_path: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Forecast 48 h PV power and return an hourly-aligned DataFrame.

    Args:
        nwp_window: NWP input of shape ``[seq_len, 14]`` in ``NWP_VARS``
            column order. Numpy, torch, or DataFrame accepted.
        ts: Timestamp at which the forecast horizon starts (horizon 0).
        kwp: Site capacity in kW.
        model: Pre-loaded ``NWPTransformer``. Provide either ``model`` or
            ``model_path``.
        model_path: Path to a checkpoint. Loaded only when ``model`` is None.
        device: Torch device.

    Returns:
        DataFrame with 48 rows and columns ``timestamp``, ``horizon_hour``,
        ``forecast_kw``.
    """
    if model is None:
        if model_path is None:
            raise ValueError("Provide either `model` or `model_path`.")
        model = load_checkpoint(model_path, device=device)

    arr = _coerce_nwp(nwp_window)
    if arr.shape[0] != model.seq_len:
        raise ValueError(
            f"nwp_window length {arr.shape[0]} != model seq_len {model.seq_len}"
        )

    x = torch.from_numpy(arr).unsqueeze(0).to(device)  # [1, seq_len, 14]
    kwp_t = torch.tensor([float(kwp)], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(x, kwp_t).squeeze(0).cpu().numpy()  # [pred_len]

    # 30-min steps -> hourly samples [0, 2, 4, ..., pred_len-2].
    # For pred_len=96, this yields 48 horizons (0..47).
    pred_hourly = pred[::2]
    horizons = np.arange(len(pred_hourly))
    base_ts = pd.Timestamp(ts)
    times = base_ts + pd.to_timedelta(horizons, unit="h")
    return pd.DataFrame({
        "timestamp": times,
        "horizon_hour": horizons,
        "forecast_kw": pred_hourly.astype(np.float32),
    })
