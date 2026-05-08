"""NWP-only PV forecasting with PatchTST + variable-level attention.

Public API:
    NWPTransformer: the model module (forecasts/nwp_transformer/model.py)
    predict: site/timestamp -> 48 h forecast DataFrame (forecasts/nwp_transformer/inference.py)
"""

from quartz_solar_forecast.forecasts.nwp_transformer.model import NWPTransformer

__all__ = ["NWPTransformer"]
