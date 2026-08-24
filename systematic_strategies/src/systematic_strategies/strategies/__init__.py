"""Quantitative trading strategies."""

from .factor_long_short import FactorLongShortStrategy
from .multi_asset_trend import MultiAssetTrendStrategy

__all__ = [
    "FactorLongShortStrategy",
    "MultiAssetTrendStrategy",
]
