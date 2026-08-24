"""Systematic quantitative trading strategies package."""

from systematic_strategies.strategies.factor_long_short import FactorLongShortStrategy
from systematic_strategies.strategies.multi_asset_trend import MultiAssetTrendStrategy

__all__ = [
    "FactorLongShortStrategy",
    "MultiAssetTrendStrategy",
]
