"""Data module: synthetic financial time series generation and data loading."""

from quant_foundations.data.loader import load_factors, load_prices
from quant_foundations.data.synthetic import (
    ALL_TICKERS,
    BENCHMARK_TICKER,
    CORE_TICKERS,
    FACTOR_NAMES,
    generate_and_save_sample_data,
    generate_synthetic_factors,
    generate_synthetic_prices,
)

__all__ = [
    "ALL_TICKERS",
    "BENCHMARK_TICKER",
    "CORE_TICKERS",
    "FACTOR_NAMES",
    "generate_synthetic_prices",
    "generate_synthetic_factors",
    "generate_and_save_sample_data",
    "load_prices",
    "load_factors",
]
