"""Data loading module for systematic trading strategies.

Provides transparent loading with automatic generation fallback.
"""

import os
from typing import Dict, Tuple, Optional
import pandas as pd

from .synthetic import (
    generate_and_save_all_sample_data,
    generate_equities_data,
    generate_pairs_data,
    generate_macro_data,
    generate_cross_sectional_universe,
)

DEFAULT_DATA_DIR = "/working_dir/systematic_strategies/data"


def load_equities(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Loads daily equities prices (SPY, QQQ, AAPL, MSFT)."""
    csv_path = os.path.join(data_dir, "equities.csv")
    if not os.path.exists(csv_path):
        generate_and_save_all_sample_data(data_dir)
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    return df


def load_pairs(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Loads cointegrated pairs prices (KO, PEP, XOM, CVX)."""
    csv_path = os.path.join(data_dir, "pairs.csv")
    if not os.path.exists(csv_path):
        generate_and_save_all_sample_data(data_dir)
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    return df


def load_macro(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Loads multi-asset macro universe prices (SPY, TLT, UUP, GLD, USO)."""
    csv_path = os.path.join(data_dir, "macro.csv")
    if not os.path.exists(csv_path):
        generate_and_save_all_sample_data(data_dir)
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    return df


def load_cross_sectional(data_dir: str = DEFAULT_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads cross-sectional stock prices and standardized factor scores."""
    prices_path = os.path.join(data_dir, "cross_sectional_prices.csv")
    factors_path = os.path.join(data_dir, "cross_sectional_factors.csv")
    if not (os.path.exists(prices_path) and os.path.exists(factors_path)):
        generate_and_save_all_sample_data(data_dir)

    prices_df = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")
    factors_df = pd.read_csv(factors_path, parse_dates=["Date"], index_col=["Date", "Ticker"])
    return prices_df, factors_df
