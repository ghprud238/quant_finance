"""Data loader utilities for pricing and factor datasets."""

from __future__ import annotations

import os
from typing import Optional, Union

import pandas as pd

from quant_foundations.data.synthetic import generate_and_save_sample_data

DEFAULT_DATA_DIR = "/working_dir/quant_foundations/data"


def load_prices(
    data_dir: Optional[str] = None,
    file_path: Optional[str] = None,
    auto_generate: bool = True,
    ticker: Optional[str] = None,
    field: Optional[str] = None,
) -> pd.DataFrame:
    """Load stock OHLCV price history from CSV file or auto-generate if missing.

    Parameters
    ----------
    data_dir : Optional[str]
        Directory path where 'sample_prices.csv' is stored. Defaults to DEFAULT_DATA_DIR.
    file_path : Optional[str]
        Direct path to CSV file. Overrides data_dir if provided.
    auto_generate : bool
        If True and the file is not found, automatically generates and saves synthetic dataset.
    ticker : Optional[str]
        If specified, extracts OHLCV columns for the given ticker.
    field : Optional[str]
        If specified (e.g. 'Close'), extracts the specified field across all tickers.

    Returns
    -------
    pd.DataFrame
        DataFrame containing loaded pricing data with DatetimeIndex.
    """
    if file_path is None:
        target_dir = data_dir or DEFAULT_DATA_DIR
        file_path = os.path.join(target_dir, "sample_prices.csv")
    else:
        target_dir = os.path.dirname(file_path) or DEFAULT_DATA_DIR

    if not os.path.exists(file_path):
        if auto_generate:
            generate_and_save_sample_data(data_dir=target_dir)
        else:
            raise FileNotFoundError(f"Price data file not found at: {file_path}")

    # Inspect first few lines to determine header depth
    with open(file_path, "r") as f:
        first_line = f.readline()
        second_line = f.readline()

    # If second line contains OHLCV field names, use two-level header
    if any(k in second_line for k in ["Open", "High", "Low", "Close", "Volume"]):
        df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)

    if ticker is not None:
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[0]:
                df = df[ticker]
            else:
                raise KeyError(f"Ticker '{ticker}' not found in columns.")
        else:
            matching_cols = [c for c in df.columns if c.startswith(f"{ticker}_") or c == ticker]
            if matching_cols:
                df = df[matching_cols]
            else:
                raise KeyError(f"Ticker '{ticker}' not found in columns.")

    if field is not None:
        if isinstance(df.columns, pd.MultiIndex):
            if field in df.columns.levels[1]:
                df = df.xs(field, axis=1, level=1)
            else:
                raise KeyError(f"Field '{field}' not found in columns.")
        else:
            if field in df.columns:
                df = df[[field]]
            else:
                matching_cols = [c for c in df.columns if c.endswith(f"_{field}")]
                if matching_cols:
                    df = df[matching_cols]
                else:
                    raise KeyError(f"Field '{field}' not found in columns.")

    return df


def load_factors(
    data_dir: Optional[str] = None,
    file_path: Optional[str] = None,
    auto_generate: bool = True,
) -> pd.DataFrame:
    """Load daily factor returns from CSV file or auto-generate if missing.

    Parameters
    ----------
    data_dir : Optional[str]
        Directory path where 'sample_factors.csv' is stored. Defaults to DEFAULT_DATA_DIR.
    file_path : Optional[str]
        Direct path to CSV file. Overrides data_dir if provided.
    auto_generate : bool
        If True and the file is not found, automatically generates and saves synthetic dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame of factor returns indexed by Date.
    """
    if file_path is None:
        target_dir = data_dir or DEFAULT_DATA_DIR
        file_path = os.path.join(target_dir, "sample_factors.csv")
    else:
        target_dir = os.path.dirname(file_path) or DEFAULT_DATA_DIR

    if not os.path.exists(file_path):
        if auto_generate:
            generate_and_save_sample_data(data_dir=target_dir)
        else:
            raise FileNotFoundError(f"Factor data file not found at: {file_path}")

    df = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
    return df
