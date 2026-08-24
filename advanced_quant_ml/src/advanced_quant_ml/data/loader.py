"""Data generation and loading utilities for quantitative finance and ML models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("/working_dir/advanced_quant_ml/data")

YIELD_CURVE_TENORS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
YIELD_CURVE_MATURITIES = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def generate_yield_curve_matrix(start_date: str = "2018-01-01", end_date: str = "2024-12-31", seed: int = 42) -> pd.DataFrame:
    """Generates realistic Treasury yield curve par yields (percentage) spanning 2018-2024."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)

    lambda_param = 0.0609

    level_path = np.zeros(n_days)
    slope_path = np.zeros(n_days)
    curv_path = np.zeros(n_days)

    for i, date in enumerate(dates):
        year = date.year + date.dayofyear / 365.25
        if year < 2020.0:
            level = 3.0 + 0.3 * np.sin(2 * np.pi * (year - 2018) / 2)
            slope = -1.0 + 0.2 * np.cos(2 * np.pi * (year - 2018))
            curv = 0.5
        elif year < 2022.0:
            progress = (year - 2020.0) / 2.0
            level = 1.4 + 0.3 * progress
            slope = -1.3 * (1 - progress) - 0.8 * progress
            curv = 0.8
        elif year < 2024.0:
            progress = (year - 2022.0) / 2.0
            level = 1.7 + 2.8 * progress
            slope = -0.5 + 2.0 * progress
            curv = -0.8 * progress
        else:
            level = 4.4 + 0.2 * np.sin(4 * np.pi * (year - 2024))
            slope = 0.8 - 0.4 * (year - 2024)
            curv = -0.2

        level_path[i] = level
        slope_path[i] = slope
        curv_path[i] = curv

    ar_noise = np.zeros((n_days, 3))
    phi = 0.98
    for i in range(1, n_days):
        ar_noise[i] = phi * ar_noise[i-1] + np.random.normal(0, [0.03, 0.04, 0.05])

    level_path += ar_noise[:, 0]
    slope_path += ar_noise[:, 1]
    curv_path += ar_noise[:, 2]

    maturities = YIELD_CURVE_MATURITIES
    yield_matrix = np.zeros((n_days, len(maturities)))

    for j, tau in enumerate(maturities):
        factor1 = (1.0 - np.exp(-tau * lambda_param)) / (tau * lambda_param)
        factor2 = factor1 - np.exp(-tau * lambda_param)
        y_tau = level_path + slope_path * factor1 + curv_path * factor2
        y_tau = np.maximum(y_tau, 0.05)
        yield_matrix[:, j] = y_tau + np.random.normal(0, 0.004, n_days)

    df_yields = pd.DataFrame(yield_matrix, index=dates, columns=YIELD_CURVE_TENORS)
    df_yields.index.name = "Date"
    return df_yields


def generate_equity_returns_data(start_date: str = "2018-01-01", end_date: str = "2024-12-31", seed: int = 42) -> pd.DataFrame:
    """Generates daily equity returns with realistic GARCH(1,1) volatility clustering and fat tails."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)

    tickers = ["SPY", "QQQ", "AAPL", "NVDA"]
    garch_params = {
        "SPY": {"omega": 1e-6, "alpha": 0.05, "beta": 0.90, "gamma": 0.08, "mu": 0.0004},
        "QQQ": {"omega": 2e-6, "alpha": 0.06, "beta": 0.88, "gamma": 0.09, "mu": 0.0005},
        "AAPL": {"omega": 3e-6, "alpha": 0.07, "beta": 0.87, "gamma": 0.08, "mu": 0.0006},
        "NVDA": {"omega": 8e-6, "alpha": 0.09, "beta": 0.85, "gamma": 0.10, "mu": 0.0012},
    }

    returns_dict = {}
    for ticker in tickers:
        p = garch_params[ticker]
        ret = np.zeros(n_days)
        var = np.zeros(n_days)
        uncond_var = p["omega"] / (1.0 - p["alpha"] - p["beta"] - p["gamma"] / 2.0)
        var[0] = uncond_var
        z = np.random.standard_t(df=6.0, size=n_days) / np.sqrt(6.0 / 4.0)

        for t in range(1, n_days):
            date = dates[t]
            macro_vol_mult = 1.0
            if date.year == 2020 and 2 <= date.month <= 4:
                macro_vol_mult = 3.5
            elif date.year == 2022:
                macro_vol_mult = 1.6

            prev_eps = ret[t-1] - p["mu"]
            leverage_indicator = 1.0 if prev_eps < 0 else 0.0
            
            var[t] = p["omega"] * macro_vol_mult + (p["alpha"] + p["gamma"] * leverage_indicator) * (prev_eps ** 2) + p["beta"] * var[t-1]
            sigma_t = np.sqrt(var[t])
            ret[t] = p["mu"] + sigma_t * z[t]

        returns_dict[ticker] = ret

    df_returns = pd.DataFrame(returns_dict, index=dates)
    df_returns.index.name = "Date"
    return df_returns


def generate_alternative_data(start_date: str = "2018-01-01", end_date: str = "2024-12-31", seed: int = 42) -> pd.DataFrame:
    """Generates synthetic alternative datasets (Sentiment Score, Order Flow Imbalance, Web Search Traffic)."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)

    sentiment = np.zeros(n_days)
    ofi = np.zeros(n_days)
    foot_traffic = np.zeros(n_days)

    for t in range(1, n_days):
        sentiment[t] = 0.85 * sentiment[t-1] + np.random.normal(0, 0.25)
        ofi[t] = 0.60 * ofi[t-1] + np.random.normal(0, 0.40)
        foot_traffic[t] = 0.90 * foot_traffic[t-1] + np.random.normal(0, 0.15)

    df_alt = pd.DataFrame({
        "Sentiment_Score": sentiment,
        "Order_Flow_Imbalance": ofi,
        "Foot_Traffic_Index": 100.0 + foot_traffic * 10.0,
    }, index=dates)
    df_alt.index.name = "Date"
    return df_alt


def generate_all_sample_datasets(data_dir: Optional[Union[str, Path]] = None) -> None:
    """Generates and persists all sample datasets to disk."""
    save_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    df_yields = generate_yield_curve_matrix()
    df_returns = generate_equity_returns_data()
    df_alt = generate_alternative_data()

    df_yields.to_csv(save_dir / "yield_curve_treasury.csv")
    df_returns.to_csv(save_dir / "equity_returns.csv")
    df_alt.to_csv(save_dir / "alternative_data.csv")


def load_yield_curve_data(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads Treasury yield curve matrix (dates x tenors)."""
    load_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    file_path = load_dir / "yield_curve_treasury.csv"
    if not file_path.exists():
        generate_all_sample_datasets(load_dir)
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    return df


def load_equity_returns(data_dir: Optional[Union[str, Path]] = None, ticker: Optional[str] = None) -> Union[pd.DataFrame, pd.Series]:
    """Loads equity return series (or single ticker series)."""
    load_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    file_path = load_dir / "equity_returns.csv"
    if not file_path.exists():
        generate_all_sample_datasets(load_dir)
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    if ticker is not None:
        if ticker in df.columns:
            return df[ticker]
        raise ValueError(f"Ticker {ticker} not found in available equity returns: {list(df.columns)}")
    return df


def load_alternative_data(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads alternative data features."""
    load_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    file_path = load_dir / "alternative_data.csv"
    if not file_path.exists():
        generate_all_sample_datasets(load_dir)
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    return df
