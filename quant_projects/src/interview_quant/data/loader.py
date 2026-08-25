"""Data loader and realistic synthetic market data generator for quant pipeline."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "JPM", "XOM", "TLT"]


def generate_market_data(
    tickers: Optional[List[str]] = None,
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generates realistic multi-asset OHLCV data with volatility clustering and regime shifts."""
    np.random.seed(seed)
    if tickers is None:
        tickers = DEFAULT_TICKERS

    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_assets = len(tickers)

    start_prices = {
        "SPY": 270.0, "QQQ": 160.0, "AAPL": 42.5, "MSFT": 85.0, "NVDA": 50.0,
        "GOOG": 52.0, "AMZN": 60.0, "JPM": 105.0, "XOM": 82.0, "TLT": 120.0
    }
    asset_betas = {
        "SPY": 1.0, "QQQ": 1.15, "AAPL": 1.10, "MSFT": 1.10, "NVDA": 1.45,
        "GOOG": 1.05, "AMZN": 1.10, "JPM": 0.95, "XOM": 0.65, "TLT": -0.35
    }

    market_returns = np.zeros(n_days)
    for d in range(n_days):
        date_curr = dates[d]
        if pd.Timestamp("2020-02-20") <= date_curr <= pd.Timestamp("2020-03-23"):
            # COVID Crash
            mu = -0.015
            vol = 0.035
        elif pd.Timestamp("2020-03-24") <= date_curr <= pd.Timestamp("2021-12-31"):
            # Recovery Rally
            mu = 0.00075
            vol = 0.010
        elif pd.Timestamp("2022-01-01") <= date_curr <= pd.Timestamp("2022-10-14"):
            # Bear Market
            mu = -0.00085
            vol = 0.014
        elif pd.Timestamp("2022-10-15") <= date_curr <= pd.Timestamp("2024-12-31"):
            # AI & Quality Expansion
            mu = 0.00080
            vol = 0.0085
        else:
            mu = 0.00045
            vol = 0.009

        market_returns[d] = np.random.normal(mu, vol)

    ret_paths = np.zeros((n_days, n_assets))
    for i, t in enumerate(tickers):
        beta = asset_betas.get(t, 1.0)
        idio_vol = 0.007 if t not in ["NVDA", "AAPL"] else 0.010
        idio_shock = np.random.normal(0, idio_vol, n_days)
        alpha = 0.0002 if t in ["NVDA", "AAPL", "MSFT"] else 0.0
        ret_paths[:, i] = beta * market_returns + alpha + idio_shock

    records = {}
    for i, t in enumerate(tickers):
        init_p = start_prices.get(t, 100.0)
        close_p = init_p * np.exp(np.cumsum(ret_paths[:, i]))
        intraday_vol = 0.010
        open_p = close_p * (1.0 + np.random.normal(0, 0.001, n_days))
        high_p = np.maximum(open_p, close_p) * (1.0 + np.abs(np.random.normal(0, intraday_vol * 0.5, n_days)))
        low_p = np.minimum(open_p, close_p) * (1.0 - np.abs(np.random.normal(0, intraday_vol * 0.5, n_days)))
        volume = np.random.lognormal(16.0, 0.4, n_days)

        records[(t, "Open")] = open_p
        records[(t, "High")] = high_p
        records[(t, "Low")] = low_p
        records[(t, "Close")] = close_p
        records[(t, "Volume")] = volume

    df = pd.DataFrame(records, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Ticker", "Field"])
    df.index.name = "Date"
    return df


def load_dataset(data_dir: str = "/working_dir/interview_quant_projects/data") -> pd.DataFrame:
    """Loads market data or generates if missing."""
    csv_path = f"{data_dir}/market_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
        return df
    df = generate_market_data()
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(csv_path)
    return df
