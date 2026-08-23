"""Synthetic financial data generator for quantitative finance research and backtesting.

Generates realistic daily OHLCV prices for core assets (AAPL, MSFT, GOOG, AMZN, XOM, TLT, SPY)
and synthetic multi-factor returns using correlated geometric Brownian motion with Cholesky
decomposition, Merton jump-diffusion processes, and macro regime shocks.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Default Tickers
CORE_TICKERS: List[str] = ["AAPL", "MSFT", "GOOG", "AMZN", "XOM", "TLT"]
BENCHMARK_TICKER: str = "SPY"
ALL_TICKERS: List[str] = CORE_TICKERS + [BENCHMARK_TICKER]

# Default factor names
FACTOR_NAMES: List[str] = [
    "MKT-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "MOM",
    "LowVol",
    "RF",
]


def _get_market_parameters() -> Dict[str, Dict[str, float]]:
    """Return realistic empirical market parameters (initial price, annualized drift, annualized vol)."""
    return {
        "AAPL": {"p0": 42.50, "mu": 0.23, "sigma": 0.28, "vol_base": 80_000_000},
        "MSFT": {"p0": 85.50, "mu": 0.21, "sigma": 0.25, "vol_base": 30_000_000},
        "GOOG": {"p0": 53.25, "mu": 0.17, "sigma": 0.26, "vol_base": 25_000_000},
        "AMZN": {"p0": 58.75, "mu": 0.18, "sigma": 0.30, "vol_base": 45_000_000},
        "XOM":  {"p0": 83.60, "mu": 0.09, "sigma": 0.26, "vol_base": 18_000_000},
        "TLT":  {"p0": 126.80, "mu": -0.01, "sigma": 0.16, "vol_base": 12_000_000},
        "SPY":  {"p0": 268.00, "mu": 0.13, "sigma": 0.18, "vol_base": 75_000_000},
    }


def _get_correlation_matrix() -> np.ndarray:
    """Return empirical correlation matrix across ALL_TICKERS."""
    # Order: AAPL, MSFT, GOOG, AMZN, XOM, TLT, SPY
    corr = np.array([
        [1.00, 0.75, 0.70, 0.68, 0.35, -0.28, 0.85],
        [0.75, 1.00, 0.74, 0.72, 0.32, -0.25, 0.88],
        [0.70, 0.74, 1.00, 0.69, 0.30, -0.26, 0.82],
        [0.68, 0.72, 0.69, 1.00, 0.28, -0.24, 0.80],
        [0.35, 0.32, 0.30, 0.28, 1.00, -0.15, 0.55],
        [-0.28, -0.25, -0.26, -0.24, -0.15, 1.00, -0.35],
        [0.85, 0.88, 0.82, 0.80, 0.55, -0.35, 1.00],
    ])
    return corr


def generate_synthetic_prices(
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    tickers: Optional[List[str]] = None,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate realistic daily OHLCV price series for core assets and SPY benchmark.

    Employs correlated geometric Brownian motion (GBM) via Cholesky decomposition of an
    empirical covariance matrix combined with Merton jump-diffusion processes and macro
    regime shocks (e.g., 2020 COVID crash, 2022 Fed rate hike drawdown, 2023 tech rally).

    Parameters
    ----------
    start_date : str
        Start date formatted as 'YYYY-MM-DD'.
    end_date : str
        End date formatted as 'YYYY-MM-DD'.
    tickers : Optional[List[str]]
        List of asset tickers. Defaults to AAPL, MSFT, GOOG, AMZN, XOM, TLT, SPY.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame indexed by Date with levels (Ticker, Field) where Field
        contains ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    if seed is not None:
        np.random.seed(seed)

    if tickers is None:
        tickers = list(ALL_TICKERS)

    market_params = _get_market_parameters()
    # Filter or ensure market params for all requested tickers
    for t in tickers:
        if t not in market_params:
            market_params[t] = {"p0": 100.0, "mu": 0.10, "sigma": 0.20, "vol_base": 20_000_000}

    dt = 1.0 / 252.0
    dates = pd.bdate_range(start_date, end_date)
    num_days = len(dates)
    num_assets = len(tickers)

    # Base volatility and drift vectors
    sigmas = np.array([market_params[t]["sigma"] for t in tickers])
    mus = np.array([market_params[t]["mu"] for t in tickers])
    p0s = np.array([market_params[t]["p0"] for t in tickers])
    base_vols = np.array([market_params[t]["vol_base"] for t in tickers])

    # Construct empirical correlation and covariance matrices
    if set(tickers) == set(ALL_TICKERS) and len(tickers) == len(ALL_TICKERS):
        corr_matrix = _get_correlation_matrix()
        # Align matrix with tickers order
        idx_map = [ALL_TICKERS.index(t) for t in tickers]
        corr_matrix = corr_matrix[np.ix_(idx_map, idx_map)]
    else:
        # Default equicorrelation matrix
        corr_matrix = np.full((num_assets, num_assets), 0.5)
        np.fill_diagonal(corr_matrix, 1.0)

    # Cholesky factor L: Sigma = L @ L.T
    cov_matrix = np.outer(sigmas, sigmas) * corr_matrix
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
    if min_eig < 1e-8:
        cov_matrix += np.eye(num_assets) * (1e-8 - min_eig)
    chol_l = np.linalg.cholesky(cov_matrix)

    # Standard Gaussian innovations
    z = np.random.standard_normal(size=(num_days, num_assets))
    correlated_z = z @ chol_l.T  # shape (num_days, num_assets)

    # Generate Merton Jump Diffusion components
    # Intensity lambda = ~10 jumps per year (~0.04 probability per day)
    jump_lambda = 0.04
    jump_occ = np.random.poisson(jump_lambda, size=(num_days, num_assets))
    jump_mean = -0.015
    jump_std = 0.03
    jump_magnitudes = np.random.normal(jump_mean, jump_std, size=(num_days, num_assets)) * jump_occ
    jump_drift_compensation = jump_lambda * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)

    # Initialize log returns array
    daily_log_returns = np.zeros((num_days, num_assets))

    for i, date in enumerate(dates):
        # Base daily drift and diffusion
        drift_t = (mus - 0.5 * sigmas**2 - jump_drift_compensation) * dt
        diff_t = correlated_z[i, :] * np.sqrt(dt)
        jump_t = jump_magnitudes[i, :]

        # Apply specific historical market regimes
        date_str = date.strftime("%Y-%m-%d")

        # 1. COVID-19 Market Crash (Feb 20, 2020 to March 23, 2020)
        if "2020-02-20" <= date_str <= "2020-03-23":
            vol_multiplier = 3.8
            diff_t = diff_t * vol_multiplier
            # Equity downward jump pressure, TLT safe haven
            equity_mask = np.array([t != "TLT" for t in tickers])
            drift_t[equity_mask] -= 0.022  # severe daily negative shock
            if "TLT" in tickers:
                tlt_idx = tickers.index("TLT")
                drift_t[tlt_idx] += 0.006  # safe haven rally

        # 2. 2020 Post-Crash Recovery & Tech Expansion (March 24, 2020 to Dec 31, 2020)
        elif "2020-03-24" <= date_str <= "2020-12-31":
            for t_idx, t in enumerate(tickers):
                if t in ["AAPL", "MSFT", "AMZN", "GOOG"]:
                    drift_t[t_idx] += 0.0015  # strong tech tailwind
                elif t == "SPY":
                    drift_t[t_idx] += 0.0010

        # 3. 2022 Fed Rate Hike Drawdown & Energy Boom (Jan 03, 2022 to Oct 31, 2022)
        elif "2022-01-03" <= date_str <= "2022-10-31":
            for t_idx, t in enumerate(tickers):
                if t in ["AAPL", "MSFT", "GOOG", "AMZN"]:
                    drift_t[t_idx] -= 0.0016  # Tech valuation compression
                elif t == "TLT":
                    drift_t[t_idx] -= 0.0018  # Bond duration sell-off
                elif t == "XOM":
                    drift_t[t_idx] += 0.0022  # Oil & energy shock rally
                elif t == "SPY":
                    drift_t[t_idx] -= 0.0010

        # 4. 2023-2024 Generative AI & Mega-Cap Tech Rally (Jan 03, 2023 to Dec 31, 2024)
        elif "2023-01-03" <= date_str <= "2024-12-31":
            for t_idx, t in enumerate(tickers):
                if t in ["MSFT", "AAPL", "GOOG", "AMZN"]:
                    drift_t[t_idx] += 0.0014  # AI rally
                elif t == "SPY":
                    drift_t[t_idx] += 0.0008

        daily_log_returns[i, :] = drift_t + diff_t + jump_t

    # Compute Close price paths
    log_price_paths = np.log(p0s) + np.cumsum(daily_log_returns, axis=0)
    close_prices = np.exp(log_price_paths)

    # Generate OHLCV with realistic intraday spreads and volume
    columns_tuples = []
    for t in tickers:
        for f in ["Open", "High", "Low", "Close", "Volume"]:
            columns_tuples.append((t, f))
    multi_cols = pd.MultiIndex.from_tuples(columns_tuples, names=["Ticker", "Field"])

    ohlcv_data = np.zeros((num_days, len(columns_tuples)))

    for t_idx, t in enumerate(tickers):
        c = close_prices[:, t_idx]
        sigma_daily = sigmas[t_idx] * np.sqrt(dt)

        # Overnight gap return (Open relative to previous Close)
        overnight_noise = np.random.normal(0.0, sigma_daily * 0.25, size=num_days)
        # Open price: for day 0 start near p0, subsequent days start from previous close * exp(overnight)
        open_prices = np.zeros(num_days)
        open_prices[0] = p0s[t_idx] * np.exp(overnight_noise[0] * 0.5)
        open_prices[1:] = c[:-1] * np.exp(overnight_noise[1:])

        # Intraday High and Low
        max_oc = np.maximum(open_prices, c)
        min_oc = np.minimum(open_prices, c)

        # Spreads above max(O, C) and below min(O, C)
        high_spread = np.abs(np.random.normal(0.0, sigma_daily * 0.65, size=num_days)) + 0.0015
        low_spread = np.abs(np.random.normal(0.0, sigma_daily * 0.65, size=num_days)) + 0.0015

        high_prices = max_oc * (1.0 + high_spread)
        low_prices = min_oc * (1.0 - low_spread)
        # Ensure positive lower bound
        low_prices = np.maximum(low_prices, 0.01)

        # Volume generation: Log-normal volume with volatility / return amplification
        abs_ret = np.abs(daily_log_returns[:, t_idx])
        vol_noise = np.random.normal(0.0, 0.25, size=num_days)
        vol_multiplier = np.exp(1.2 * (abs_ret / (sigma_daily + 1e-8) - 0.5) + vol_noise)
        volume = np.round(base_vols[t_idx] * np.clip(vol_multiplier, 0.2, 8.0)).astype(int)

        # Populate OHLCV block
        base_col = t_idx * 5
        ohlcv_data[:, base_col] = np.round(open_prices, 4)
        ohlcv_data[:, base_col + 1] = np.round(high_prices, 4)
        ohlcv_data[:, base_col + 2] = np.round(low_prices, 4)
        ohlcv_data[:, base_col + 3] = np.round(c, 4)
        ohlcv_data[:, base_col + 4] = volume

    df_prices = pd.DataFrame(ohlcv_data, index=dates, columns=multi_cols)
    df_prices.index.name = "Date"
    return df_prices


def generate_synthetic_factors(
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    prices_df: Optional[pd.DataFrame] = None,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate synthetic daily factor returns (Fama-French + Momentum + LowVol + RF).

    Factors generated:
    - MKT-RF: Market excess return
    - SMB: Small Minus Big (Size factor)
    - HML: High Minus Low (Value factor)
    - RMW: Robust Minus Weak (Profitability factor)
    - CMA: Conservative Minus Aggressive (Investment factor)
    - MOM: Momentum factor (Winners minus Losers)
    - LowVol: Low Volatility anomaly factor
    - RF: Risk-free rate (daily cash return)

    Parameters
    ----------
    start_date : str
        Start date formatted as 'YYYY-MM-DD'.
    end_date : str
        End date formatted as 'YYYY-MM-DD'.
    prices_df : Optional[pd.DataFrame]
        Optional pre-generated price DataFrame to align SPY market returns.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame of factor returns indexed by Date.
    """
    if seed is not None:
        np.random.seed(seed + 100)

    dates = pd.bdate_range(start_date, end_date)
    num_days = len(dates)

    # 1. Risk-Free Rate (RF)
    # Empirical trajectory: 2018-2019 ~2.0%, 2020-2021 ~0.08%, 2022-2024 ~4.5-5.3%
    rf_daily = np.zeros(num_days)
    for i, date in enumerate(dates):
        year = date.year
        if year in [2018, 2019]:
            annual_rf = 0.020 + np.random.normal(0, 0.001)
        elif year in [2020, 2021]:
            annual_rf = 0.001 + np.random.normal(0, 0.0002)
        elif year == 2022:
            # Ramping from 0.2% to 4.0%
            day_fraction = date.dayofyear / 365.0
            annual_rf = 0.002 + day_fraction * 0.038
        elif year == 2023:
            annual_rf = 0.050 + np.random.normal(0, 0.001)
        else:  # 2024
            annual_rf = 0.048 + np.random.normal(0, 0.001)
        rf_daily[i] = max(annual_rf / 252.0, 0.0)

    # 2. Market Factor (MKT-RF)
    if prices_df is not None and "SPY" in prices_df.columns.levels[0]:
        spy_close = prices_df["SPY"]["Close"].values
        spy_ret = np.zeros(num_days)
        spy_ret[1:] = (spy_close[1:] - spy_close[:-1]) / spy_close[:-1]
        mkt_rf = spy_ret - rf_daily + np.random.normal(0, 0.001, size=num_days)
    else:
        # Generate independent synthetic market factor
        mkt_ret = np.random.normal(0.10 / 252.0, 0.16 / np.sqrt(252.0), size=num_days)
        mkt_rf = mkt_ret - rf_daily

    # 3. Style and Risk Factors (SMB, HML, RMW, CMA, MOM, LowVol)
    # Correlation structure among style factors:
    # Value (HML) vs Momentum (MOM) historically negatively correlated (~ -0.35)
    # Quality (RMW) and Investment (CMA) positively correlated (~ +0.30)
    # LowVol positively correlated with HML/Quality, negatively with High Beta
    num_style = 6
    style_corr = np.array([
        # SMB,  HML,   RMW,   CMA,   MOM,  LowVol
        [1.00,  0.10, -0.20, -0.10,  0.05, -0.15],  # SMB
        [0.10,  1.00,  0.15,  0.40, -0.35,  0.25],  # HML
        [-0.20, 0.15,  1.00,  0.30,  0.10,  0.30],  # RMW
        [-0.10, 0.40,  0.30,  1.00, -0.15,  0.20],  # CMA
        [0.05, -0.35,  0.10, -0.15,  1.00, -0.20],  # MOM
        [-0.15, 0.25,  0.30,  0.20, -0.20,  1.00],  # LowVol
    ])

    style_sigmas = np.array([0.08, 0.10, 0.07, 0.06, 0.13, 0.09]) / np.sqrt(252.0)
    style_mus = np.array([0.015, 0.025, 0.035, 0.020, 0.060, 0.030]) / 252.0

    cov_style = np.outer(style_sigmas, style_sigmas) * style_corr
    chol_style = np.linalg.cholesky(cov_style)

    z_style = np.random.standard_normal(size=(num_days, num_style))
    style_innovations = z_style @ chol_style.T

    # Factor arrays
    smb = style_mus[0] + style_innovations[:, 0]
    hml = style_mus[1] + style_innovations[:, 1]
    rmw = style_mus[2] + style_innovations[:, 2]
    cma = style_mus[3] + style_innovations[:, 3]
    mom = style_mus[4] + style_innovations[:, 4]
    low_vol = style_mus[5] + style_innovations[:, 5]

    # Incorporate macro regime factor shifts
    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        # 2020 Crash: LowVol outperforms, Value lags
        if "2020-02-20" <= date_str <= "2020-03-23":
            low_vol[i] += 0.003
            hml[i] -= 0.004
            mom[i] += 0.002
        # 2022 Value Rotation: HML surges, Momentum whipsaws
        elif "2022-01-03" <= date_str <= "2022-10-31":
            hml[i] += 0.0015
            low_vol[i] += 0.0010
            smb[i] -= 0.0008

    factors_dict = {
        "MKT-RF": np.round(mkt_rf, 6),
        "SMB": np.round(smb, 6),
        "HML": np.round(hml, 6),
        "RMW": np.round(rmw, 6),
        "CMA": np.round(cma, 6),
        "MOM": np.round(mom, 6),
        "LowVol": np.round(low_vol, 6),
        "RF": np.round(rf_daily, 6),
    }

    df_factors = pd.DataFrame(factors_dict, index=dates)
    df_factors.index.name = "Date"
    return df_factors


def generate_and_save_sample_data(
    data_dir: str = "/working_dir/quant_foundations/data",
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample OHLCV prices and factor returns and save them to CSV files.

    Parameters
    ----------
    data_dir : str
        Directory to save sample_prices.csv and sample_factors.csv.
    start_date : str
        Start date formatted as 'YYYY-MM-DD'.
    end_date : str
        End date formatted as 'YYYY-MM-DD'.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (prices_df, factors_df).
    """
    os.makedirs(data_dir, exist_ok=True)

    prices_df = generate_synthetic_prices(
        start_date=start_date, end_date=end_date, seed=seed
    )
    factors_df = generate_synthetic_factors(
        start_date=start_date, end_date=end_date, prices_df=prices_df, seed=seed
    )

    prices_path = os.path.join(data_dir, "sample_prices.csv")
    factors_path = os.path.join(data_dir, "sample_factors.csv")

    prices_df.to_csv(prices_path)
    factors_df.to_csv(factors_path)

    return prices_df, factors_df


if __name__ == "__main__":
    p_df, f_df = generate_and_save_sample_data()
    print(f"Generated prices shape: {p_df.shape}")
    print(f"Generated factors shape: {f_df.shape}")
