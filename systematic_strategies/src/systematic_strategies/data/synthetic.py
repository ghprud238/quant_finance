"""Synthetic market data generator for systematic trading strategies.

Generates realistic historical price series spanning 2018 to 2024 for:
- Equity Benchmarks & Single Stocks (SPY, QQQ, AAPL, MSFT)
- Cointegrated Asset Pairs (KO/PEP, XOM/CVX with Ornstein-Uhlenbeck mean-reverting spreads)
- Multi-Asset Macro Universe (SPY, TLT, UUP, GLD, USO)
- Cross-Sectional Stock Universe (20 stocks with multi-factor scores: Value, Momentum, Quality, Low_Vol, Size)
"""

from typing import Dict, Tuple, List, Optional
import os
import numpy as np
import pandas as pd


def generate_date_range(start_date: str = "2018-01-01", end_date: str = "2024-12-31") -> pd.DatetimeIndex:
    """Generates business day calendar index."""
    return pd.bdate_range(start=start_date, end=end_date, name="Date")


def generate_equities_data(
    dates: Optional[pd.DatetimeIndex] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates realistic daily OHLCV price series for SPY, QQQ, AAPL, MSFT."""
    if dates is None:
        dates = generate_date_range()
    n_days = len(dates)
    rng = np.random.default_rng(seed)

    tickers = ["SPY", "QQQ", "AAPL", "MSFT"]
    initial_prices = {"SPY": 268.0, "QQQ": 156.0, "AAPL": 43.0, "MSFT": 86.0}
    annual_drifts = {"SPY": 0.12, "QQQ": 0.16, "AAPL": 0.22, "MSFT": 0.20}
    annual_vols = {"SPY": 0.18, "QQQ": 0.23, "AAPL": 0.28, "MSFT": 0.26}

    corr = np.array([
        [1.00, 0.90, 0.78, 0.82],
        [0.90, 1.00, 0.85, 0.88],
        [0.78, 0.85, 1.00, 0.75],
        [0.82, 0.88, 0.75, 1.00],
    ])
    L = np.linalg.cholesky(corr)

    dt = 1.0 / 252.0
    uncorr_shocks = rng.standard_normal((n_days, len(tickers)))
    corr_shocks = uncorr_shocks @ L.T

    price_dict = {}
    for i, ticker in enumerate(tickers):
        drift = annual_drifts[ticker]
        vol = annual_vols[ticker]
        ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * corr_shocks[:, i]

        for t, date in enumerate(dates):
            if pd.Timestamp("2020-02-20") <= date <= pd.Timestamp("2020-03-23"):
                ret[t] -= 0.015 + rng.uniform(0.0, 0.02)
            elif pd.Timestamp("2020-04-01") <= date <= pd.Timestamp("2020-12-31"):
                ret[t] += 0.0018
            elif pd.Timestamp("2022-01-01") <= date <= pd.Timestamp("2022-10-15"):
                ret[t] -= 0.0012
            elif date >= pd.Timestamp("2023-01-01"):
                ret[t] += 0.0008

        log_prices = np.log(initial_prices[ticker]) + np.cumsum(ret)
        price_dict[ticker] = np.exp(log_prices)

    return pd.DataFrame(price_dict, index=dates)


def generate_pairs_data(
    dates: Optional[pd.DatetimeIndex] = None,
    seed: int = 101,
) -> pd.DataFrame:
    """Generates cointegrated asset pairs: KO vs PEP, XOM vs CVX."""
    if dates is None:
        dates = generate_date_range()
    n_days = len(dates)
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    # 1. KO vs PEP
    ko_ret = (0.07 - 0.5 * 0.14**2) * dt + 0.14 * np.sqrt(dt) * rng.standard_normal(n_days)
    ko_prices = 46.0 * np.exp(np.cumsum(ko_ret))

    theta_ko_pep = 0.06
    sigma_ko_pep = 0.80
    spread_ko_pep = np.zeros(n_days)
    for t in range(1, n_days):
        spread_ko_pep[t] = spread_ko_pep[t - 1] - theta_ko_pep * spread_ko_pep[t - 1] + sigma_ko_pep * rng.standard_normal()

    beta_pep = 2.45
    pep_prices = beta_pep * ko_prices + spread_ko_pep + 15.0

    # 2. XOM vs CVX
    xom_ret = (0.10 - 0.5 * 0.26**2) * dt + 0.26 * np.sqrt(dt) * rng.standard_normal(n_days)
    xom_prices = 83.0 * np.exp(np.cumsum(xom_ret))

    theta_xom_cvx = 0.045
    sigma_xom_cvx = 1.40
    spread_xom_cvx = np.zeros(n_days)
    for t in range(1, n_days):
        spread_xom_cvx[t] = spread_xom_cvx[t - 1] - theta_xom_cvx * spread_xom_cvx[t - 1] + sigma_xom_cvx * rng.standard_normal()

    beta_cvx = 1.42
    cvx_prices = beta_cvx * xom_prices + spread_xom_cvx + 10.0

    return pd.DataFrame({
        "KO": ko_prices,
        "PEP": pep_prices,
        "XOM": xom_prices,
        "CVX": cvx_prices,
    }, index=dates)


def generate_macro_data(
    dates: Optional[pd.DatetimeIndex] = None,
    seed: int = 202,
) -> pd.DataFrame:
    """Generates multi-asset macro price series: SPY, TLT, UUP, GLD, USO."""
    if dates is None:
        dates = generate_date_range()
    n_days = len(dates)
    rng = np.random.default_rng(seed)

    assets = ["SPY", "TLT", "UUP", "GLD", "USO"]
    initial_prices = {"SPY": 268.0, "TLT": 125.0, "UUP": 24.0, "GLD": 124.0, "USO": 96.0}
    drifts = {"SPY": 0.12, "TLT": 0.01, "UUP": 0.03, "GLD": 0.09, "USO": 0.04}
    vols = {"SPY": 0.18, "TLT": 0.15, "UUP": 0.08, "GLD": 0.16, "USO": 0.38}

    corr = np.array([
        [ 1.00, -0.35, -0.25,  0.08,  0.32],
        [-0.35,  1.00,  0.10,  0.22, -0.28],
        [-0.25,  0.10,  1.00, -0.45, -0.20],
        [ 0.08,  0.22, -0.45,  1.00,  0.15],
        [ 0.32, -0.28, -0.20,  0.15,  1.00],
    ])
    L = np.linalg.cholesky(corr)
    dt = 1.0 / 252.0

    shocks = rng.standard_normal((n_days, len(assets))) @ L.T
    macro_prices = {}
    for i, asset in enumerate(assets):
        mu = drifts[asset]
        sigma = vols[asset]
        ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks[:, i]

        for t, date in enumerate(dates):
            if pd.Timestamp("2022-01-01") <= date <= pd.Timestamp("2022-10-31"):
                if asset == "TLT":
                    ret[t] -= 0.0016
                elif asset == "UUP":
                    ret[t] += 0.0008

        prices = initial_prices[asset] * np.exp(np.cumsum(ret))
        macro_prices[asset] = prices

    return pd.DataFrame(macro_prices, index=dates)


def generate_cross_sectional_universe(
    n_stocks: int = 20,
    dates: Optional[pd.DatetimeIndex] = None,
    seed: int = 303,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates cross-sectional stock universe with returns and multi-factor scores."""
    if dates is None:
        dates = generate_date_range()
    n_days = len(dates)
    rng = np.random.default_rng(seed)

    tickers = [f"STK_{i+1:02d}" for i in range(n_stocks)]
    factor_names = ["Value", "Momentum", "Quality", "Low_Vol", "Size"]
    true_factor_loadings = rng.standard_normal((n_stocks, len(factor_names)))
    true_factor_loadings = (true_factor_loadings - true_factor_loadings.mean(axis=0)) / true_factor_loadings.std(axis=0)

    factor_drifts = [0.03, 0.06, 0.04, 0.03, 0.02]
    factor_vols = [0.10, 0.12, 0.08, 0.09, 0.11]
    factor_returns = np.zeros((n_days, len(factor_names)))
    for f in range(len(factor_names)):
        factor_returns[:, f] = (factor_drifts[f] / 252.0) + (factor_vols[f] / np.sqrt(252.0)) * rng.standard_normal(n_days)

    mkt_ret = (0.10 / 252.0) + (0.16 / np.sqrt(252.0)) * rng.standard_normal(n_days)

    stock_returns = np.zeros((n_days, n_stocks))
    for s in range(n_stocks):
        idio_vol = rng.uniform(0.18, 0.35) / np.sqrt(252.0)
        idio_shock = idio_vol * rng.standard_normal(n_days)
        stock_returns[:, s] = (
            mkt_ret +
            0.35 * (factor_returns @ true_factor_loadings[s, :]) +
            idio_shock
        )

    initial_prices = rng.uniform(25.0, 150.0, size=n_stocks)
    stock_prices = np.zeros((n_days, n_stocks))
    for s in range(n_stocks):
        stock_prices[:, s] = initial_prices[s] * np.exp(np.cumsum(stock_returns[:, s]))

    prices_df = pd.DataFrame(stock_prices, index=dates, columns=tickers)

    records = []
    for t_idx, date in enumerate(dates):
        for s, ticker in enumerate(tickers):
            if t_idx >= 126:
                mom_val = (prices_df.iloc[t_idx, s] / prices_df.iloc[t_idx - 126, s]) - 1.0
            else:
                mom_val = rng.standard_normal() * 0.1

            val_score = true_factor_loadings[s, 0] + 0.1 * rng.standard_normal()
            mom_score = mom_val + 0.1 * rng.standard_normal()
            qual_score = true_factor_loadings[s, 2] + 0.05 * rng.standard_normal()
            low_vol_score = -abs(stock_returns[max(0, t_idx-21):t_idx+1, s].std()) if t_idx >= 21 else rng.standard_normal()
            size_score = true_factor_loadings[s, 4] + 0.05 * rng.standard_normal()

            records.append({
                "Date": date,
                "Ticker": ticker,
                "Value": val_score,
                "Momentum": mom_score,
                "Quality": qual_score,
                "Low_Vol": low_vol_score,
                "Size": size_score,
            })

    factors_raw = pd.DataFrame(records)
    grouped = factors_raw.groupby("Date")
    standardized_list = []
    for date, group in grouped:
        g = group.copy()
        for col in factor_names:
            std = g[col].std()
            if std > 1e-6:
                g[col] = (g[col] - g[col].mean()) / std
            else:
                g[col] = 0.0
        standardized_list.append(g)

    factors_df = pd.concat(standardized_list, ignore_index=True)
    factors_df.set_index(["Date", "Ticker"], inplace=True)

    return prices_df, factors_df


def generate_and_save_all_sample_data(data_dir: str = "/working_dir/systematic_strategies/data") -> Dict[str, str]:
    """Generates all datasets and saves them to CSV format."""
    os.makedirs(data_dir, exist_ok=True)
    dates = generate_date_range()

    paths = {}

    eq_df = generate_equities_data(dates=dates)
    eq_path = os.path.join(data_dir, "equities.csv")
    eq_df.to_csv(eq_path)
    paths["equities"] = eq_path

    pairs_df = generate_pairs_data(dates=dates)
    pairs_path = os.path.join(data_dir, "pairs.csv")
    pairs_df.to_csv(pairs_path)
    paths["pairs"] = pairs_path

    macro_df = generate_macro_data(dates=dates)
    macro_path = os.path.join(data_dir, "macro.csv")
    macro_df.to_csv(macro_path)
    paths["macro"] = macro_path

    cs_prices, cs_factors = generate_cross_sectional_universe(n_stocks=20, dates=dates)
    cs_prices_path = os.path.join(data_dir, "cross_sectional_prices.csv")
    cs_factors_path = os.path.join(data_dir, "cross_sectional_factors.csv")
    cs_prices.to_csv(cs_prices_path)
    cs_factors.to_csv(cs_factors_path)
    paths["cross_sectional_prices"] = cs_prices_path
    paths["cross_sectional_factors"] = cs_factors_path

    return paths
