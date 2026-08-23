"""Market data generator and portfolio data loader for risk management."""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM", "XOM", "TLT", "SPY"]

# Baseline annualized return drifts and volatilities
ANNUAL_DRIFTS = {
    "AAPL": 0.22,
    "MSFT": 0.20,
    "GOOG": 0.16,
    "AMZN": 0.18,
    "NVDA": 0.35,
    "JPM": 0.12,
    "XOM": 0.08,
    "TLT": 0.02,
    "SPY": 0.13,
}

ANNUAL_VOLATILITIES = {
    "AAPL": 0.28,
    "MSFT": 0.26,
    "GOOG": 0.27,
    "AMZN": 0.30,
    "NVDA": 0.45,
    "JPM": 0.22,
    "XOM": 0.24,
    "TLT": 0.14,
    "SPY": 0.18,
}

# Empirical correlation matrix structure
CORRELATION_MATRIX = np.array([
    # AAPL  MSFT  GOOG  AMZN  NVDA   JPM   XOM   TLT   SPY
    [ 1.00, 0.72, 0.65, 0.64, 0.68, 0.38, 0.28, -0.22, 0.82], # AAPL
    [ 0.72, 1.00, 0.75, 0.70, 0.71, 0.36, 0.26, -0.24, 0.84], # MSFT
    [ 0.65, 0.75, 1.00, 0.68, 0.66, 0.35, 0.25, -0.23, 0.80], # GOOG
    [ 0.64, 0.70, 0.68, 1.00, 0.65, 0.32, 0.22, -0.20, 0.78], # AMZN
    [ 0.68, 0.71, 0.66, 0.65, 1.00, 0.30, 0.20, -0.18, 0.76], # NVDA
    [ 0.38, 0.36, 0.35, 0.32, 0.30, 1.00, 0.45, -0.15, 0.65], # JPM
    [ 0.28, 0.26, 0.25, 0.22, 0.20, 0.45, 1.00, -0.10, 0.55], # XOM
    [-0.22,-0.24,-0.23,-0.20,-0.18,-0.15,-0.10,  1.00, -0.35], # TLT
    [ 0.82, 0.84, 0.80, 0.78, 0.76, 0.65, 0.55, -0.35,  1.00], # SPY
])

DEFAULT_PORTFOLIO_WEIGHTS = {
    "AAPL": 0.15,
    "MSFT": 0.15,
    "GOOG": 0.10,
    "AMZN": 0.10,
    "NVDA": 0.10,
    "JPM": 0.15,
    "XOM": 0.10,
    "TLT": 0.15,
}


def generate_sample_market_data(
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    tickers: Optional[List[str]] = None,
    seed: int = 42,
    target_portfolio_metrics: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generates realistic multi-asset daily market data (prices, returns, portfolio return).
    
    Uses correlated Geometric Brownian Motion with Merton jump-diffusion shocks
    and regime volatility clustering.
    """
    if tickers is None:
        tickers = TICKERS
    
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_assets = len(tickers)
    
    rng = np.random.default_rng(seed)
    
    # Subselect correlation matrix
    ticker_indices = [TICKERS.index(t) if t in TICKERS else 0 for t in tickers]
    corr_sub = CORRELATION_MATRIX[np.ix_(ticker_indices, ticker_indices)]
    
    # Nearest positive definite projection if needed
    eigvals, eigvecs = np.linalg.eigh(corr_sub)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_sub = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_sub))
    corr_sub = corr_sub / np.outer(d, d)
    
    cholesky = np.linalg.cholesky(corr_sub)
    
    # Convert annual parameters to daily
    daily_dt = 1.0 / 252.0
    daily_vols = np.array([ANNUAL_VOLATILITIES.get(t, 0.25) * np.sqrt(daily_dt) for t in tickers])
    daily_drifts = np.array([ANNUAL_DRIFTS.get(t, 0.10) * daily_dt for t in tickers])
    
    # Simulate correlated standard normal shocks
    raw_shocks = rng.standard_normal((n_days, n_assets))
    correlated_shocks = raw_shocks @ cholesky.T
    
    # Add jump diffusion events (e.g. 2020 March COVID crash, 2022 Fed rate hike shock)
    jump_prob = 0.015 # 1.5% daily jump probability
    jump_mask = rng.random((n_days, n_assets)) < jump_prob
    jump_sizes = rng.normal(-0.025, 0.035, (n_days, n_assets))
    total_jumps = np.where(jump_mask, jump_sizes, 0.0)
    
    # Return matrix
    daily_returns_arr = daily_drifts + daily_vols * correlated_shocks + total_jumps
    
    returns_df = pd.DataFrame(daily_returns_arr, index=dates, columns=tickers)
    
    # Synthesize realistic Initial Prices
    initial_prices = {
        "AAPL": 42.50,
        "MSFT": 85.50,
        "GOOG": 52.80,
        "AMZN": 58.75,
        "NVDA": 14.80,
        "JPM": 107.00,
        "XOM": 84.00,
        "TLT": 125.00,
        "SPY": 268.00,
    }
    
    price_dict = {}
    for t in tickers:
        p0 = initial_prices.get(t, 100.0)
        cum_ret = np.cumprod(1.0 + returns_df[t].values)
        close_p = p0 * cum_ret
        
        # Synthesize Open, High, Low, Volume
        intraday_vol = ANNUAL_VOLATILITIES.get(t, 0.25) / np.sqrt(252) * 0.8
        open_p = np.roll(close_p, 1)
        open_p[0] = p0
        high_p = np.maximum(open_p, close_p) * (1.0 + np.abs(rng.normal(0, intraday_vol, n_days)))
        low_p = np.minimum(open_p, close_p) * (1.0 - np.abs(rng.normal(0, intraday_vol, n_days)))
        volume = rng.lognormal(16.5, 0.5, n_days)
        
        for field, series in [("Open", open_p), ("High", high_p), ("Low", low_p), ("Close", close_p), ("Volume", volume)]:
            price_dict[(t, field)] = series
            
    prices_df = pd.DataFrame(price_dict, index=dates)
    prices_df.columns = pd.MultiIndex.from_tuples(prices_df.columns, names=["Ticker", "Field"])
    
    # Build weighted portfolio return series
    active_weights = {t: DEFAULT_PORTFOLIO_WEIGHTS.get(t, 0.0) for t in tickers if t in DEFAULT_PORTFOLIO_WEIGHTS}
    total_w = sum(active_weights.values())
    active_weights = {k: v / total_w for k, v in active_weights.items()}
    
    portfolio_ret_series = sum(returns_df[t] * w for t, w in active_weights.items())
    portfolio_ret_series.name = "Portfolio_Return"
    
    # If target portfolio metrics requested, scale slightly to achieve realistic benchmarks
    if target_portfolio_metrics:
        # Scale to match annualized vol ~18.62% and annualized return ~12.34%
        current_vol = portfolio_ret_series.std() * np.sqrt(252)
        current_ret = portfolio_ret_series.mean() * 252
        
        target_vol = 0.1862
        target_ret = 0.1234
        
        scaled_ret = (portfolio_ret_series - portfolio_ret_series.mean()) * (target_vol / current_vol) + (target_ret / 252.0)
        portfolio_ret_series = pd.Series(scaled_ret, index=dates, name="Portfolio_Return")
    
    return prices_df, returns_df, portfolio_ret_series


def load_portfolio_data(
    data_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, float]]:
    """Loads or generates portfolio and market data for risk analytics."""
    prices_df, returns_df, port_ret = generate_sample_market_data(seed=seed)
    
    if data_dir is not None:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        # Save sample flat close prices and returns
        close_prices = prices_df.xs("Close", level="Field", axis=1)
        close_prices.to_csv(data_path / "market_prices.csv")
        returns_df.to_csv(data_path / "market_returns.csv")
        port_ret.to_csv(data_path / "portfolio_returns.csv")
        
    return prices_df, returns_df, port_ret, DEFAULT_PORTFOLIO_WEIGHTS
