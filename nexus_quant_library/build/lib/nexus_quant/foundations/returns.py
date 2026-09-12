"""
Foundations: Return Calculation & Compounding Engine.
"""

from typing import Union
import numpy as np
import pandas as pd


def simple_returns(prices: Union[pd.Series, pd.DataFrame, np.ndarray]) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """Calculate simple arithmetic returns: (P_t - P_{t-1}) / P_{t-1}."""
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        return prices.pct_change()
    arr = np.asarray(prices, dtype=float)
    ret = np.full_like(arr, np.nan)
    ret[1:] = (arr[1:] - arr[:-1]) / arr[:-1]
    return ret


def log_returns(prices: Union[pd.Series, pd.DataFrame, np.ndarray]) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """Calculate continuously compounded log returns: ln(P_t / P_{t-1})."""
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        return np.log(prices / prices.shift(1))
    arr = np.asarray(prices, dtype=float)
    ret = np.full_like(arr, np.nan)
    ret[1:] = np.log(arr[1:] / arr[:-1])
    return ret


def cumulative_returns(returns: Union[pd.Series, pd.DataFrame], is_log: bool = False) -> Union[pd.Series, pd.DataFrame]:
    """Compute cumulative wealth trajectory starting at 1.0."""
    if is_log:
        return np.exp(returns.cumsum())
    return (1.0 + returns.fillna(0.0)).cumprod()


def annualized_return(returns: Union[pd.Series, pd.DataFrame, np.ndarray], periods_per_year: int = 252, geometric: bool = True) -> float:
    """Calculate annualized return (CAGR)."""
    clean = np.asarray(returns, dtype=float).flatten()
    clean = clean[~np.isnan(clean)]
    n = len(clean)
    if n == 0:
        return 0.0
    if geometric:
        cum = np.prod(1.0 + clean)
        if cum <= 0:
            return -1.0
        return float(cum ** (periods_per_year / n) - 1.0)
    return float(np.mean(clean) * periods_per_year)


def rolling_returns(prices: Union[pd.Series, pd.DataFrame], window: int = 21) -> Union[pd.Series, pd.DataFrame]:
    """Compute rolling window returns."""
    return prices.pct_change(window)
