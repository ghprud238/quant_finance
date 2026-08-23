"""Return calculation engine for quantitative finance analysis.

Provides functions for arithmetic (simple) returns, logarithmic (continuously compounded)
returns, cumulative returns, rolling window returns, and annualized compound growth rates (CAGR).
"""

from __future__ import annotations

from typing import Union, overload

import numpy as np
import pandas as pd


@overload
def simple_returns(
    prices: pd.Series,
    fillna_zero: bool = False,
) -> pd.Series:
    ...


@overload
def simple_returns(
    prices: pd.DataFrame,
    fillna_zero: bool = False,
) -> pd.DataFrame:
    ...


def simple_returns(
    prices: Union[pd.Series, pd.DataFrame],
    fillna_zero: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    r"""Calculate arithmetic (simple) returns from asset price series.

    The simple return at time :math:`t` is defined as:
    .. math::
        R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1

    Parameters
    ----------
    prices : Union[pd.Series, pd.DataFrame]
        Asset price series or DataFrame of prices indexed by Date.
    fillna_zero : bool, default False
        If True, replaces the initial NaN with 0.0.

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Arithmetic returns series or DataFrame.
    """
    ret = prices.pct_change()
    if fillna_zero:
        ret = ret.fillna(0.0)
    return ret


@overload
def log_returns(
    prices: pd.Series,
    fillna_zero: bool = False,
) -> pd.Series:
    ...


@overload
def log_returns(
    prices: pd.DataFrame,
    fillna_zero: bool = False,
) -> pd.DataFrame:
    ...


def log_returns(
    prices: Union[pd.Series, pd.DataFrame],
    fillna_zero: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    r"""Calculate continuously compounded (logarithmic) returns from asset price series.

    The log return at time :math:`t` is defined as:
    .. math::
        r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})

    Parameters
    ----------
    prices : Union[pd.Series, pd.DataFrame]
        Asset price series or DataFrame of prices indexed by Date.
    fillna_zero : bool, default False
        If True, replaces the initial NaN with 0.0.

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Logarithmic returns series or DataFrame.
    """
    ret = np.log(prices / prices.shift(1))
    if fillna_zero:
        ret = ret.fillna(0.0)
    return ret


@overload
def cumulative_returns(
    returns: pd.Series,
    is_log: bool = False,
) -> pd.Series:
    ...


@overload
def cumulative_returns(
    returns: pd.DataFrame,
    is_log: bool = False,
) -> pd.DataFrame:
    ...


def cumulative_returns(
    returns: Union[pd.Series, pd.DataFrame],
    is_log: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    r"""Calculate cumulative returns from a stream of periodic returns.

    For simple returns:
    .. math::
        R^{cum}_t = \prod_{\tau=1}^t (1 + R_\tau) - 1

    For log returns:
    .. math::
        r^{cum}_t = \sum_{\tau=1}^t r_\tau

    Parameters
    ----------
    returns : Union[pd.Series, pd.DataFrame]
        Periodic return series or DataFrame.
    is_log : bool, default False
        If True, treats inputs as log returns (uses cumulative summation).
        If False, treats inputs as simple returns (uses cumulative product).

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Cumulative returns series or DataFrame.
    """
    # Replace starting NaN with 0 so cumulative series starts cleanly
    cleaned = returns.fillna(0.0)
    if is_log:
        return cleaned.cumsum()
    else:
        return (1.0 + cleaned).cumprod() - 1.0


@overload
def rolling_returns(
    prices: pd.Series,
    window: int = 21,
    is_log: bool = False,
) -> pd.Series:
    ...


@overload
def rolling_returns(
    prices: pd.DataFrame,
    window: int = 21,
    is_log: bool = False,
) -> pd.DataFrame:
    ...


def rolling_returns(
    prices: Union[pd.Series, pd.DataFrame],
    window: int = 21,
    is_log: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    r"""Calculate rolling multi-period returns over a sliding window.

    For simple returns over window :math:`k`:
    .. math::
        R_{t, k} = \frac{P_t - P_{t-k}}{P_{t-k}} = \frac{P_t}{P_{t-k}} - 1

    For log returns over window :math:`k`:
    .. math::
        r_{t, k} = \ln(P_t) - \ln(P_{t-k})

    Parameters
    ----------
    prices : Union[pd.Series, pd.DataFrame]
        Asset price series or DataFrame.
    window : int, default 21
        Lookback window size in number of periods (e.g. 21 trading days ~ 1 month).
    is_log : bool, default False
        If True, calculates rolling log return. If False, calculates rolling simple return.

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Rolling return series or DataFrame.
    """
    if window <= 0:
        raise ValueError(f"Window must be a positive integer, got {window}")

    if is_log:
        return np.log(prices / prices.shift(window))
    else:
        return prices.pct_change(periods=window)


def annualized_return(
    returns: Union[pd.Series, pd.DataFrame],
    is_log: bool = False,
    periods_per_year: int = 252,
) -> Union[float, pd.Series]:
    r"""Calculate the annualized return (CAGR or annualized log growth rate).

    For simple returns, calculates the Compound Annual Growth Rate (CAGR):
    .. math::
        \text{CAGR} = \left( \prod_{t=1}^N (1 + R_t) \right)^{\frac{\text{periods\_per\_year}}{N}} - 1

    For log returns, calculates the annualized mean log return:
    .. math::
        \bar{r}_{\text{ann}} = \left( \frac{1}{N} \sum_{t=1}^N r_t \right) \times \text{periods\_per\_year}

    Parameters
    ----------
    returns : Union[pd.Series, pd.DataFrame]
        Periodic returns (daily, weekly, monthly).
    is_log : bool, default False
        Whether returns are log returns or simple returns.
    periods_per_year : int, default 252
        Number of trading periods per year (252 for daily, 52 for weekly, 12 for monthly).

    Returns
    -------
    Union[float, pd.Series]
        Annualized return as a scalar float (for Series) or Series (for DataFrame).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            lambda col: annualized_return(col, is_log=is_log, periods_per_year=periods_per_year)
        )

    clean_ret = returns.dropna()
    n = len(clean_ret)
    if n == 0:
        return np.nan

    if is_log:
        mean_log_ret = float(clean_ret.mean())
        return mean_log_ret * periods_per_year
    else:
        compounded = float(np.prod(1.0 + clean_ret))
        if compounded <= 0:
            return -1.0
        return float(compounded ** (periods_per_year / n) - 1.0)
