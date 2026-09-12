"""
Foundations: Realized & Range-Based Volatility Estimators.
Implements Close-to-Close, Parkinson (1980), Garman-Klass (1980), Rogers-Satchell (1991), and Yang-Zhang (2000).
"""

from typing import Union, List, Dict
import numpy as np
import pandas as pd


def close_to_close_volatility(
    prices: Union[pd.Series, pd.DataFrame],
    window: int = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> Union[pd.Series, pd.DataFrame]:
    """Rolling sample standard deviation of close-to-close returns."""
    ret = np.log(prices / prices.shift(1))
    vol = ret.rolling(window=window).std(ddof=1)
    if annualized:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def parkinson_volatility(
    df_ohlc: pd.DataFrame,
    window: int = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Parkinson (1980) High-Low range volatility estimator.
    sigma_P = sqrt( 1 / (4 * ln(2) * N) * sum( ln(H/L)^2 ) ) * sqrt(252)
    """
    high = df_ohlc["High"]
    low = df_ohlc["Low"]
    hl_ratio = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_var = factor * hl_ratio.rolling(window=window).mean()
    vol = np.sqrt(np.maximum(rolling_var, 0.0))
    if annualized:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def garman_klass_volatility(
    df_ohlc: pd.DataFrame,
    window: int = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Garman-Klass (1980) OHLC range volatility estimator.
    sigma_GK = sqrt( 1/N * sum[ 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2 ] ) * sqrt(252)
    """
    o, h, l, c = df_ohlc["Open"], df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"]
    log_hl = np.log(h / l) ** 2
    log_co = np.log(c / o) ** 2
    term = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    rolling_var = term.rolling(window=window).mean()
    vol = np.sqrt(np.maximum(rolling_var, 0.0))
    if annualized:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def rogers_satchell_volatility(
    df_ohlc: pd.DataFrame,
    window: int = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Rogers-Satchell (1991) volatility estimator (robust to non-zero drift).
    sigma_RS = sqrt( 1/N * sum[ ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) ] ) * sqrt(252)
    """
    o, h, l, c = df_ohlc["Open"], df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"]
    term = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    rolling_var = term.rolling(window=window).mean()
    vol = np.sqrt(np.maximum(rolling_var, 0.0))
    if annualized:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def yang_zhang_volatility(
    df_ohlc: pd.DataFrame,
    window: int = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Yang-Zhang (2000) minimum variance unbiased estimator combining overnight jump and continuous intraday vol.
    sigma_YZ^2 = sigma_overnight^2 + k * sigma_open_to_close^2 + (1-k) * sigma_RS^2
    """
    o, h, l, c = df_ohlc["Open"], df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"]
    prev_c = c.shift(1)

    log_oc = np.log(o / prev_c)
    log_co = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_hc = np.log(h / c)
    log_lc = np.log(l / c)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    var_o = (log_oc - log_oc.rolling(window=window).mean()) ** 2
    sigma_o_sq = var_o.rolling(window=window).sum() / (window - 1)

    var_c = (log_co - log_co.rolling(window=window).mean()) ** 2
    sigma_c_sq = var_c.rolling(window=window).sum() / (window - 1)

    rs_term = log_ho * log_hc + log_lo * log_lc
    sigma_rs_sq = rs_term.rolling(window=window).mean()

    sigma_yz_sq = sigma_o_sq + k * sigma_c_sq + (1.0 - k) * sigma_rs_sq
    vol = np.sqrt(np.maximum(sigma_yz_sq, 0.0))
    if annualized:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def volatility_cone(
    df_ohlc: pd.DataFrame,
    windows: List[int] = [10, 21, 63, 126, 252],
) -> pd.DataFrame:
    """Compute rolling volatility percentiles (Min, 25%, Median, 75%, Max) across horizons."""
    records = []
    for w in windows:
        if len(df_ohlc) >= w:
            vol = yang_zhang_volatility(df_ohlc, window=w).dropna()
            records.append({
                "Window": w,
                "Min": float(vol.min()),
                "P25": float(vol.quantile(0.25)),
                "Median": float(vol.median()),
                "P75": float(vol.quantile(0.75)),
                "Max": float(vol.max()),
                "Current": float(vol.iloc[-1]) if len(vol) > 0 else 0.0,
            })
    return pd.DataFrame(records).set_index("Window")
