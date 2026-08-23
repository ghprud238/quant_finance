"""Historical and realized volatility estimators for financial time series.

Provides classical and advanced range-based volatility models:
- Close-to-Close volatility (sample standard deviation of returns)
- Parkinson (1980) High-Low volatility
- Garman-Klass (1980) Open-High-Low-Close volatility
- Rogers-Satchell (1991) non-zero drift volatility
- Yang-Zhang (2000) minimum variance unbiased jump & continuous volatility
- Volatility Cone analyzer across multi-horizon rolling lookback windows
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quant_foundations.analyzer.returns import log_returns, simple_returns


def _extract_ohlc(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Extract Open, High, Low, Close series from a DataFrame (case-insensitive)."""
    cols_lower = {str(c).lower(): c for c in df.columns}
    required = ["open", "high", "low", "close"]
    missing = [r for r in required if r not in cols_lower]
    if missing:
        raise ValueError(
            f"Missing required OHLC columns: {missing}. Available columns: {list(df.columns)}"
        )

    o = df[cols_lower["open"]].astype(float)
    h = df[cols_lower["high"]].astype(float)
    l = df[cols_lower["low"]].astype(float)
    c = df[cols_lower["close"]].astype(float)
    return o, h, l, c


def _extract_hl(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """Extract High and Low series from a DataFrame (case-insensitive)."""
    cols_lower = {str(c).lower(): c for c in df.columns}
    required = ["high", "low"]
    missing = [r for r in required if r not in cols_lower]
    if missing:
        raise ValueError(
            f"Missing required High/Low columns: {missing}. Available columns: {list(df.columns)}"
        )

    h = df[cols_lower["high"]].astype(float)
    l = df[cols_lower["low"]].astype(float)
    return h, l


def close_to_close_volatility(
    prices: Union[pd.Series, pd.DataFrame],
    window: Optional[int] = 21,
    annualized: bool = True,
    is_log: bool = True,
    periods_per_year: int = 252,
) -> Union[float, pd.Series, pd.DataFrame]:
    r"""Calculate classical close-to-close realized volatility.

    For rolling window :math:`N`:
    .. math::
        \sigma_{\text{C2C}} = \sqrt{ \frac{\text{periods\_per\_year}}{N - 1} \sum_{t=1}^N (r_t - \bar{r})^2 }

    Parameters
    ----------
    prices : Union[pd.Series, pd.DataFrame]
        Price series or DataFrame of asset prices (or Close prices).
    window : Optional[int], default 21
        Rolling window size. If None, computes full sample volatility.
    annualized : bool, default True
        If True, scales by :math:`\sqrt{\text{periods\_per\_year}}`.
    is_log : bool, default True
        If True, uses logarithmic returns; if False, uses simple returns.
    periods_per_year : int, default 252
        Number of trading periods in a year.

    Returns
    -------
    Union[float, pd.Series, pd.DataFrame]
        Rolling volatility series/DataFrame or scalar if window is None for Series.
    """
    if is_log:
        ret = log_returns(prices)
    else:
        ret = simple_returns(prices)

    scale = np.sqrt(periods_per_year) if annualized else 1.0

    if window is None:
        if isinstance(ret, pd.DataFrame):
            return ret.std(ddof=1) * scale
        else:
            return float(ret.dropna().std(ddof=1) * scale)
    else:
        if window <= 1:
            raise ValueError(f"Window must be greater than 1, got {window}")
        return ret.rolling(window=window).std(ddof=1) * scale


def parkinson_volatility(
    df_ohlc: pd.DataFrame,
    window: Optional[int] = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> Union[float, pd.Series]:
    r"""Calculate Parkinson (1980) High-Low range volatility estimator.

    Parkinson volatility is approximately 5 times more efficient than close-to-close volatility
    under geometric Brownian motion with zero drift:
    .. math::
        \sigma_P = \sqrt{\frac{\text{periods\_per\_year}}{4 \ln 2 \cdot N} \sum_{t=1}^N \left(\ln \frac{H_t}{L_t}\right)^2}

    Parameters
    ----------
    df_ohlc : pd.DataFrame
        DataFrame containing 'High' and 'Low' price columns.
    window : Optional[int], default 21
        Rolling window size. If None, returns full sample volatility.
    annualized : bool, default True
        If True, scales by :math:`\sqrt{\text{periods\_per\_year}}`.
    periods_per_year : int, default 252
        Trading periods per year.

    Returns
    -------
    Union[float, pd.Series]
        Rolling Parkinson volatility Series or scalar float.
    """
    h, l = _extract_hl(df_ohlc)
    # Daily variance estimator
    daily_var = (np.log(h / l) ** 2) / (4.0 * np.log(2.0))
    scale = periods_per_year if annualized else 1.0

    if window is None:
        return float(np.sqrt(daily_var.dropna().mean() * scale))
    else:
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        return np.sqrt(daily_var.rolling(window=window).mean() * scale)


def garman_klass_volatility(
    df_ohlc: pd.DataFrame,
    window: Optional[int] = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> Union[float, pd.Series]:
    r"""Calculate Garman-Klass (1980) Open-High-Low-Close volatility estimator.

    Garman-Klass incorporates open and close prices in addition to extremes, achieving
    up to 8 times the efficiency of close-to-close volatility under zero drift:
    .. math::
        \sigma_{GK} = \sqrt{\frac{\text{periods\_per\_year}}{N} \sum_{t=1}^N \left[ 0.5 \left(\ln\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1) \left(\ln\frac{C_t}{O_t}\right)^2 \right]}

    Parameters
    ----------
    df_ohlc : pd.DataFrame
        DataFrame containing 'Open', 'High', 'Low', 'Close' price columns.
    window : Optional[int], default 21
        Rolling window size. If None, returns full sample volatility.
    annualized : bool, default True
        If True, scales by :math:`\sqrt{\text{periods\_per\_year}}`.
    periods_per_year : int, default 252
        Trading periods per year.

    Returns
    -------
    Union[float, pd.Series]
        Rolling Garman-Klass volatility Series or scalar float.
    """
    o, h, l, c = _extract_ohlc(df_ohlc)
    daily_var = 0.5 * (np.log(h / l) ** 2) - (2.0 * np.log(2.0) - 1.0) * (np.log(c / o) ** 2)
    # Ensure non-negative due to numerical precision
    daily_var = np.maximum(daily_var, 0.0)
    scale = periods_per_year if annualized else 1.0

    if window is None:
        return float(np.sqrt(daily_var.dropna().mean() * scale))
    else:
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        return np.sqrt(daily_var.rolling(window=window).mean() * scale)


def rogers_satchell_volatility(
    df_ohlc: pd.DataFrame,
    window: Optional[int] = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> Union[float, pd.Series]:
    r"""Calculate Rogers-Satchell (1991) non-zero drift volatility estimator.

    Allows for non-zero price drift while preserving high efficiency without bias:
    .. math::
        \sigma_{RS} = \sqrt{\frac{\text{periods\_per\_year}}{N} \sum_{t=1}^N \left[ \ln\left(\frac{H_t}{C_t}\right)\ln\left(\frac{H_t}{O_t}\right) + \ln\left(\frac{L_t}{C_t}\right)\ln\left(\frac{L_t}{O_t}\right) \right]}

    Parameters
    ----------
    df_ohlc : pd.DataFrame
        DataFrame containing 'Open', 'High', 'Low', 'Close' price columns.
    window : Optional[int], default 21
        Rolling window size. If None, returns full sample volatility.
    annualized : bool, default True
        If True, scales by :math:`\sqrt{\text{periods\_per\_year}}`.
    periods_per_year : int, default 252
        Trading periods per year.

    Returns
    -------
    Union[float, pd.Series]
        Rolling Rogers-Satchell volatility Series or scalar float.
    """
    o, h, l, c = _extract_ohlc(df_ohlc)
    daily_var = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    daily_var = np.maximum(daily_var, 0.0)
    scale = periods_per_year if annualized else 1.0

    if window is None:
        return float(np.sqrt(daily_var.dropna().mean() * scale))
    else:
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        return np.sqrt(daily_var.rolling(window=window).mean() * scale)


def yang_zhang_volatility(
    df_ohlc: pd.DataFrame,
    window: Optional[int] = 21,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> Union[float, pd.Series]:
    r"""Calculate Yang-Zhang (2000) minimum variance unbiased volatility estimator.

    Combines overnight jump variance and continuous intraday price diffusion:
    .. math::
        V_{YZ} = V_{\text{open}} + k V_{\text{close}} + (1 - k) V_{RS}
    where:
    .. math::
        k = \frac{0.34}{1.34 + \frac{N + 1}{N - 1}}
        V_{\text{open}} = \frac{1}{N - 1} \sum_{t=1}^N (o_t - \bar{o})^2, \quad o_t = \ln(O_t / C_{t-1})
        V_{\text{close}} = \frac{1}{N - 1} \sum_{t=1}^N (c_t - \bar{c})^2, \quad c_t = \ln(C_t / O_t)
        V_{RS} = \frac{1}{N} \sum_{t=1}^N \left[ \ln\left(\frac{H_t}{C_t}\right)\ln\left(\frac{H_t}{O_t}\right) + \ln\left(\frac{L_t}{C_t}\right)\ln\left(\frac{L_t}{O_t}\right) \right]

    Parameters
    ----------
    df_ohlc : pd.DataFrame
        DataFrame containing 'Open', 'High', 'Low', 'Close' price columns.
    window : Optional[int], default 21
        Rolling window size. If None, computes across the full sample.
    annualized : bool, default True
        If True, scales by :math:`\sqrt{\text{periods\_per\_year}}`.
    periods_per_year : int, default 252
        Trading periods per year.

    Returns
    -------
    Union[float, pd.Series]
        Rolling Yang-Zhang volatility Series or scalar float.
    """
    o, h, l, c = _extract_ohlc(df_ohlc)

    o_ret = np.log(o / c.shift(1))
    c_ret = np.log(c / o)
    rs_var = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    scale = periods_per_year if annualized else 1.0

    if window is None:
        n = len(df_ohlc.dropna())
        if n <= 1:
            return np.nan
        k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
        v_open = float(o_ret.dropna().var(ddof=1))
        v_close = float(c_ret.dropna().var(ddof=1))
        v_rs = float(rs_var.dropna().mean())
        total_var = max(v_open + k * v_close + (1.0 - k) * v_rs, 0.0)
        return float(np.sqrt(total_var * scale))
    else:
        if window <= 1:
            raise ValueError(f"Window must be greater than 1, got {window}")
        k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
        v_open = o_ret.rolling(window=window).var(ddof=1)
        v_close = c_ret.rolling(window=window).var(ddof=1)
        v_rs = rs_var.rolling(window=window).mean()
        total_var = np.maximum(v_open + k * v_close + (1.0 - k) * v_rs, 0.0)
        return np.sqrt(total_var * scale)


def volatility_cone(
    df_ohlc: Union[pd.Series, pd.DataFrame],
    windows: Optional[List[int]] = None,
    estimator: str = "close_to_close",
    quantiles: Optional[List[float]] = None,
    annualized: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compute volatility cone percentiles (min, 25%, 50%, 75%, max) across multiple lookback horizons.

    The volatility cone is a standard quantitative tool for comparing the current realized
    volatility against historical distribution bands across terms (e.g., 10d, 21d, 63d, 126d, 252d).

    Parameters
    ----------
    df_ohlc : Union[pd.Series, pd.DataFrame]
        Asset price series (for close-to-close) or OHLC DataFrame.
    windows : Optional[List[int]], default [10, 21, 63, 126, 252]
        List of lookback windows in trading days.
    estimator : str, default 'close_to_close'
        Volatility estimator to use: 'close_to_close', 'parkinson', 'garman_klass',
        'rogers_satchell', or 'yang_zhang'.
    quantiles : Optional[List[float]], default [0.0, 0.25, 0.50, 0.75, 1.0]
        Percentiles to evaluate for each window horizon.
    annualized : bool, default True
        Whether output volatilities are annualized.
    periods_per_year : int, default 252
        Trading periods per year.

    Returns
    -------
    pd.DataFrame
        Volatility cone summary table indexed by window horizon.
    """
    if windows is None:
        windows = [10, 21, 63, 126, 252]
    if quantiles is None:
        quantiles = [0.0, 0.25, 0.50, 0.75, 1.0]

    estimators_map: Dict[str, Callable] = {
        "close_to_close": lambda df, w: close_to_close_volatility(
            df if isinstance(df, pd.Series) else df["Close"] if "Close" in df.columns else df,
            window=w,
            annualized=annualized,
            periods_per_year=periods_per_year,
        ),
        "parkinson": lambda df, w: parkinson_volatility(
            df, window=w, annualized=annualized, periods_per_year=periods_per_year
        ),
        "garman_klass": lambda df, w: garman_klass_volatility(
            df, window=w, annualized=annualized, periods_per_year=periods_per_year
        ),
        "rogers_satchell": lambda df, w: rogers_satchell_volatility(
            df, window=w, annualized=annualized, periods_per_year=periods_per_year
        ),
        "yang_zhang": lambda df, w: yang_zhang_volatility(
            df, window=w, annualized=annualized, periods_per_year=periods_per_year
        ),
    }

    estimator_clean = estimator.lower().replace("-", "_").replace(" ", "_")
    if estimator_clean not in estimators_map:
        raise ValueError(
            f"Unknown estimator '{estimator}'. Valid options: {list(estimators_map.keys())}"
        )

    calc_fn = estimators_map[estimator_clean]
    cone_records = []

    for w in windows:
        vol_series = calc_fn(df_ohlc, w)
        clean_vol = vol_series.dropna()
        if len(clean_vol) == 0:
            rec = {f"{int(q * 100)}%": np.nan for q in quantiles}
            rec["current"] = np.nan
        else:
            rec = {}
            for q in quantiles:
                col_name = "min" if q == 0.0 else "max" if q == 1.0 else f"{int(q * 100)}%"
                rec[col_name] = float(np.quantile(clean_vol, q))
            rec["current"] = float(clean_vol.iloc[-1])

        rec["window"] = w
        cone_records.append(rec)

    df_cone = pd.DataFrame(cone_records)
    df_cone = df_cone.set_index("window")
    return df_cone
