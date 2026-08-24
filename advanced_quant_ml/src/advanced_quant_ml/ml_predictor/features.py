"""Financial Feature Engineering & Fractional Differentiation Engine.

Project 24: Generates multi-horizon momentum, range volatility, technical indicators,
and memory-preserving fractionally differenced features for return prediction.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from scipy import stats


def get_ffd_weights(d: float, thres: float = 1e-4, max_lags: int = 2000) -> np.ndarray:
    """Calculates fixed-width window fractional differentiation weights.

    Formula:
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    Args:
        d: Degree of fractional differentiation (0.0 <= d <= 1.0).
        thres: Weight cutoff threshold to truncate infinite memory series.
        max_lags: Maximum number of lag weights to compute.

    Returns:
        NumPy array of weights [w_0, w_1, ..., w_K].
    """
    w = [1.0]
    k = 1
    while k < max_lags:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w)


def frac_diff_ffd(
    series: Union[pd.Series, np.ndarray],
    d: float,
    thres: float = 1e-4,
) -> pd.Series:
    """Applies Fixed-Width Window Fractional Differencing (FFD).

    Preserves long-term memory of price series while achieving stationarity
    (López de Prado, 2018).

    Args:
        series: Raw price or financial time series.
        d: Fractional differentiation degree (e.g. 0.35 - 0.50).
        thres: Weight threshold cutoff.

    Returns:
        pd.Series with fractionally differenced values (initial warm-up NaNs dropped/aligned).
    """
    if isinstance(series, pd.Series):
        dates = series.index
        s_arr = series.values.astype(float)
    else:
        s_arr = np.asarray(series, dtype=float)
        dates = pd.RangeIndex(len(s_arr))

    if d == 0.0:
        return pd.Series(s_arr, index=dates, name="frac_diff_d0")
    if d == 1.0:
        res = np.diff(s_arr, prepend=np.nan)
        return pd.Series(res, index=dates, name="frac_diff_d1")

    w = get_ffd_weights(d, thres=thres)
    width = len(w)
    n = len(s_arr)

    output = np.full(n, np.nan)
    # Vectorized convolution / sliding dot-product
    w_rev = w[::-1]
    for i in range(width - 1, n):
        window = s_arr[i - width + 1 : i + 1]
        output[i] = np.dot(w_rev, window)

    return pd.Series(output, index=dates, name=f"frac_diff_d{d:.2f}")


def adf_test_pvalue(series: np.ndarray, max_lags: int = 1) -> float:
    """Performs fast Augmented Dickey-Fuller unit-root test on a 1D array."""
    clean = series[~np.isnan(series)]
    if len(clean) < 30:
        return 1.0

    dy = np.diff(clean)
    y_lag = clean[:-1]
    n = len(dy)

    if max_lags <= 0:
        X = np.column_stack([np.ones(n), y_lag])
        y = dy
    else:
        dy_lag = np.diff(clean[:-1])
        X = np.column_stack([np.ones(n - 1), y_lag[1:], dy_lag])
        y = dy[1:]

    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        res = y - np.dot(X, beta)
        sigma2 = np.sum(res**2) / max(len(y) - X.shape[1], 1)
        cov = sigma2 * np.linalg.inv(np.dot(X.T, X))
        gamma_t_stat = beta[1] / np.sqrt(max(cov[1, 1], 1e-14))

        if gamma_t_stat < -3.43:
            return 0.01
        elif gamma_t_stat < -2.86:
            return 0.05
        elif gamma_t_stat < -2.57:
            return 0.10
        elif gamma_t_stat < -1.94:
            return 0.30
        else:
            return 0.70
    except Exception:
        return 1.0


def find_min_d(
    series: Union[pd.Series, np.ndarray],
    p_threshold: float = 0.05,
    d_step: float = 0.05,
    max_d: float = 1.0,
) -> float:
    """Finds minimum degree d of fractional differencing achieving stationarity."""
    if isinstance(series, pd.Series):
        s_arr = series.dropna().values
    else:
        s_arr = np.asarray(series)
        s_arr = s_arr[~np.isnan(s_arr)]

    for d in np.arange(0.0, max_d + 1e-5, d_step):
        fd = frac_diff_ffd(s_arr, d=round(float(d), 3))
        pval = adf_test_pvalue(fd.values)
        if pval <= p_threshold:
            return round(float(d), 3)
    return max_d


class FinancialFeatureEngineer:
    """Feature Engineering Engine for Quantitative & Machine Learning Models."""

    def __init__(
        self,
        momentum_windows: List[int] = [1, 5, 21, 63, 126],
        volatility_windows: List[int] = [10, 21, 63],
        frac_diff_d: Optional[float] = None,
        target_horizon: int = 1,
    ):
        self.momentum_windows = momentum_windows
        self.volatility_windows = volatility_windows
        self.frac_diff_d = frac_diff_d
        self.target_horizon = target_horizon

    def engineer_features(
        self,
        df_ohlc: pd.DataFrame,
        include_target: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Generates rich alpha feature matrix from OHLCV market data."""
        df = df_ohlc.copy()
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        dates = close.index

        features = pd.DataFrame(index=dates)

        # 1. Multi-Horizon Lagged Returns / Momentum
        for w in self.momentum_windows:
            features[f"return_{w}d"] = close.pct_change(w)
            features[f"log_return_{w}d"] = np.log(close / close.shift(w))

        # 2. Rolling Close-to-Close Volatilities
        for w in self.volatility_windows:
            daily_ret = close.pct_change()
            features[f"vol_cc_{w}d"] = daily_ret.rolling(w).std() * np.sqrt(252)

        # Range-Based Volatilities
        has_ohlc = all(col in df.columns for col in ["Open", "High", "Low", "Close"])
        if has_ohlc:
            high = df["High"]
            low = df["Low"]
            open_p = df["Open"]

            # Parkinson Volatility
            hl_ratio = np.log(high / low) ** 2
            for w in [21, 63]:
                features[f"vol_parkinson_{w}d"] = np.sqrt((252.0 / (4.0 * np.log(2.0) * w)) * hl_ratio.rolling(w).sum())

            # Garman-Klass Volatility
            gk_term = 0.5 * (np.log(high / low) ** 2) - (2.0 * np.log(2.0) - 1.0) * (np.log(close / open_p) ** 2)
            for w in [21, 63]:
                features[f"vol_garman_klass_{w}d"] = np.sqrt((252.0 / w) * gk_term.rolling(w).sum().clip(lower=0.0))

        # 3. Technical Indicators
        # RSI 14
        delta_p = close.diff()
        gain = delta_p.clip(lower=0.0)
        loss = -delta_p.clip(upper=0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean().replace(0.0, 1e-12)
        rs = avg_gain / avg_loss
        features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Bollinger Band Z-Score
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std().replace(0.0, 1e-12)
        features["bollinger_zscore_20"] = (close - sma_20) / std_20

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        features["macd_line"] = macd_line
        features["macd_signal"] = macd_signal
        features["macd_hist"] = macd_line - macd_signal

        # Moving Average Ratios
        features["ma_ratio_20_200"] = sma_20 / close.rolling(200).mean().replace(0.0, 1e-12)
        features["ma_ratio_10_50"] = close.rolling(10).mean() / close.rolling(50).mean().replace(0.0, 1e-12)

        # 4. Fractionally Differenced Feature (FFD)
        d_val = self.frac_diff_d if self.frac_diff_d is not None else 0.40
        features[f"frac_diff_d{d_val:.2f}"] = frac_diff_ffd(close, d=d_val)

        # 5. Forward Return Target
        target = None
        if include_target:
            target = close.pct_change(self.target_horizon).shift(-self.target_horizon)
            target.name = f"target_return_{self.target_horizon}d"

        # Align and drop warm-up NaNs
        valid_idx = features.dropna().index
        if include_target and target is not None:
            valid_idx = valid_idx.intersection(target.dropna().index)
            features = features.loc[valid_idx]
            target = target.loc[valid_idx]
        else:
            features = features.loc[valid_idx]

        return features, target
