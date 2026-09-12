"""
Financial Machine Learning: Feature Engineering, Fractional Differencing (FFD) & Purged TimeSeries CV.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


def adfuller_test(series: pd.Series, maxlag: int = 1) -> Tuple[float, float]:
    """Lightweight Augmented Dickey-Fuller (ADF) test using pure NumPy/SciPy."""
    ts = series.dropna().values
    n = len(ts)
    if n < 10:
        return 0.0, 1.0

    dy = np.diff(ts)
    y_lag = ts[:-1]
    
    # Regression: dy_t = alpha + beta * y_{t-1} + gamma * dy_{t-1} + eps_t
    if maxlag >= 1 and len(dy) > 2:
        X = np.column_stack([np.ones(len(dy) - 1), y_lag[1:], dy[:-1]])
        y = dy[1:]
    else:
        X = np.column_stack([np.ones(len(dy)), y_lag])
        y = dy

    # OLS: beta_hat = (X'X)^(-1) X'y
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        params = np.linalg.solve(XtX + 1e-8 * np.eye(X.shape[1]), Xty)
        residuals = y - X @ params
        s2 = np.sum(residuals ** 2) / max(len(y) - X.shape[1], 1)
        cov = s2 * np.linalg.inv(XtX + 1e-8 * np.eye(X.shape[1]))
        beta = params[1]
        se_beta = np.sqrt(max(cov[1, 1], 1e-10))
        t_stat = beta / se_beta

        # Approximate p-value from MacKinnon critical values (-3.43 for 1%, -2.86 for 5%, -2.57 for 10%)
        # Standard logistic approximation for ADF p-value
        p_val = float(1.0 / (1.0 + np.exp(-1.5 * (t_stat + 2.86))))
        return float(t_stat), float(np.clip(p_val, 0.0, 1.0))
    except Exception:
        return 0.0, 1.0


def get_weights_ffd(d: float, threshold: float = 1e-4, max_lags: int = 200) -> np.ndarray:
    """Computes weights for fixed-width window fractional differencing."""
    w = [1.0]
    k = 1
    while k < max_lags:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff_fixed_width(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Applies fixed-width window fractional differentiation to a time series."""
    weights = get_weights_ffd(d, threshold=threshold)
    width = len(weights)
    res = {}
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1]
        res[series.index[i]] = np.dot(weights, window)
    return pd.Series(res, name=f"{series.name}_fracdiff_{d:.2f}")


def find_min_d(series: pd.Series, max_d: float = 1.0, step: float = 0.05, p_val_threshold: float = 0.05) -> float:
    """Finds minimal differencing parameter d that achieves stationarity."""
    for d in np.arange(0.0, max_d + step, step):
        fd = frac_diff_fixed_width(series, d)
        if len(fd) > 20:
            _, p_val = adfuller_test(fd.dropna(), maxlag=1)
            if p_val <= p_val_threshold:
                return float(round(d, 2))
    return 1.0


class FinancialFeatureEngineer:
    """Generates stationary technical, volatility, momentum, and fractional differentiation features."""

    def __init__(self, frac_diff_d: Optional[float] = None):
        self.frac_diff_d = frac_diff_d

    def engineer_features(self, df_ohlc: pd.DataFrame, target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Engineers multi-horizon feature matrix X and forward return target y."""
        df = df_ohlc.copy()
        features = pd.DataFrame(index=df.index)

        # Multi-horizon returns & momentum
        close = df["Close"]
        for h in [1, 5, 21, 63, 126]:
            features[f"ret_{h}d"] = close.pct_change(h)

        # Volatility features
        features["vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        if "High" in df.columns and "Low" in df.columns:
            hl = np.log(df["High"] / df["Low"])
            features["parkinson_vol"] = np.sqrt((hl ** 2).rolling(21).mean() / (4 * np.log(2)) * 252)
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            co = np.log(df["Close"] / df["Open"])
            hl = np.log(df["High"] / df["Low"])
            features["garman_klass_vol"] = np.sqrt(
                (0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)).rolling(21).mean() * 252
            )

        # Technical Oscillators
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # Bollinger Z-Score
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features["bollinger_z"] = (close - sma20) / std20.replace(0, np.nan)

        # Moving average ratios
        features["ma_ratio_20_200"] = close.rolling(20).mean() / close.rolling(200).mean() - 1.0

        # Fractional Differencing
        if self.frac_diff_d is None:
            d_opt = find_min_d(close.dropna())
            self.frac_diff_d = d_opt
        
        fd_series = frac_diff_fixed_width(close, self.frac_diff_d)
        features["frac_diff"] = fd_series

        # Target forward returns
        target = close.pct_change(target_horizon).shift(-target_horizon).rename("target_return")

        combined = pd.concat([features, target], axis=1).dropna()
        X = combined.drop(columns=["target_return"])
        y = combined["target_return"]
        return X, y


class PurgedTimeSeriesSplit:
    """Purged and Embargoed K-Fold Cross-Validation for Financial Time Series."""

    def __init__(self, n_splits: int = 5, purge_window: int = 5, embargo_window: int = 5):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False
            
            left_purge_start = max(0, test_start - self.purge_window)
            train_mask[left_purge_start:test_start] = False

            right_embargo_end = min(n_samples, test_end + self.embargo_window)
            train_mask[test_end:right_embargo_end] = False

            train_idx = indices[train_mask]
            splits.append((train_idx, test_idx))
        return splits


@dataclass
class MLPredictorResult:
    information_coefficient: float
    rank_information_coefficient: float
    mse: float
    r_squared: float
    directional_hit_rate: float
    feature_importances: pd.Series
    oof_predictions: pd.Series
    actual_returns: pd.Series

    def summary(self) -> str:
        return f"""Financial ML Performance:
  - Out-of-Sample IC (Pearson):      {self.information_coefficient:+.4f}
  - Rank IC (Spearman):              {self.rank_information_coefficient:+.4f}
  - Directional Hit Rate:            {self.directional_hit_rate:.2%}
  - Mean Squared Error (MSE):        {self.mse:.6f}
  - Out-of-Sample R^2:               {self.r_squared:+.4f}
  - Top 3 Alpha Drivers:             {', '.join(self.feature_importances.head(3).index.tolist())}"""


class MLReturnPredictor:
    """Regularized Financial ML Predictor with Purged K-Fold Cross Validation."""

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0, n_splits: int = 5, purge_window: int = 5):
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.n_splits = n_splits
        self.purge_window = purge_window

    def fit_predict_cv(self, X: pd.DataFrame, y: pd.Series) -> MLPredictorResult:
        cv = PurgedTimeSeriesSplit(n_splits=self.n_splits, purge_window=self.purge_window)
        oof_preds = pd.Series(index=y.index, dtype=float)
        weights_list = []

        X_mat = X.values
        y_vec = y.values

        for train_idx, test_idx in cv.split(X):
            X_tr, y_tr = X_mat[train_idx], y_vec[train_idx]
            X_te = X_mat[test_idx]

            mu, std = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-8
            X_tr_std = (X_tr - mu) / std
            X_te_std = (X_te - mu) / std

            p = X_tr_std.shape[1]
            XtX = X_tr_std.T @ X_tr_std
            Xty = X_tr_std.T @ y_tr
            w = np.linalg.solve(XtX + self.alpha * np.eye(p), Xty)
            weights_list.append(w)

            y_pred = X_te_std @ w
            oof_preds.iloc[test_idx] = y_pred

        valid_mask = ~oof_preds.isna()
        y_true = y[valid_mask]
        y_hat = oof_preds[valid_mask]

        ic, _ = stats.pearsonr(y_true, y_hat)
        rank_ic, _ = stats.spearmanr(y_true, y_hat)
        mse = float(np.mean((y_true - y_hat) ** 2))
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_hat) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        hit_rate = float(np.mean(np.sign(y_true) == np.sign(y_hat)))

        avg_weights = np.mean(weights_list, axis=0)
        feat_imp = pd.Series(np.abs(avg_weights), index=X.columns).sort_values(ascending=False)

        return MLPredictorResult(
            information_coefficient=float(ic),
            rank_information_coefficient=float(rank_ic),
            mse=mse,
            r_squared=r2,
            directional_hit_rate=hit_rate,
            feature_importances=feat_imp,
            oof_predictions=y_hat,
            actual_returns=y_true,
        )
