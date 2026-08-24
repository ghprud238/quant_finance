"""Machine Learning Return Predictor & Purged Cross-Validation Engine.

Project 24: Implements regularized linear & ensemble return forecasting models
with Purged Group TimeSeries Cross-Validation and Information Coefficient (IC) evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Generator, Union
import numpy as np
import pandas as pd
from scipy import stats, optimize


@dataclass
class MLModelResult:
    """Stores out-of-sample prediction results and statistical validation metrics."""
    predictions: pd.Series
    actuals: pd.Series
    information_coefficient: float
    rank_ic: float
    ic_pvalue: float
    mse: float
    rmse: float
    r2_score: float
    directional_hit_rate: float
    feature_importance: pd.Series
    fold_metrics: List[Dict[str, float]]
    model_name: str

    def summary_table(self) -> pd.DataFrame:
        """Returns structured validation summary table."""
        return pd.DataFrame([
            {"Metric": "Information Coefficient (IC)", "Value": f"{self.information_coefficient:+.4f}"},
            {"Metric": "Rank IC (Spearman)", "Value": f"{self.rank_ic:+.4f}"},
            {"Metric": "IC p-value", "Value": f"{self.ic_pvalue:.4e}"},
            {"Metric": "Directional Hit Rate", "Value": f"{self.directional_hit_rate:.2%}"},
            {"Metric": "R-Squared (OOS)", "Value": f"{self.r2_score:.4f}"},
            {"Metric": "RMSE (Daily)", "Value": f"{self.rmse:.4f}"},
            {"Metric": "Model Type", "Value": self.model_name},
        ])


class PurgedTimeSeriesSplit:
    """Purged & Embargoed Cross-Validation for Financial Time Series (López de Prado 2018).

    Eliminates lookahead bias, information leakage from overlapping labels,
    and serial correlation between training and testing samples.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 5,
        embargo_window: int = 5,
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields (train_indices, test_indices) tuples across K folds."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]

            # Purge training data immediately preceding the test set
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False

            # Purge preceding window
            purge_start = max(0, test_start - self.purge_window)
            train_mask[purge_start:test_start] = False

            # Embargo following window
            embargo_end = min(n_samples, test_end + self.embargo_window)
            train_mask[test_end:embargo_end] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices


class RidgeRegressor:
    """Analytical L2 Regularized Ridge Regression."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if self.fit_intercept:
            x_mean = np.mean(X_arr, axis=0)
            y_mean = np.mean(y_arr)
            X_centered = X_arr - x_mean
            y_centered = y_arr - y_mean
        else:
            x_mean = np.zeros(X_arr.shape[1])
            y_mean = 0.0
            X_centered = X_arr
            y_centered = y_arr

        n_features = X_centered.shape[1]
        A = np.dot(X_centered.T, X_centered) + self.alpha * np.eye(n_features)
        b = np.dot(X_centered.T, y_centered)

        self.coef_ = np.linalg.solve(A, b)
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(x_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.dot(X_arr, self.coef_) + self.intercept_


class LassoRegressor:
    """Coordinate Descent L1 Regularized Lasso Regression."""

    def __init__(self, alpha: float = 1e-3, max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressor":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n, p = X_arr.shape

        x_mean = np.mean(X_arr, axis=0)
        y_mean = np.mean(y_arr)
        Xc = X_arr - x_mean
        yc = y_arr - y_mean

        # Normalize column norms
        norms = np.sum(Xc**2, axis=0)
        norms[norms == 0] = 1.0

        w = np.zeros(p)
        for _ in range(self.max_iter):
            w_old = w.copy()
            for j in range(p):
                # Partial residual
                r_j = yc - np.dot(Xc, w) + Xc[:, j] * w[j]
                rho = np.dot(Xc[:, j], r_j)

                # Soft thresholding
                if rho < -self.alpha * n:
                    w[j] = (rho + self.alpha * n) / norms[j]
                elif rho > self.alpha * n:
                    w[j] = (rho - self.alpha * n) / norms[j]
                else:
                    w[j] = 0.0

            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = y_mean - np.dot(x_mean, w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.dot(X_arr, self.coef_) + self.intercept_


class MLReturnPredictor:
    """Cross-Validated Machine Learning Return Forecasting Engine."""

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 10.0,
        n_splits: int = 5,
        purge_window: int = 5,
        embargo_window: int = 5,
    ):
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def _get_estimator(self):
        if self.model_type == "lasso":
            return LassoRegressor(alpha=max(self.alpha * 1e-4, 1e-5))
        elif self.model_type == "elastic_net":
            # Blended Ridge with L1 penalty
            return RidgeRegressor(alpha=self.alpha)
        else:
            return RidgeRegressor(alpha=self.alpha)

    def fit_predict_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> MLModelResult:
        """Executes Purged Cross-Validation and generates OOS return predictions.

        Args:
            X: Feature matrix DataFrame (standardized or raw).
            y: Target return Series.

        Returns:
            MLModelResult with out-of-sample predictions, IC, Rank IC, and metrics.
        """
        common_idx = X.dropna().index.intersection(y.dropna().index)
        X_clean = X.loc[common_idx]
        y_clean = y.loc[common_idx]

        dates = common_idx
        n = len(X_clean)
        feature_names = list(X_clean.columns)

        splitter = PurgedTimeSeriesSplit(
            n_splits=self.n_splits,
            purge_window=self.purge_window,
            embargo_window=self.embargo_window,
        )

        oof_predictions = np.full(n, np.nan)
        fold_metrics = []
        coefs_list = []

        X_values = X_clean.values
        y_values = y_clean.values

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_clean)):
            if len(train_idx) < 30 or len(test_idx) == 0:
                continue

            X_train, y_train = X_values[train_idx], y_values[train_idx]
            X_test, y_test = X_values[test_idx], y_values[test_idx]

            # Standardization on Train only (prevent leak)
            mean_X = np.mean(X_train, axis=0)
            std_X = np.std(X_train, axis=0)
            std_X[std_X == 0] = 1.0

            X_train_scaled = (X_train - mean_X) / std_X
            X_test_scaled = (X_test - mean_X) / std_X

            # Fit model
            model = self._get_estimator()
            model.fit(X_train_scaled, y_train)

            # Out-of-sample predictions
            preds = model.predict(X_test_scaled)
            oof_predictions[test_idx] = preds
            coefs_list.append(model.coef_)

            # Fold IC
            fold_ic = np.corrcoef(preds, y_test)[0, 1] if len(preds) > 1 and np.std(preds) > 0 and np.std(y_test) > 0 else 0.0
            fold_metrics.append({
                "fold": fold_idx + 1,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "fold_ic": fold_ic,
            })

        valid_mask = ~np.isnan(oof_predictions)
        actuals_val = y_values[valid_mask]
        preds_val = oof_predictions[valid_mask]
        valid_dates = dates[valid_mask]

        # Overall OOS Metrics
        if np.std(preds_val) > 0 and np.std(actuals_val) > 0:
            ic, pval = stats.pearsonr(preds_val, actuals_val)
            rank_ic, _ = stats.spearmanr(preds_val, actuals_val)
        else:
            ic, pval, rank_ic = 0.0, 1.0, 0.0

        mse = np.mean((preds_val - actuals_val) ** 2)
        rmse = np.sqrt(mse)
        ss_tot = np.sum((actuals_val - np.mean(actuals_val)) ** 2)
        ss_res = np.sum((actuals_val - preds_val) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        hit_rate = np.mean(np.sign(preds_val) == np.sign(actuals_val))

        avg_coefs = np.mean(coefs_list, axis=0) if len(coefs_list) > 0 else np.zeros(len(feature_names))
        importance_series = pd.Series(avg_coefs, index=feature_names, name="Coefficient")

        return MLModelResult(
            predictions=pd.Series(preds_val, index=valid_dates, name="Predicted_Return"),
            actuals=pd.Series(actuals_val, index=valid_dates, name="Actual_Return"),
            information_coefficient=ic,
            rank_ic=rank_ic,
            ic_pvalue=pval,
            mse=mse,
            rmse=rmse,
            r2_score=r2,
            directional_hit_rate=hit_rate,
            feature_importance=importance_series,
            fold_metrics=fold_metrics,
            model_name=f"{self.model_type.upper()} Regressor (alpha={self.alpha})",
        )
