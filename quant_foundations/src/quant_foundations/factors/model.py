"""
Multi-factor linear regression engine supporting OLS, White HC0/HC1 robust standard errors,
Ridge regularized factor regression, and standard factor model presets.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


# Standard factor model specifications
FACTOR_MODEL_PRESETS: Dict[str, List[str]] = {
    "capm": ["Mkt-RF"],
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "carhart4": ["Mkt-RF", "SMB", "HML", "MOM"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "custom": ["Market", "Value", "Size", "Momentum", "Quality", "Low_Vol"],
}


class MultiFactorRegression:
    """
    Multi-Factor Linear Regression for Asset Return Exposure Analysis.

    Fits the multi-factor asset pricing model:
        R_{i,t} - R_{f,t} = \alpha_i + \sum_{k=1}^K \beta_{i,k} F_{k,t} + \epsilon_{i,t}

    Features:
    - OLS estimation with ordinary homoskedastic or White robust standard errors (HC0, HC1).
    - Ridge L2 regularization for handling multicollinear factor portfolios.
    - Built-in presets for CAPM, Fama-French 3-Factor, Carhart 4-Factor, Fama-French 5-Factor,
      and Custom multi-factor suites (Market, Value, Size, Momentum, Quality, Low Vol).
    - Statistical inference: t-statistics, p-values (via scipy.stats.t.sf), R-squared, Adjusted R-squared.
    - Annualized alpha and standard errors.
    """

    def __init__(
        self,
        model_type: str = "custom",
        cov_type: str = "hc1",
        regularization: Optional[str] = None,
        alpha_ridge: float = 0.0,
        annualization_factor: int = 252,
        fit_intercept: bool = True,
    ):
        """
        Initialize the MultiFactorRegression model.

        Parameters
        ----------
        model_type : str, default 'custom'
            Factor model specification: 'capm', 'ff3', 'carhart4', 'ff5', or 'custom'.
        cov_type : str, default 'hc1'
            Covariance matrix type for standard errors:
            - 'hc1': White HC1 robust standard errors with degrees-of-freedom adjustment (N / (N - p)).
            - 'hc0': White HC0 robust standard errors.
            - 'homoskedastic' / 'ols': Standard OLS homoskedastic variance.
            - 'robust': Alias for 'hc1'.
        regularization : str or None, default None
            Regularization type: None or 'ridge'.
        alpha_ridge : float, default 0.0
            L2 regularization penalty parameter when regularization='ridge'.
        annualization_factor : int, default 252
            Number of trading periods per year (252 for daily, 12 for monthly).
        fit_intercept : bool, default True
            Whether to include an intercept term (alpha).
        """
        self.model_type = model_type.lower()
        self.cov_type = cov_type.lower()
        self.regularization = regularization.lower() if regularization else None
        self.alpha_ridge = float(alpha_ridge)
        self.annualization_factor = annualization_factor
        self.fit_intercept = fit_intercept

        # Estimated parameters
        self.is_fitted: bool = False
        self.alpha: float = 0.0
        self.annualized_alpha: float = 0.0
        self.alpha_se: float = 0.0
        self.alpha_t_stat: float = 0.0
        self.alpha_p_value: float = 1.0

        self.betas: Dict[str, float] = {}
        self.betas_series: pd.Series = pd.Series(dtype=float)
        self.standard_errors: Dict[str, float] = {}
        self.t_stats: Dict[str, float] = {}
        self.p_values: Dict[str, float] = {}

        self.r_squared: float = 0.0
        self.adj_r_squared: float = 0.0
        self.f_statistic: float = 0.0
        self.f_p_value: float = 1.0

        self.residuals: np.ndarray = np.array([])
        self.fitted_values: np.ndarray = np.array([])
        self.cov_matrix: np.ndarray = np.array([])
        self.factor_names: List[str] = []
        self.n_observations: int = 0
        self.df_residuals: int = 0

        # Aligned inputs
        self.excess_asset_returns_: np.ndarray = np.array([])
        self.factor_matrix_: np.ndarray = np.array([])
        self.index_: Optional[pd.Index] = None

    def fit(
        self,
        asset_returns: Union[pd.Series, pd.DataFrame, np.ndarray],
        factor_returns: Union[pd.Series, pd.DataFrame, np.ndarray],
        risk_free_rate: Union[float, pd.Series, np.ndarray] = 0.0,
        factor_names: Optional[List[str]] = None,
    ) -> "MultiFactorRegression":
        """
        Fit the multi-factor regression model.

        Parameters
        ----------
        asset_returns : pd.Series, pd.DataFrame, or np.ndarray
            Asset return series R_{i,t}.
        factor_returns : pd.Series, pd.DataFrame, or np.ndarray
            Factor return series or DataFrame F_{k,t}.
        risk_free_rate : float, pd.Series, or np.ndarray, default 0.0
            Risk-free rate R_{f,t}.
        factor_names : list of str, optional
            Names of factors if not inferrable from factor_returns DataFrame.

        Returns
        -------
        self : MultiFactorRegression
            The fitted model instance.
        """
        y_raw, X_raw, f_names, idx = self._prepare_data(
            asset_returns, factor_returns, risk_free_rate, factor_names
        )

        self.index_ = idx
        self.factor_names = f_names
        N, K = X_raw.shape
        self.n_observations = N

        # Design matrix Z
        if self.fit_intercept:
            Z = np.column_stack([np.ones(N), X_raw])
            p = K + 1
        else:
            Z = X_raw
            p = K

        self.df_residuals = max(1, N - p)

        # Solve regression
        ZtZ = Z.T @ Z
        Zty = Z.T @ y_raw

        if self.regularization == "ridge" or self.alpha_ridge > 0:
            # Do not penalize intercept if present
            penalty_diag = np.full(p, self.alpha_ridge)
            if self.fit_intercept:
                penalty_diag[0] = 0.0
            Gamma = np.diag(penalty_diag)
            try:
                theta = np.linalg.solve(ZtZ + Gamma, Zty)
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(ZtZ + Gamma) @ Zty
        else:
            try:
                theta = np.linalg.solve(ZtZ, Zty)
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(ZtZ) @ Zty

        # Fitted values & residuals
        y_hat = Z @ theta
        residuals = y_raw - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)

        # R-squared and Adj R-squared
        if ss_tot > 0:
            r2 = 1.0 - (ss_res / ss_tot)
        else:
            r2 = 0.0
        r2 = max(0.0, min(1.0, r2))

        if N > p and ss_tot > 0:
            adj_r2 = 1.0 - (1.0 - r2) * ((N - 1) / (N - p))
        else:
            adj_r2 = r2

        # Covariance Matrix of Estimates
        cov_matrix = self._compute_covariance_matrix(Z, residuals, ZtZ, N, p)

        # Standard errors
        diag_cov = np.diag(cov_matrix)
        diag_cov = np.maximum(diag_cov, 1e-16)
        se = np.sqrt(diag_cov)

        # t-stats & two-tailed p-values (scipy.stats.t.sf)
        t_stats = theta / se
        p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=self.df_residuals)

        # F-statistic for overall regression significance (excluding intercept)
        if self.fit_intercept and K > 0 and (N - p) > 0 and (1.0 - r2) > 1e-12:
            f_stat = (r2 / K) / ((1.0 - r2) / (N - p))
            f_pval = float(stats.f.sf(f_stat, K, N - p))
        else:
            f_stat = 0.0
            f_pval = 1.0

        # Unpack parameters
        if self.fit_intercept:
            self.alpha = float(theta[0])
            self.alpha_se = float(se[0])
            self.alpha_t_stat = float(t_stats[0])
            self.alpha_p_value = float(p_values[0])
            self.annualized_alpha = self.alpha * self.annualization_factor

            beta_values = theta[1:]
            beta_se = se[1:]
            beta_t = t_stats[1:]
            beta_p = p_values[1:]
        else:
            self.alpha = 0.0
            self.alpha_se = 0.0
            self.alpha_t_stat = 0.0
            self.alpha_p_value = 1.0
            self.annualized_alpha = 0.0

            beta_values = theta
            beta_se = se
            beta_t = t_stats
            beta_p = p_values

        self.betas = {name: float(b) for name, b in zip(f_names, beta_values)}
        self.betas_series = pd.Series(self.betas, name="Beta")
        self.standard_errors = {name: float(s) for name, s in zip(f_names, beta_se)}
        self.t_stats = {name: float(t) for name, t in zip(f_names, beta_t)}
        self.p_values = {name: float(p_val) for name, p_val in zip(f_names, beta_p)}

        self.r_squared = float(r2)
        self.adj_r_squared = float(adj_r2)
        self.f_statistic = float(f_stat)
        self.f_p_value = float(f_pval)
        self.residuals = residuals
        self.fitted_values = y_hat
        self.cov_matrix = cov_matrix
        self.excess_asset_returns_ = y_raw
        self.factor_matrix_ = X_raw
        self.is_fitted = True

        return self

    def predict(
        self,
        factor_returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict expected excess asset returns given factor returns.

        Parameters
        ----------
        factor_returns : pd.Series, pd.DataFrame, or np.ndarray
            New factor returns matrix.

        Returns
        -------
        y_pred : np.ndarray
            Predicted excess asset returns.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")

        if isinstance(factor_returns, pd.DataFrame):
            X = factor_returns[self.factor_names].values
        elif isinstance(factor_returns, pd.Series):
            X = factor_returns.values.reshape(-1, 1)
        else:
            X = np.asarray(factor_returns)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

        betas_arr = np.array([self.betas[name] for name in self.factor_names])
        y_pred = X @ betas_arr
        if self.fit_intercept:
            y_pred += self.alpha
        return y_pred

    def _compute_covariance_matrix(
        self,
        Z: np.ndarray,
        residuals: np.ndarray,
        ZtZ: np.ndarray,
        N: int,
        p: int,
    ) -> np.ndarray:
        """Compute the coefficient covariance matrix using specified cov_type."""
        try:
            inv_ZtZ = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            inv_ZtZ = np.linalg.pinv(ZtZ)

        cov_type = self.cov_type
        if cov_type in ("homoskedastic", "ols"):
            s2 = np.sum(residuals ** 2) / max(1, N - p)
            return s2 * inv_ZtZ

        # White heteroskedasticity-consistent covariance
        Ze = Z * residuals[:, np.newaxis]
        meat = Ze.T @ Ze
        cov_hc0 = inv_ZtZ @ meat @ inv_ZtZ

        if cov_type == "hc0":
            return cov_hc0
        elif cov_type in ("hc1", "robust"):
            scale = N / max(1, N - p)
            return scale * cov_hc0
        else:
            scale = N / max(1, N - p)
            return scale * cov_hc0

    def _prepare_data(
        self,
        asset_returns: Union[pd.Series, pd.DataFrame, np.ndarray],
        factor_returns: Union[pd.Series, pd.DataFrame, np.ndarray],
        risk_free_rate: Union[float, pd.Series, np.ndarray],
        factor_names: Optional[List[str]],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[pd.Index]]:
        """Align data, subtract risk-free rate, drop NaNs, and extract factor names."""
        # Determine factor dataframe and names
        if isinstance(factor_returns, pd.DataFrame):
            X_df = factor_returns.copy()
            if self.model_type in FACTOR_MODEL_PRESETS and factor_names is None:
                preset_cols = FACTOR_MODEL_PRESETS[self.model_type]
                if all(col in X_df.columns for col in preset_cols):
                    X_df = X_df[preset_cols]
            f_names = list(X_df.columns)
            target_index = X_df.index
        elif isinstance(factor_returns, pd.Series):
            col_name = factor_returns.name if factor_returns.name else "Factor_1"
            X_df = pd.DataFrame({col_name: factor_returns})
            f_names = [str(col_name)]
            target_index = factor_returns.index
        else:
            X_arr = np.asarray(factor_returns)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            num_factors = X_arr.shape[1]
            if factor_names is not None and len(factor_names) == num_factors:
                f_names = list(factor_names)
            elif self.model_type in FACTOR_MODEL_PRESETS and len(FACTOR_MODEL_PRESETS[self.model_type]) == num_factors:
                f_names = list(FACTOR_MODEL_PRESETS[self.model_type])
            else:
                f_names = [f"Factor_{i+1}" for i in range(num_factors)]
            X_df = pd.DataFrame(X_arr, columns=f_names)
            target_index = None

        # Convert asset_returns to Series
        if isinstance(asset_returns, pd.DataFrame):
            y_s = asset_returns.iloc[:, 0].copy()
            if target_index is None:
                target_index = y_s.index
        elif isinstance(asset_returns, pd.Series):
            y_s = asset_returns.copy()
            if target_index is None:
                target_index = y_s.index
        else:
            y_arr = np.asarray(asset_returns).ravel()
            idx = target_index if (target_index is not None and len(target_index) == len(y_arr)) else None
            y_s = pd.Series(y_arr, index=idx)

        # Harmonize index between y_s and X_df if needed
        if target_index is not None:
            if not y_s.index.equals(X_df.index) and len(y_s) == len(X_df):
                # If lengths match but indices differ (e.g. RangeIndex vs DatetimeIndex), adopt X_df.index
                y_s.index = X_df.index
            elif not y_s.index.equals(X_df.index):
                # Align if partial overlap
                common_idx = y_s.index.intersection(X_df.index)
                if len(common_idx) > 0:
                    y_s = y_s.loc[common_idx]
                    X_df = X_df.loc[common_idx]

        # Handle risk-free rate
        if isinstance(risk_free_rate, (pd.Series, pd.DataFrame)):
            rf_s = risk_free_rate.iloc[:, 0].copy() if isinstance(risk_free_rate, pd.DataFrame) else risk_free_rate.copy()
            if len(rf_s) == len(y_s) and not rf_s.index.equals(y_s.index):
                rf_s.index = y_s.index
        elif isinstance(risk_free_rate, np.ndarray) and risk_free_rate.size > 1:
            rf_s = pd.Series(risk_free_rate.ravel(), index=y_s.index)
        else:
            rf_val = float(risk_free_rate) if isinstance(risk_free_rate, (int, float)) else 0.0
            rf_s = pd.Series(rf_val, index=y_s.index)

        # Align on common index
        combined = pd.concat([y_s.rename("y"), X_df, rf_s.rename("rf")], axis=1, join="inner").dropna()
        if len(combined) == 0:
            raise ValueError("Input series have no overlapping non-NaN observations.")

        idx = combined.index
        y_aligned = combined["y"].values - combined["rf"].values
        X_aligned = combined[f_names].values

        return y_aligned, X_aligned, f_names, idx

    def summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of the factor regression results.

        Returns
        -------
        summary_df : pd.DataFrame
            Table with factor exposures, standard errors, t-stats, and p-values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing summary.")

        rows = []
        if self.fit_intercept:
            rows.append({
                "Parameter": "Alpha (Daily)",
                "Coefficient": self.alpha,
                "Std_Error": self.alpha_se,
                "t_Stat": self.alpha_t_stat,
                "p_Value": self.alpha_p_value,
                "Significance": self._format_significance(self.alpha_p_value),
            })
            rows.append({
                "Parameter": "Alpha (Annualized)",
                "Coefficient": self.annualized_alpha,
                "Std_Error": self.alpha_se * self.annualization_factor,
                "t_Stat": self.alpha_t_stat,
                "p_Value": self.alpha_p_value,
                "Significance": self._format_significance(self.alpha_p_value),
            })

        for name in self.factor_names:
            p_val = self.p_values[name]
            rows.append({
                "Parameter": name,
                "Coefficient": self.betas[name],
                "Std_Error": self.standard_errors[name],
                "t_Stat": self.t_stats[name],
                "p_Value": p_val,
                "Significance": self._format_significance(p_val),
            })

        df = pd.DataFrame(rows).set_index("Parameter")
        return df

    @staticmethod
    def _format_significance(p_val: float) -> str:
        """Helper to return standard significance stars."""
        if p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        elif p_val < 0.1:
            return "."
        return ""
