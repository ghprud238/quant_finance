"""
Factor Exposure Report and Variance Risk Decomposition.

Computes factor exposures (betas), annualized alpha, statistical significance,
and detailed variance decomposition (systematic vs. idiosyncratic risk attribution).
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import scipy.stats as stats

from quant_foundations.factors.model import MultiFactorRegression


class FactorExposureReport:
    """
    Factor Exposure & Risk Attribution Report.

    Provides a comprehensive factor decomposition:
    - Factor betas, standard errors, t-statistics, p-values, significance.
    - Annualized alpha with inference.
    - Model fit statistics: R^2, Adjusted R^2, F-statistic.
    - Variance decomposition: Systematic risk (beta^T Sigma_F beta) vs. Idiosyncratic risk (sigma^2_epsilon).
    - Percentage risk attribution per individual factor.
    - Formatted summary tables styled for quantitative analysis dashboards.
    """

    def __init__(
        self,
        model: Optional[MultiFactorRegression] = None,
        asset_returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        factor_returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        risk_free_rate: Union[float, pd.Series, np.ndarray] = 0.0,
        model_type: str = "custom",
        cov_type: str = "hc1",
        annualization_factor: int = 252,
        factor_names: Optional[List[str]] = None,
    ):
        """
        Initialize and generate the Factor Exposure Report.

        Parameters
        ----------
        model : MultiFactorRegression, optional
            Pre-fitted MultiFactorRegression instance. If None, model is fitted
            from asset_returns and factor_returns.
        asset_returns : pd.Series, pd.DataFrame, or np.ndarray, optional
            Asset return series R_{i,t}. Required if model is None.
        factor_returns : pd.Series, pd.DataFrame, or np.ndarray, optional
            Factor return series or DataFrame F_{k,t}. Required if model is None.
        risk_free_rate : float, pd.Series, or np.ndarray, default 0.0
            Risk-free rate R_{f,t}.
        model_type : str, default 'custom'
            Preset factor model ('capm', 'ff3', 'carhart4', 'ff5', 'custom').
        cov_type : str, default 'hc1'
            Covariance matrix type for standard errors ('hc1', 'hc0', 'ols').
        annualization_factor : int, default 252
            Number of periods per year.
        factor_names : list of str, optional
            Explicit factor names.
        """
        self.annualization_factor = annualization_factor

        if model is not None and model.is_fitted:
            self.model = model
        else:
            if asset_returns is None or factor_returns is None:
                raise ValueError("Must provide either a fitted model or both asset_returns and factor_returns.")
            self.model = MultiFactorRegression(
                model_type=model_type,
                cov_type=cov_type,
                annualization_factor=annualization_factor,
            )
            self.model.fit(
                asset_returns=asset_returns,
                factor_returns=factor_returns,
                risk_free_rate=risk_free_rate,
                factor_names=factor_names,
            )

        self._compute_variance_decomposition()

    def _compute_variance_decomposition(self) -> None:
        """Compute systematic risk, idiosyncratic risk, and factor risk attributions."""
        m = self.model
        K = len(m.factor_names)
        N = m.n_observations

        # Factor covariance matrix Sigma_F
        X = m.factor_matrix_  # shape (N, K)
        if N > 1:
            self.factor_cov_matrix = np.cov(X, rowvar=False, ddof=1)
            if K == 1:
                self.factor_cov_matrix = np.array([[float(self.factor_cov_matrix)]])
        else:
            self.factor_cov_matrix = np.zeros((K, K))

        betas_vec = np.array([m.betas[name] for name in m.factor_names])

        # Systematic variance: beta^T * Sigma_F * beta
        self.systematic_variance = float(betas_vec.T @ self.factor_cov_matrix @ betas_vec)

        # Idiosyncratic variance: Var(epsilon)
        residuals = m.residuals
        if len(residuals) > 1:
            self.idiosyncratic_variance = float(np.var(residuals, ddof=1))
        else:
            self.idiosyncratic_variance = 0.0

        # Total excess return variance
        y = m.excess_asset_returns_
        if len(y) > 1:
            self.total_variance = float(np.var(y, ddof=1))
        else:
            self.total_variance = self.systematic_variance + self.idiosyncratic_variance

        # Protect against numerical float precision
        decomp_total = self.systematic_variance + self.idiosyncratic_variance
        base_var = decomp_total if decomp_total > 1e-16 else max(self.total_variance, 1e-16)

        # Risk percentages
        self.systematic_risk_pct = (self.systematic_variance / base_var) * 100.0
        self.idiosyncratic_risk_pct = (self.idiosyncratic_variance / base_var) * 100.0

        # Annualized Volatilities
        ann = self.annualization_factor
        self.total_volatility_ann = np.sqrt(max(0.0, self.total_variance * ann))
        self.systematic_volatility_ann = np.sqrt(max(0.0, self.systematic_variance * ann))
        self.idiosyncratic_volatility_ann = np.sqrt(max(0.0, self.idiosyncratic_variance * ann))

        # Risk attribution per factor: VC_k = beta_k * (Sigma_F * beta)_k
        # sum(VC_k) = beta^T * Sigma_F * beta = systematic_variance
        if K > 0 and self.systematic_variance > 1e-16:
            Sigma_beta = self.factor_cov_matrix @ betas_vec
            self.factor_variance_contributions = {
                name: float(betas_vec[i] * Sigma_beta[i])
                for i, name in enumerate(m.factor_names)
            }
            self.factor_systematic_risk_pct = {
                name: (vc / self.systematic_variance) * 100.0
                for name, vc in self.factor_variance_contributions.items()
            }
            self.factor_total_risk_pct = {
                name: (vc / base_var) * 100.0
                for name, vc in self.factor_variance_contributions.items()
            }
        else:
            self.factor_variance_contributions = {name: 0.0 for name in m.factor_names}
            self.factor_systematic_risk_pct = {name: 0.0 for name in m.factor_names}
            self.factor_total_risk_pct = {name: 0.0 for name in m.factor_names}

    def summary_table(self) -> pd.DataFrame:
        """
        Generate the factor exposure summary table.

        Returns
        -------
        df : pd.DataFrame
            Table with columns: ['Factor', 'Beta', 'Std_Error', 't_Stat', 'p_Value', 'Risk_Attribution_Pct', 'Significance']
        """
        m = self.model
        rows = []
        for name in m.factor_names:
            p_val = m.p_values[name]
            rows.append({
                "Factor": name,
                "Beta": m.betas[name],
                "Std_Error": m.standard_errors[name],
                "t_Stat": m.t_stats[name],
                "p_Value": p_val,
                "Risk_Attribution_Pct": self.factor_systematic_risk_pct.get(name, 0.0),
                "Significance": self.model._format_significance(p_val),
            })
        df = pd.DataFrame(rows).set_index("Factor")
        return df

    def model_metrics(self) -> Dict[str, Union[float, str, int]]:
        """
        Return high-level summary metrics of the model.

        Returns
        -------
        metrics : dict
            Dictionary of model performance and risk decomposition metrics.
        """
        m = self.model
        return {
            "Model_Type": m.model_type.upper(),
            "Observations": m.n_observations,
            "R_Squared": m.r_squared,
            "Adjusted_R_Squared": m.adj_r_squared,
            "F_Statistic": m.f_statistic,
            "F_p_Value": m.f_p_value,
            "Daily_Alpha": m.alpha,
            "Annualized_Alpha": m.annualized_alpha,
            "Alpha_Std_Error": m.alpha_se * self.annualization_factor,
            "Alpha_t_Stat": m.alpha_t_stat,
            "Alpha_p_Value": m.alpha_p_value,
            "Alpha_Significance": m._format_significance(m.alpha_p_value),
            "Total_Volatility_Ann": self.total_volatility_ann,
            "Systematic_Volatility_Ann": self.systematic_volatility_ann,
            "Idiosyncratic_Volatility_Ann": self.idiosyncratic_volatility_ann,
            "Systematic_Risk_Pct": self.systematic_risk_pct,
            "Idiosyncratic_Risk_Pct": self.idiosyncratic_risk_pct,
        }

    def variance_decomposition_table(self) -> pd.DataFrame:
        """
        Return the variance risk decomposition summary table.

        Returns
        -------
        df : pd.DataFrame
            Systematic vs Idiosyncratic risk breakdown.
        """
        data = [
            {
                "Risk_Component": "Systematic (Factor) Risk",
                "Variance_Daily": self.systematic_variance,
                "Volatility_Annualized": self.systematic_volatility_ann,
                "Pct_Total_Risk": self.systematic_risk_pct,
            },
            {
                "Risk_Component": "Idiosyncratic (Specific) Risk",
                "Variance_Daily": self.idiosyncratic_variance,
                "Volatility_Annualized": self.idiosyncratic_volatility_ann,
                "Pct_Total_Risk": self.idiosyncratic_risk_pct,
            },
            {
                "Risk_Component": "Total Excess Risk",
                "Variance_Daily": self.total_variance,
                "Volatility_Annualized": self.total_volatility_ann,
                "Pct_Total_Risk": 100.0,
            },
        ]
        return pd.DataFrame(data).set_index("Risk_Component")

    def to_dict(self) -> Dict:
        """Return full factor exposure report as a dictionary."""
        return {
            "model_metrics": self.model_metrics(),
            "factor_table": self.summary_table().to_dict(orient="index"),
            "variance_decomposition": self.variance_decomposition_table().to_dict(orient="index"),
        }

    def __str__(self) -> str:
        """Formatted string representation of factor exposure report."""
        metrics = self.model_metrics()
        lines = [
            "=" * 78,
            f"FACTOR EXPOSURE & RISK ATTRIBUTION REPORT ({metrics['Model_Type']})",
            "=" * 78,
            f"Observations: {metrics['Observations']} | Annualized Alpha: {metrics['Annualized_Alpha']:.4f} (t={metrics['Alpha_t_Stat']:.2f}, p={metrics['Alpha_p_Value']:.4f}{metrics['Alpha_Significance']})",
            f"R-Squared: {metrics['R_Squared']:.4f} | Adj. R-Squared: {metrics['Adjusted_R_Squared']:.4f}",
            f"Total Vol (Ann): {metrics['Total_Volatility_Ann']:.2%} | Sys Vol: {metrics['Systematic_Volatility_Ann']:.2%} | Idio Vol: {metrics['Idiosyncratic_Volatility_Ann']:.2%}",
            f"Systematic Risk: {metrics['Systematic_Risk_Pct']:.1f}% | Idiosyncratic Risk: {metrics['Idiosyncratic_Risk_Pct']:.1f}%",
            "-" * 78,
            "FACTOR BETAS & RISK ATTRIBUTION:",
            "-" * 78,
            self.summary_table().to_string(),
            "=" * 78,
        ]
        return "\n".join(lines)
