"""Global Macro Cross-Asset & Emerging Markets Sovereign Risk Contagion Model (Project 47).

Implements Vector Autoregression (VAR), Diebold-Yilmaz (2012) Spillover Index,
Generalized Forecast Error Variance Decomposition (GFEVD), and Archimedean
Copula Tail Dependence (Clayton & Gumbel) for systemic sovereign debt crisis modeling.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, rankdata


@dataclass
class DieboldYilmazResult:
    """Results of Diebold-Yilmaz (2012) Volatility & Risk Spillover Index Decomposition."""
    total_spillover_index: float  # Percentage [0, 100%]
    spillover_matrix: pd.DataFrame  # K x K pairwise directional spillovers (%)
    directional_to_others: pd.Series  # Outward spillover from each entity (%)
    directional_from_others: pd.Series  # Inward spillover absorbed by each entity (%)
    net_spillover: pd.Series  # Net transmitter (> 0) or receiver (< 0)
    net_transmitters: List[str]
    net_receivers: List[str]


@dataclass
class CopulaTailDependenceResult:
    """Results of bivariate Copula tail dependence calibration (Clayton & Gumbel)."""
    country_pair: Tuple[str, str]
    kendall_tau: float
    clayton_theta: float
    clayton_lower_tail_dependence: float  # lambda_L = 2^(-1/theta)
    gumbel_theta: float
    gumbel_upper_tail_dependence: float   # lambda_U = 2 - 2^(1/theta)
    asymmetric_tail_risk: str  # 'ELEVATED_CRISIS_CONTAGION' vs 'BALANCED'


@dataclass
class SovereignRiskReport:
    """Consolidated sovereign macro risk and systemic fragility report."""
    as_of_date: str
    total_spillover_index: float
    top_systemic_transmitter: str
    top_vulnerable_receiver: str
    spillover_table: pd.DataFrame
    copula_tail_table: pd.DataFrame


class SovereignContagionEngine:
    """Diebold-Yilmaz Spillover & Extreme Tail Dependence Sovereign Risk Engine."""

    def __init__(self, var_lags: int = 2, forecast_horizon: int = 10):
        self.var_lags = var_lags
        self.forecast_horizon = forecast_horizon

    def estimate_var_parameters(self, data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Estimates Vector Autoregression VAR(p) coefficient matrices and residual covariance Sigma via OLS."""
        X_raw = data_df.values
        T, K = X_raw.shape
        p = self.var_lags

        # Construct lagged design matrix
        Y = X_raw[p:]  # (T-p) x K
        X_lagged = np.hstack([X_raw[p - i - 1: T - i - 1] for i in range(p)])  # (T-p) x (K*p)
        X_design = np.hstack([np.ones((T - p, 1)), X_lagged])  # Add constant

        # OLS: B = (X'X)^(-1) X'Y
        B = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ Y)  # (1 + K*p) x K

        # Residuals & Covariance Sigma
        residuals = Y - X_design @ B
        Sigma = (residuals.T @ residuals) / (T - p - K * p - 1)

        # Extract companion matrix A
        A_matrices = B[1:].reshape(p, K, K)  # p matrices of K x K

        return A_matrices, Sigma

    def compute_diebold_yilmaz_spillovers(
        self,
        series_df: pd.DataFrame,
        var_lags: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
    ) -> DieboldYilmazResult:
        """Computes Generalized Forecast Error Variance Decomposition (GFEVD) and Diebold-Yilmaz Spillover Index."""
        p = var_lags if var_lags is not None else self.var_lags
        H = forecast_horizon if forecast_horizon is not None else self.forecast_horizon

        # Work on first differences (daily changes / returns)
        diff_df = series_df.diff().dropna()
        columns = list(diff_df.columns)
        K = len(columns)

        A_matrices, Sigma = self.estimate_var_parameters(diff_df)

        # 1. Compute VMA (Vector Moving Average) impulse response matrices Psi_h
        Psi = [np.eye(K)]
        for h in range(1, H):
            psi_h = np.zeros((K, K))
            for j in range(min(h, p)):
                psi_h += Psi[h - j - 1] @ A_matrices[j]
            Psi.append(psi_h)

        # 2. Generalized Forecast Error Variance Decomposition (Pesaran & Shin 1998, Diebold & Yilmaz 2012)
        # theta_ij(H) = (sigma_jj^(-1) * sum_{h=0}^{H-1} (e_i' Psi_h Sigma e_j)^2) / sum_{h=0}^{H-1} (e_i' Psi_h Sigma Psi_h' e_i)
        sigma_diag = np.diag(Sigma)
        theta = np.zeros((K, K))

        for i in range(K):
            denom = 0.0
            for h in range(H):
                denom += (Psi[h] @ Sigma @ Psi[h].T)[i, i]

            for j in range(K):
                numer = 0.0
                for h in range(H):
                    term = (Psi[h] @ Sigma)[i, j]
                    numer += term ** 2
                theta[i, j] = (numer / (sigma_diag[j] * max(1e-12, denom)))

        # Normalize rows so that each row sums to 100%
        theta_norm = theta / np.sum(theta, axis=1, keepdims=True) * 100.0

        spillover_matrix = pd.DataFrame(theta_norm, index=columns, columns=columns)

        # Directional spillovers
        # 'From Others' to i: sum_{j != i} theta_ij
        from_others = pd.Series([100.0 - theta_norm[i, i] for i in range(K)], index=columns)

        # 'To Others' from j: sum_{i != j} theta_ij
        to_others = pd.Series(np.sum(theta_norm, axis=0) - np.diag(theta_norm), index=columns)

        # Net Spillover: To - From
        net_spill = to_others - from_others

        # Total Spillover Index S(H): sum_{i!=j} theta_ij / K
        total_index = float(np.sum(from_others) / K)

        transmitters = list(net_spill[net_spill > 0].sort_values(ascending=False).index)
        receivers = list(net_spill[net_spill < 0].sort_values().index)

        return DieboldYilmazResult(
            total_spillover_index=round(total_index, 2),
            spillover_matrix=spillover_matrix.round(2),
            directional_to_others=to_others.round(2),
            directional_from_others=from_others.round(2),
            net_spillover=net_spill.round(2),
            net_transmitters=transmitters,
            net_receivers=receivers,
        )

    def fit_bivariate_clayton_copula(self, u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        """Fits bivariate Clayton Copula via pseudo-MLE and computes lower tail dependence lambda_L."""
        # Convert to uniform pseudo-observations in (0, 1)
        u_p = (rankdata(u) - 0.5) / len(u)
        v_p = (rankdata(v) - 0.5) / len(v)

        # Clayton log-likelihood: l(theta) = sum ln( (1+theta) * (u*v)^(-1-theta) * (u^-theta + v^-theta - 1)^(-2 - 1/theta) )
        def neg_log_lik(theta_val):
            th = theta_val[0]
            if th <= 1e-4 or th > 30.0:
                return 1e8
            term1 = np.log(1.0 + th)
            term2 = (-1.0 - th) * (np.log(u_p) + np.log(v_p))
            inner = u_p ** (-th) + v_p ** (-th) - 1.0
            inner = np.maximum(1e-12, inner)
            term3 = (-2.0 - 1.0 / th) * np.log(inner)
            return -float(np.sum(term1 + term2 + term3))

        res = minimize(neg_log_lik, x0=[1.5], bounds=[(1e-3, 25.0)], method="L-BFGS-B")
        theta_opt = float(res.x[0])
        # Lower tail dependence lambda_L = 2^(-1/theta)
        lambda_L = float(2.0 ** (-1.0 / theta_opt)) if theta_opt > 0 else 0.0

        return round(theta_opt, 3), round(lambda_L, 4)

    def fit_bivariate_gumbel_copula(self, u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        """Fits bivariate Gumbel Copula via pseudo-MLE and computes upper tail dependence lambda_U."""
        u_p = (rankdata(u) - 0.5) / len(u)
        v_p = (rankdata(v) - 0.5) / len(v)

        # Kendall's tau relation for Gumbel: tau = 1 - 1/theta => theta = 1 / (1 - tau)
        from scipy.stats import kendalltau
        tau_val, _ = kendalltau(u, v)
        tau_clean = float(np.clip(tau_val, 0.01, 0.90))
        theta_est = 1.0 / (1.0 - tau_clean)

        # Upper tail dependence lambda_U = 2 - 2^(1/theta)
        lambda_U = float(2.0 - 2.0 ** (1.0 / theta_est))

        return round(theta_est, 3), round(lambda_U, 4)

    def evaluate_pairwise_contagion(
        self,
        data_df: pd.DataFrame,
        pairs: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        """Evaluates Clayton & Gumbel Copula tail dependence across key sovereign pairs."""
        rows = []
        for c1, c2 in pairs:
            if c1 not in data_df.columns or c2 not in data_df.columns:
                continue

            r1 = data_df[c1].diff().dropna().values
            r2 = data_df[c2].diff().dropna().values

            from scipy.stats import kendalltau
            tau, _ = kendalltau(r1, r2)

            cl_th, lambda_L = self.fit_bivariate_clayton_copula(r1, r2)
            gu_th, lambda_U = self.fit_bivariate_gumbel_copula(r1, r2)

            asym = "ELEVATED_CRISIS_CONTAGION" if lambda_L > 0.25 else "MODERATE"

            rows.append({
                "Country_Pair": f"{c1} - {c2}",
                "Kendall_Tau": round(float(tau), 3),
                "Clayton_Theta": cl_th,
                "Lower_Tail_Dep_lambda_L": lambda_L,
                "Gumbel_Theta": gu_th,
                "Upper_Tail_Dep_lambda_U": lambda_U,
                "Contagion_Risk": asym,
            })

        return pd.DataFrame(rows)

    def generate_full_sovereign_report(
        self,
        cds_spreads_df: pd.DataFrame,
        as_of_date: Optional[str] = None,
    ) -> SovereignRiskReport:
        """Generates end-to-end sovereign contagion and systemic risk report."""
        date_str = as_of_date if as_of_date else str(cds_spreads_df.index[-1].date())

        dy_res = self.compute_diebold_yilmaz_spillovers(cds_spreads_df)

        pairs = [
            ("US", "Germany"),
            ("US", "Brazil"),
            ("Germany", "Italy"),
            ("Italy", "Greece"),
            ("Turkey", "Brazil"),
            ("South_Africa", "Turkey"),
            ("India", "Brazil"),
        ]
        copula_df = self.evaluate_pairwise_contagion(cds_spreads_df, pairs)

        spill_table = pd.DataFrame({
            "Country": dy_res.net_spillover.index,
            "To_Others (%)": dy_res.directional_to_others.values,
            "From_Others (%)": dy_res.directional_from_others.values,
            "Net_Spillover (%)": dy_res.net_spillover.values,
            "Systemic_Role": ["NET_TRANSMITTER" if v > 0 else "NET_RECEIVER" for v in dy_res.net_spillover.values],
        }).sort_values("Net_Spillover (%)", ascending=False)

        top_trans = dy_res.net_transmitters[0] if dy_res.net_transmitters else "N/A"
        top_recv = dy_res.net_receivers[0] if dy_res.net_receivers else "N/A"

        return SovereignRiskReport(
            as_of_date=date_str,
            total_spillover_index=dy_res.total_spillover_index,
            top_systemic_transmitter=top_trans,
            top_vulnerable_receiver=top_recv,
            spillover_table=spill_table,
            copula_tail_table=copula_df,
        )
