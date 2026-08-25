"""Wasserstein Distributionally Robust Portfolio Optimization (DRO).

Module 35: Implements data-driven distributionally robust portfolio optimization
using optimal transport ambiguity balls (Kuhn, Esfahani 2018; Blanchet et al. 2019;
Gao & Kleywegt 2022).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


@dataclass
class DROResult:
    """Dataclass holding output metrics for a Wasserstein DRO portfolio optimization."""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    nominal_objective: float
    robust_objective: float
    worst_case_loss: float
    epsilon: float
    norm_p: Union[int, float, str]
    risk_aversion: float
    effective_n_assets: float
    herfindahl_index: float
    converged: bool
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Expected Return (Ann)": self.expected_return,
            "Volatility (Ann)": self.volatility,
            "Sharpe Ratio (Rf=2%)": self.sharpe_ratio,
            "Nominal Objective": self.nominal_objective,
            "Robust Objective (Worst-Case)": self.robust_objective,
            "Worst-Case Loss Upper Bound": self.worst_case_loss,
            "Wasserstein Radius (eps)": self.epsilon,
            "Dual Norm p": str(self.norm_p),
            "Risk Aversion (gamma)": self.risk_aversion,
            "Effective N Assets (1/HHI)": self.effective_n_assets,
            "Herfindahl Index (HHI)": self.herfindahl_index,
            "Convergence": self.converged,
        }

    def summary_table(self) -> pd.DataFrame:
        data = self.to_dict()
        df = pd.DataFrame({
            "Metric": list(data.keys()),
            "Value": [
                f"{v:.2%}" if "Return" in k or "Volatility" in k or "Loss" in k
                else f"{v:.4f}" if isinstance(v, (float, np.floating))
                else str(v)
                for k, v in data.items()
            ]
        })
        return df


class WassersteinDROOptimizer:
    """Wasserstein Distributionally Robust Portfolio Optimizer.
    
    Solves the min-max distributionally robust portfolio selection problem:
        min_{w in W} max_{Q: W_p(Q, P_hat) <= eps} E_Q[ -w^T xi + (gamma / 2) w^T Sigma w ]
        
    Via strong duality for Wasserstein optimal transport, this problem is equivalent to
    the finite-dimensional convex regularized program:
        min_{w in W} { -w^T mu_hat + (gamma / 2) w^T Sigma_hat w + eps * ||w||_q }
        s.t. sum(w_i) = 1, w_i >= 0 (or custom box bounds)
    where ||w||_q is the dual norm of the ground cost metric ||xi - xi_hat||_p.
    """

    def __init__(
        self,
        expected_returns: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        returns_data: Optional[pd.DataFrame] = None,
        risk_aversion: float = 1.0,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        asset_names: Optional[List[str]] = None,
    ):
        self.periods_per_year = periods_per_year
        self.risk_aversion = max(1e-4, float(risk_aversion))
        self.risk_free_rate = float(risk_free_rate)

        if returns_data is not None:
            if not isinstance(returns_data, pd.DataFrame):
                returns_data = pd.DataFrame(returns_data)
            self.returns_data = returns_data.dropna()
            self.asset_names = list(returns_data.columns)
            self.mu = returns_data.mean().values * self.periods_per_year
            self.cov = returns_data.cov().values * self.periods_per_year
        elif expected_returns is not None and cov_matrix is not None:
            self.returns_data = None
            if isinstance(expected_returns, pd.Series):
                self.asset_names = list(expected_returns.index)
                self.mu = expected_returns.values.astype(float)
            else:
                self.mu = np.asarray(expected_returns, dtype=float)
                self.asset_names = asset_names or [f"Asset_{i+1}" for i in range(len(self.mu))]

            if isinstance(cov_matrix, pd.DataFrame):
                self.cov = cov_matrix.values.astype(float)
            else:
                self.cov = np.asarray(cov_matrix, dtype=float)
        else:
            raise ValueError("Must provide either 'returns_data' DataFrame or both 'expected_returns' and 'cov_matrix'.")

        self.n_assets = len(self.mu)
        if self.cov.shape != (self.n_assets, self.n_assets):
            raise ValueError(f"Dimension mismatch: mu has length {self.n_assets}, but cov shape is {self.cov.shape}.")

        # Ensure positive semi-definiteness
        self.cov = 0.5 * (self.cov + self.cov.T)
        min_eig = np.min(np.linalg.eigvalsh(self.cov))
        if min_eig < 1e-8:
            self.cov += (1e-6 - min_eig) * np.eye(self.n_assets)

    def _compute_norm(self, w: np.ndarray, norm_p: Union[int, float, str]) -> float:
        """Computes regularizing dual norm ||w||_p."""
        if norm_p == 2 or norm_p == 2.0 or norm_p == "l2":
            return float(np.linalg.norm(w, 2))
        elif norm_p == 1 or norm_p == 1.0 or norm_p == "l1":
            return float(np.linalg.norm(w, 1))
        elif norm_p == "inf" or norm_p == np.inf or norm_p == "linf":
            return float(np.max(np.abs(w)))
        elif norm_p == "mahalanobis":
            return float(np.sqrt(np.maximum(1e-12, w @ self.cov @ w)))
        else:
            p = float(norm_p)
            return float(np.sum(np.abs(w) ** p) ** (1.0 / p))

    def _compute_norm_gradient(self, w: np.ndarray, norm_p: Union[int, float, str]) -> np.ndarray:
        """Computes analytical gradient of dual norm ||w||_p with respect to w."""
        eps_safe = 1e-8
        if norm_p == 2 or norm_p == 2.0 or norm_p == "l2":
            norm_val = np.linalg.norm(w, 2)
            if norm_val < eps_safe:
                return np.zeros_like(w)
            return w / norm_val
        elif norm_p == 1 or norm_p == 1.0 or norm_p == "l1":
            return np.sign(w)
        elif norm_p == "inf" or norm_p == np.inf or norm_p == "linf":
            # Soft approximation for inf-norm gradient using p=16
            p = 16.0
            norm_val = np.sum(np.abs(w) ** p) ** (1.0 / p)
            if norm_val < eps_safe:
                return np.zeros_like(w)
            return (np.sign(w) * (np.abs(w) ** (p - 1.0))) / (norm_val ** (p - 1.0))
        elif norm_p == "mahalanobis":
            quad = float(w @ self.cov @ w)
            if quad < eps_safe:
                return np.zeros_like(w)
            return (self.cov @ w) / np.sqrt(quad)
        else:
            p = float(norm_p)
            norm_val = np.sum(np.abs(w) ** p) ** (1.0 / p)
            if norm_val < eps_safe:
                return np.zeros_like(w)
            return (np.sign(w) * (np.abs(w) ** (p - 1.0))) / (norm_val ** (p - 1.0))

    def optimize(
        self,
        epsilon: float = 0.01,
        norm_p: Union[int, float, str] = 2,
        allow_short: bool = False,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        initial_weights: Optional[np.ndarray] = None,
    ) -> DROResult:
        """Solves the Wasserstein Distributionally Robust Optimization problem.
        
        Parameters
        ----------
        epsilon : float
            Wasserstein radius (uncertainty budget) >= 0.
        norm_p : int, float, or str
            Dual norm: 2 (Euclidean / shrinkage), 1 (sparsity), 'inf' (peak constraint), 'mahalanobis'.
        allow_short : bool
            Whether short positions (negative weights) are allowed.
        min_weight : float
            Lower bound per asset.
        max_weight : float
            Upper bound per asset.
        initial_weights : Optional[np.ndarray]
            Warm-start vector.
        """
        eps = max(0.0, float(epsilon))
        gamma = self.risk_aversion
        n = self.n_assets

        # Objective Function
        def objective(w: np.ndarray) -> float:
            nominal_return = float(np.dot(w, self.mu))
            nominal_var = float(np.dot(w, self.cov @ w))
            regularizer = eps * self._compute_norm(w, norm_p)
            return -nominal_return + 0.5 * gamma * nominal_var + regularizer

        # Analytical Jacobian
        def jacobian(w: np.ndarray) -> np.ndarray:
            grad_nom = -self.mu + gamma * (self.cov @ w)
            grad_reg = eps * self._compute_norm_gradient(w, norm_p)
            return grad_nom + grad_reg

        # Budget constraint: sum(w) = 1.0
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}
        ]

        # Bounds
        if allow_short:
            bounds = [(min_weight if min_weight < 0 else -1.0, max_weight) for _ in range(n)]
        else:
            bounds = [(max(0.0, min_weight), min(1.0, max_weight)) for _ in range(n)]

        # Initial point (equal-weight default)
        if initial_weights is not None:
            w0 = np.asarray(initial_weights, dtype=float)
            w0 = w0 / np.sum(w0)
        else:
            w0 = np.ones(n) / n

        # Solve via SLSQP
        res = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=jacobian,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9, "disp": False},
        )

        w_opt = res.x
        # Clean small numerical noise and normalize
        if not allow_short:
            w_opt = np.maximum(0.0, w_opt)
        w_opt = w_opt / np.sum(w_opt)

        # Compute performance and risk metrics
        exp_ret = float(np.dot(w_opt, self.mu))
        vol = float(np.sqrt(np.maximum(1e-12, np.dot(w_opt, self.cov @ w_opt))))
        sharpe = (exp_ret - self.risk_free_rate) / vol if vol > 1e-6 else 0.0

        nom_obj = -exp_ret + 0.5 * gamma * (vol ** 2)
        norm_penalty = eps * self._compute_norm(w_opt, norm_p)
        rob_obj = nom_obj + norm_penalty
        worst_case_loss = -exp_ret + norm_penalty

        hhi = float(np.sum(w_opt ** 2))
        eff_n = float(1.0 / hhi) if hhi > 1e-8 else 1.0

        weight_series = pd.Series(w_opt, index=self.asset_names, name="DRO_Weight")

        return DROResult(
            weights=weight_series,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            nominal_objective=nom_obj,
            robust_objective=rob_obj,
            worst_case_loss=worst_case_loss,
            epsilon=eps,
            norm_p=norm_p,
            risk_aversion=gamma,
            effective_n_assets=eff_n,
            herfindahl_index=hhi,
            converged=res.success,
            status_message=res.message,
        )

    def robust_efficient_frontier(
        self,
        epsilon: float = 0.01,
        norm_p: Union[int, float, str] = 2,
        n_points: int = 50,
        allow_short: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Calculates the Robust Efficient Frontier for a given Wasserstein radius eps."""
        gamma_values = np.logspace(-2, 3, n_points)
        returns = []
        volatilities = []
        sharpe_ratios = []
        robust_objectives = []
        weights_matrix = []

        orig_gamma = self.risk_aversion
        try:
            for g in gamma_values:
                self.risk_aversion = g
                sol = self.optimize(epsilon=epsilon, norm_p=norm_p, allow_short=allow_short)
                returns.append(sol.expected_return)
                volatilities.append(sol.volatility)
                sharpe_ratios.append(sol.sharpe_ratio)
                robust_objectives.append(sol.robust_objective)
                weights_matrix.append(sol.weights.values)
        finally:
            self.risk_aversion = orig_gamma

        return {
            "gammas": gamma_values,
            "returns": np.array(returns),
            "volatilities": np.array(volatilities),
            "sharpe_ratios": np.array(sharpe_ratios),
            "robust_objectives": np.array(robust_objectives),
            "weights": np.array(weights_matrix),
            "epsilon": epsilon,
        }

    def evaluate_ambiguity_sweep(
        self,
        epsilons: Optional[np.ndarray] = None,
        norm_p: Union[int, float, str] = 2,
    ) -> pd.DataFrame:
        """Sweeps across Wasserstein radii eps in [0, eps_max] to demonstrate regularizing shrinkage."""
        if epsilons is None:
            epsilons = np.linspace(0.0, 0.05, 11)

        records = []
        for eps in epsilons:
            res = self.optimize(epsilon=eps, norm_p=norm_p)
            records.append({
                "Epsilon": eps,
                "Exp_Return": res.expected_return,
                "Volatility": res.volatility,
                "Sharpe_Ratio": res.sharpe_ratio,
                "Nominal_Obj": res.nominal_objective,
                "Robust_Obj": res.robust_objective,
                "Effective_N": res.effective_n_assets,
                "HHI": res.herfindahl_index,
                "Max_Weight": res.weights.max(),
                "Min_Weight": res.weights.min(),
            })

        return pd.DataFrame(records)

    @staticmethod
    def estimate_wasserstein_radius(
        returns_df: pd.DataFrame,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        random_state: int = 42,
    ) -> float:
        """Estimates the empirical Wasserstein uncertainty radius eps using bootstrap concentration."""
        rng = np.random.default_rng(random_state)
        n_obs, n_dim = returns_df.shape
        X = returns_df.values

        # Bootstrap empirical distribution differences
        distances = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_obs, size=n_obs, replace=True)
            X_boot = X[idx]
            diff_mean = np.linalg.norm(np.mean(X, axis=0) - np.mean(X_boot, axis=0), 2)
            distances.append(diff_mean)

        eps_est = float(np.percentile(distances, confidence_level * 100))
        return max(1e-4, eps_est)

    @classmethod
    def out_of_sample_comparison(
        cls,
        train_returns: pd.DataFrame,
        test_returns: pd.DataFrame,
        epsilon: float = 0.015,
        risk_aversion: float = 1.0,
        risk_free_rate: float = 0.02,
    ) -> pd.DataFrame:
        """Runs comprehensive out-of-sample benchmark comparison:
        1. Nominal Mean-Variance (Sample Average Approximation - SAA)
        2. Equal-Weight (1/N Heuristic)
        3. Ledoit-Wolf Covariance Shrinkage Mean-Variance
        4. Wasserstein Distributionally Robust Portfolio Optimization (DRO)
        """
        train_df = train_returns.dropna()
        test_df = test_returns.dropna()
        assets = list(train_df.columns)
        n = len(assets)

        # 1. Equal-Weight
        w_eq = np.ones(n) / n

        # 2. Nominal SAA Mean-Variance (eps = 0)
        opt_nominal = cls(returns_data=train_df, risk_aversion=risk_aversion, risk_free_rate=risk_free_rate)
        res_nominal = opt_nominal.optimize(epsilon=0.0, norm_p=2)
        w_nom = res_nominal.weights.values

        # 3. Ledoit-Wolf Shrinkage
        from scipy.spatial.distance import cdist
        cov_sample = train_df.cov().values * 252
        mean_ret = train_df.mean().values * 252
        # Simple shrinkage target: diagonal variance
        shrink_target = np.diag(np.diag(cov_sample))
        delta_shrink = 0.25
        cov_shrunk = (1 - delta_shrink) * cov_sample + delta_shrink * shrink_target
        opt_shrunk = cls(expected_returns=mean_ret, cov_matrix=cov_shrunk, risk_aversion=risk_aversion, risk_free_rate=risk_free_rate, asset_names=assets)
        res_shrunk = opt_shrunk.optimize(epsilon=0.0, norm_p=2)
        w_shrunk = res_shrunk.weights.values

        # 4. Wasserstein DRO (eps > 0)
        opt_dro = cls(returns_data=train_df, risk_aversion=risk_aversion, risk_free_rate=risk_free_rate)
        res_dro = opt_dro.optimize(epsilon=epsilon, norm_p=2)
        w_dro = res_dro.weights.values

        strategies = {
            "1/N Equal Weight": w_eq,
            "Nominal Markowitz SAA": w_nom,
            "Ledoit-Wolf Shrinkage": w_shrunk,
            "Wasserstein DRO (Robust)": w_dro,
        }

        records = []
        for name, w in strategies.items():
            port_ret_series = test_df.dot(w)
            cagr = float(np.prod(1.0 + port_ret_series) ** (252.0 / len(port_ret_series)) - 1.0)
            vol_ann = float(port_ret_series.std() * np.sqrt(252))
            sharpe = (cagr - risk_free_rate) / vol_ann if vol_ann > 1e-6 else 0.0

            # Drawdown
            cum = (1.0 + port_ret_series).cumprod()
            peak = cum.cummax()
            dd = (cum - peak) / peak
            max_dd = float(dd.min())

            hhi = float(np.sum(w ** 2))
            eff_n = float(1.0 / hhi) if hhi > 1e-8 else 1.0

            records.append({
                "Strategy": name,
                "OOS_CAGR": cagr,
                "OOS_Volatility": vol_ann,
                "OOS_Sharpe": sharpe,
                "OOS_Max_Drawdown": max_dd,
                "Effective_N_Assets": eff_n,
                "Max_Asset_Weight": float(np.max(w)),
            })

        return pd.DataFrame(records)
