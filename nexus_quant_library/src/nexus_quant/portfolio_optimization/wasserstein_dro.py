"""
Portfolio Optimization: Wasserstein Distributionally Robust Optimization (DRO).
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class WassersteinDROOptimizer:
    """
    Solves distributionally robust portfolio optimization under a 1-Wasserstein ambiguity ball:
    min_{w in W} [ -w^T mu + (gamma/2) w^T Sigma w + epsilon * ||w||_2 ]
    Subject to sum(w) = 1, w >= 0.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.5,
        risk_free_rate: float = 0.02,
        asset_names: Optional[List[str]] = None,
    ):
        self.mu = np.asarray(expected_returns, dtype=float)
        self.cov = np.asarray(cov_matrix, dtype=float)
        self.gamma = risk_aversion
        self.rf = risk_free_rate
        self.n_assets = len(self.mu)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]

    def optimize(self, epsilon: float = 0.015) -> Dict[str, Any]:
        """Solve regularized convex dual program for a given ambiguity radius epsilon."""
        def obj(w):
            exp_loss = -float(w @ self.mu)
            var_pen = 0.5 * self.gamma * float(w @ self.cov @ w)
            l2_reg = epsilon * float(np.linalg.norm(w, 2))
            return exp_loss + var_pen + l2_reg

        def grad(w):
            norm_w = np.linalg.norm(w, 2)
            reg_grad = (epsilon * w / norm_w) if norm_w > 1e-8 else np.zeros_like(w)
            return -self.mu + self.gamma * (self.cov @ w) + reg_grad

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]
        bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
        w0 = np.full(self.n_assets, 1.0 / self.n_assets)

        res = minimize(obj, w0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP", options={"ftol": 1e-9})
        w_opt = res.x

        ret_opt = float(w_opt @ self.mu)
        vol_opt = float(np.sqrt(w_opt @ self.cov @ w_opt))
        sharpe = float((ret_opt - self.rf) / max(vol_opt, 1e-6))
        eff_n = float(1.0 / np.sum(w_opt**2)) # 1/HHI diversification

        return {
            "weights": w_opt,
            "expected_return": ret_opt,
            "volatility": vol_opt,
            "sharpe_ratio": sharpe,
            "effective_n_assets": eff_n,
            "robust_objective": float(res.fun),
            "epsilon": epsilon,
        }

    def ambiguity_sweep(self, epsilons: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Evaluate weight shrinkage and diversification as epsilon increases."""
        if epsilons is None:
            epsilons = np.linspace(0.0, 0.05, 11)

        records = []
        for eps in epsilons:
            sol = self.optimize(epsilon=eps)
            records.append({
                "Epsilon": eps,
                "Return": sol["expected_return"],
                "Volatility": sol["volatility"],
                "Sharpe": sol["sharpe_ratio"],
                "Effective_N": sol["effective_n_assets"],
                "Max_Weight": float(np.max(sol["weights"])),
            })
        return pd.DataFrame(records)
