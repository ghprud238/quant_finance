"""
Portfolio Optimization: Equal Risk Contribution (ERC / Risk Parity).
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParityOptimizer:
    """
    Equal Risk Contribution (Risk Parity) portfolio solver using Spinu (2013) log-barrier formulation:
    min_x [ 0.5 * x^T Sigma x - (1/N) * sum(ln(x_i)) ], w = x / sum(x_i).
    """

    def __init__(self, cov_matrix: np.ndarray, asset_names: Optional[List[str]] = None):
        self.cov = np.asarray(cov_matrix, dtype=float)
        self.n_assets = self.cov.shape[0]
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for risk parity weights and percentage risk contributions.
        Returns: (weights, risk_contributions_pct).
        """
        def obj(x):
            var_term = 0.5 * float(x @ self.cov @ x)
            log_term = (1.0 / self.n_assets) * np.sum(np.log(np.maximum(x, 1e-8)))
            return var_term - log_term

        def grad(x):
            return (self.cov @ x) - (1.0 / self.n_assets) / np.maximum(x, 1e-8)

        bounds = [(1e-6, None) for _ in range(self.n_assets)]
        x0 = np.full(self.n_assets, 1.0 / np.sqrt(np.diag(self.cov)))

        res = minimize(obj, x0, jac=grad, bounds=bounds, method="L-BFGS-B")
        x_opt = res.x
        w_opt = x_opt / np.sum(x_opt)

        # Risk contributions: RC_i = w_i * (Sigma @ w)_i
        total_var = float(w_opt @ self.cov @ w_opt)
        marginal_risk = self.cov @ w_opt
        rc = w_opt * marginal_risk
        rc_pct = rc / total_var

        return w_opt, rc_pct
