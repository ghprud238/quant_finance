"""
Portfolio Optimization: Modern Portfolio Theory (Markowitz MPT) & Efficient Frontier.
"""

from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimalPortfolio:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    asset_names: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"Asset": self.asset_names, "Weight": self.weights})
        return df


class MarkowitzOptimizer:
    """Mean-Variance Portfolio Optimizer solving constrained quadratic programs."""

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[List[str]] = None,
        risk_free_rate: float = 0.02,
    ):
        self.mu = np.asarray(expected_returns, dtype=float)
        self.cov = np.asarray(cov_matrix, dtype=float)
        self.n_assets = len(self.mu)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]
        self.rf = risk_free_rate

    def min_volatility(self) -> OptimalPortfolio:
        """Global Minimum Volatility portfolio: min w^T Sigma w s.t. sum(w)=1, w >= 0."""
        def obj(w):
            return float(w @ self.cov @ w)

        def grad(w):
            return 2.0 * self.cov @ w

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]
        bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
        w0 = np.full(self.n_assets, 1.0 / self.n_assets)

        res = minimize(obj, w0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP", options={"ftol": 1e-9})
        w_opt = res.x
        ret_opt = float(w_opt @ self.mu)
        vol_opt = float(np.sqrt(w_opt @ self.cov @ w_opt))
        sharpe = float((ret_opt - self.rf) / max(vol_opt, 1e-6))

        return OptimalPortfolio(w_opt, ret_opt, vol_opt, sharpe, self.asset_names)

    def max_sharpe_ratio(self) -> OptimalPortfolio:
        """Tangency portfolio maximizing Sharpe Ratio."""
        def obj(w):
            r = float(w @ self.mu)
            vol = float(np.sqrt(w @ self.cov @ w))
            return -(r - self.rf) / max(vol, 1e-8)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
        w0 = np.full(self.n_assets, 1.0 / self.n_assets)

        res = minimize(obj, w0, bounds=bounds, constraints=cons, method="SLSQP", options={"ftol": 1e-9})
        w_opt = res.x
        ret_opt = float(w_opt @ self.mu)
        vol_opt = float(np.sqrt(w_opt @ self.cov @ w_opt))
        sharpe = float((ret_opt - self.rf) / max(vol_opt, 1e-6))

        return OptimalPortfolio(w_opt, ret_opt, vol_opt, sharpe, self.asset_names)

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Compute n_points optimal frontier portfolios spanning min-vol to max-return."""
        min_vol_p = self.min_volatility()
        max_ret = float(np.max(self.mu))
        target_returns = np.linspace(min_vol_p.expected_return, max_ret, n_points)

        frontier_records = []
        bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
        w0 = min_vol_p.weights

        for target_r in target_returns:
            def obj(w):
                return float(w @ self.cov @ w)
            def grad(w):
                return 2.0 * self.cov @ w

            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},
                {"type": "eq", "fun": lambda w, tr=target_r: (w @ self.mu) - tr, "jac": lambda w: self.mu},
            ]
            res = minimize(obj, w0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP", options={"ftol": 1e-8})
            if res.success:
                w = res.x
                vol = float(np.sqrt(w @ self.cov @ w))
                sharpe = float((target_r - self.rf) / max(vol, 1e-6))
                frontier_records.append({"Return": target_r, "Volatility": vol, "Sharpe": sharpe})

        return pd.DataFrame(frontier_records)
