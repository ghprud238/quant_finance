"""Mean-Variance Portfolio Optimization & Efficient Frontier Engine.

Formulates and solves Markowitz Modern Portfolio Theory (MPT) optimizations:
- Minimum Volatility Portfolio
- Maximum Sharpe Ratio (Tangency) Portfolio
- Target Return & Target Volatility Portfolios
- Efficient Frontier Curve Generator
- Random Portfolio Monte Carlo Simulation Generator
- Capital Allocation Line (CAL)

Author: Quant Risk & Portfolio Analytics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class OptimizationResult:
    """Structured container for portfolio optimization results."""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str = ""
    optimization_type: str = ""
    risk_free_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to a clean dictionary."""
        return {
            "optimization_type": self.optimization_type,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_free_rate": self.risk_free_rate,
            "success": self.success,
            "message": self.message,
            "weights": self.weights.to_dict(),
        }

    def summary(self) -> str:
        """Returns a formatted string summary of the optimized portfolio."""
        lines = [
            f"=== Portfolio Optimization Result: {self.optimization_type} ===",
            f"  Status:          {'SUCCESS' if self.success else 'FAILED'} ({self.message})",
            f"  Expected Return: {self.expected_return:+.2%}",
            f"  Volatility (Ann):{self.volatility:.2%}",
            f"  Sharpe Ratio:    {self.sharpe_ratio:.2f} (Rf = {self.risk_free_rate:.1%})",
            "  Asset Allocations:",
        ]
        for asset, w in self.weights.items():
            if abs(w) >= 1e-4:
                lines.append(f"    - {asset:<12}: {w:6.2%}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(type='{self.optimization_type}', return={self.expected_return:.2%}, "
            f"vol={self.volatility:.2%}, sharpe={self.sharpe_ratio:.2f}, success={self.success})"
        )


@dataclass
class EfficientFrontierResult:
    """Container for the Efficient Frontier curve generation results."""
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights: pd.DataFrame
    min_vol_portfolio: OptimizationResult
    max_sharpe_portfolio: OptimizationResult
    risk_free_rate: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the efficient frontier points as a DataFrame."""
        df = pd.DataFrame(
            {
                "Return": self.returns,
                "Volatility": self.volatilities,
                "Sharpe_Ratio": self.sharpe_ratios,
            }
        )
        return pd.concat([df, self.weights.reset_index(drop=True)], axis=1)

    def summary(self) -> str:
        """Returns summary text for the generated efficient frontier."""
        lines = [
            "=== Efficient Frontier Summary ===",
            f"  Number of Frontier Points: {len(self.returns)}",
            f"  Min Volatility Point:      Vol = {self.min_vol_portfolio.volatility:.2%}, Return = {self.min_vol_portfolio.expected_return:+.2%}",
            f"  Max Sharpe Ratio Point:    Sharpe = {self.max_sharpe_portfolio.sharpe_ratio:.2f} (Vol = {self.max_sharpe_portfolio.volatility:.2%}, Return = {self.max_sharpe_portfolio.expected_return:+.2%})",
        ]
        return "\n".join(lines)


@dataclass
class SimulatedPortfoliosResult:
    """Container for Monte Carlo simulated random portfolios."""
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights: np.ndarray
    asset_names: List[str]
    risk_free_rate: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Returns simulated portfolios as a pandas DataFrame."""
        df = pd.DataFrame(
            {
                "Return": self.returns,
                "Volatility": self.volatilities,
                "Sharpe_Ratio": self.sharpe_ratios,
            }
        )
        w_df = pd.DataFrame(self.weights, columns=self.asset_names)
        return pd.concat([df, w_df], axis=1)


class MeanVarianceOptimizer:
    """Markowitz Mean-Variance Portfolio Optimizer & Efficient Frontier Engine.

    Formulates and solves standard and constrained portfolio optimizations:
    1. Global Minimum Volatility Portfolio
    2. Tangency / Maximum Sharpe Ratio Portfolio
    3. Target Return Portfolio (Minimum variance for target return)
    4. Target Volatility Portfolio (Maximum return for target risk)
    5. Continuous Efficient Frontier Curve
    6. Random Portfolio Simulation Cloud
    7. Capital Allocation Line (CAL)

    Parameters
    ----------
    expected_returns : Union[pd.Series, np.ndarray, Sequence[float]], optional
        Annualized expected returns for each asset.
    cov_matrix : Union[pd.DataFrame, np.ndarray], optional
        Annualized covariance matrix of asset returns.
    returns_df : pd.DataFrame, optional
        Historical asset returns DataFrame (alternative to expected_returns & cov_matrix).
    risk_free_rate : float, default=0.0
        Annualized risk-free rate for Sharpe ratio and CAL calculations.
    weight_bounds : Union[Tuple[float, float], Sequence[Tuple[float, float]]], default=(0.0, 1.0)
        Lower and upper bounds for asset weights. Default is long-only (0.0, 1.0).
    asset_names : Sequence[str], optional
        Names of the assets if inputs are raw numpy arrays.
    periods_per_year : int, default=252
        Number of trading periods per year for annualization.
    """

    def __init__(
        self,
        expected_returns: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        weight_bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (0.0, 1.0),
        asset_names: Optional[Sequence[str]] = None,
        periods_per_year: int = 252,
    ) -> None:
        self.periods_per_year = periods_per_year
        self.risk_free_rate = float(risk_free_rate)

        # Parse inputs
        if returns_df is not None:
            if not isinstance(returns_df, pd.DataFrame):
                returns_df = pd.DataFrame(returns_df)
            clean_df = returns_df.dropna()
            self.expected_returns = clean_df.mean() * periods_per_year
            self.cov_matrix = clean_df.cov() * periods_per_year
            self.asset_names = list(clean_df.columns)
        elif expected_returns is not None and cov_matrix is not None:
            if isinstance(expected_returns, pd.Series):
                self.asset_names = list(expected_returns.index)
                self.expected_returns = expected_returns.copy()
            elif asset_names is not None:
                self.asset_names = list(asset_names)
                self.expected_returns = pd.Series(expected_returns, index=self.asset_names)
            else:
                self.asset_names = [f"Asset_{i}" for i in range(len(expected_returns))]
                self.expected_returns = pd.Series(expected_returns, index=self.asset_names)

            if isinstance(cov_matrix, pd.DataFrame):
                self.cov_matrix = cov_matrix.copy()
            else:
                self.cov_matrix = pd.DataFrame(cov_matrix, index=self.asset_names, columns=self.asset_names)
        else:
            raise ValueError("Must provide either (expected_returns, cov_matrix) or returns_df.")

        # Validate dimensions
        self.n_assets = len(self.asset_names)
        if self.cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Covariance matrix shape {self.cov_matrix.shape} does not match number of assets ({self.n_assets})."
            )

        # Set bounds
        if isinstance(weight_bounds, tuple) and len(weight_bounds) == 2 and isinstance(weight_bounds[0], (int, float)):
            self.bounds = [weight_bounds for _ in range(self.n_assets)]
        elif isinstance(weight_bounds, (list, tuple)) and len(weight_bounds) == self.n_assets:
            self.bounds = [tuple(b) for b in weight_bounds]
        else:
            raise ValueError(f"Invalid weight_bounds. Must be (min, max) or sequence of length {self.n_assets}.")

        self._mu = self.expected_returns.values.astype(np.float64)
        self._sigma = self.cov_matrix.values.astype(np.float64)

        # Numerical safety: ensure positive semi-definiteness / symmetry
        self._sigma = 0.5 * (self._sigma + self._sigma.T)
        eigvals = np.linalg.eigvalsh(self._sigma)
        if np.any(eigvals < -1e-8):
            min_eig = np.min(eigvals)
            self._sigma += (abs(min_eig) + 1e-6) * np.eye(self.n_assets)

    def portfolio_performance(
        self,
        weights: Union[np.ndarray, pd.Series, Sequence[float]],
        risk_free_rate: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Calculates expected return, volatility, and Sharpe ratio for given weights.

        Parameters
        ----------
        weights : array-like
            Asset allocation weights.
        risk_free_rate : float, optional
            Risk-free rate override. Defaults to self.risk_free_rate.

        Returns
        -------
        Tuple[float, float, float]
            (expected_return, volatility, sharpe_ratio)
        """
        rf = self.risk_free_rate if risk_free_rate is None else float(risk_free_rate)
        w = np.asarray(weights, dtype=np.float64).flatten()
        if len(w) != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} weights, received {len(w)}.")

        port_return = float(np.dot(w, self._mu))
        port_variance = float(np.dot(w, np.dot(self._sigma, w)))
        port_volatility = float(np.sqrt(max(port_variance, 0.0)))

        sharpe = (port_return - rf) / (port_volatility + 1e-12) if port_volatility > 0 else 0.0
        return port_return, port_volatility, sharpe

    def _initial_guesses(self) -> List[np.ndarray]:
        """Generates a suite of diverse initial weight guesses for numerical optimization."""
        guesses = []
        # 1. Equal weight
        guesses.append(np.ones(self.n_assets) / self.n_assets)

        # 2. Inverse volatility
        diag_vol = np.sqrt(np.maximum(np.diag(self._sigma), 1e-8))
        inv_vol = 1.0 / diag_vol
        guesses.append(inv_vol / np.sum(inv_vol))

        # 3. Individual asset one-hot allocations
        for i in range(self.n_assets):
            w = np.zeros(self.n_assets)
            w[i] = 1.0
            guesses.append(w)

        # 4. Maximum return asset focus
        max_ret_idx = np.argmax(self._mu)
        w_max = np.zeros(self.n_assets)
        w_max[max_ret_idx] = 1.0
        guesses.append(w_max)

        return guesses

    def min_volatility(
        self,
        target_return: Optional[float] = None,
        custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Finds the Global Minimum Volatility Portfolio, or the minimum volatility for a target return.

        Optimization Formulation:
            min_w  w^T Sigma w
            s.t.   sum(w_i) = 1
                   l_i <= w_i <= u_i
                   w^T mu >= target_return (if target_return is provided)

        Parameters
        ----------
        target_return : float, optional
            Minimum target expected return constraint.
        custom_bounds : Sequence[Tuple[float, float]], optional
            Optional bounds override for this optimization.

        Returns
        -------
        OptimizationResult
            Optimized portfolio result.
        """
        bounds = custom_bounds if custom_bounds is not None else self.bounds

        def objective(w: np.ndarray) -> float:
            return float(np.dot(w, np.dot(self._sigma, w)))

        def jacobian(w: np.ndarray) -> np.ndarray:
            return 2.0 * np.dot(self._sigma, w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]

        if target_return is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: np.dot(w, self._mu) - target_return,
                    "jac": lambda w: self._mu,
                }
            )

        best_opt = None
        best_fun = float("inf")

        for x0 in self._initial_guesses():
            x0_proj = np.array([np.clip(x0[i], bounds[i][0], bounds[i][1]) for i in range(self.n_assets)])
            if np.sum(x0_proj) > 0:
                x0_proj = x0_proj / np.sum(x0_proj)
            else:
                x0_proj = np.ones(self.n_assets) / self.n_assets

            opt = optimize.minimize(
                objective,
                x0_proj,
                method="SLSQP",
                jac=jacobian,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500},
            )
            if opt.success and opt.fun < best_fun:
                best_opt = opt
                best_fun = opt.fun

        if best_opt is None:
            best_opt = optimize.minimize(
                objective,
                np.ones(self.n_assets) / self.n_assets,
                method="SLSQP",
                jac=jacobian,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )

        w_opt = best_opt.x
        w_opt = np.where(np.abs(w_opt) < 1e-7, 0.0, w_opt)
        w_opt = w_opt / np.sum(w_opt)

        ret, vol, sr = self.portfolio_performance(w_opt)
        weights_series = pd.Series(w_opt, index=self.asset_names, name="weights")

        opt_type = "Target Return Minimum Volatility" if target_return is not None else "Global Minimum Volatility"
        return OptimizationResult(
            weights=weights_series,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sr,
            success=bool(best_opt.success),
            message=str(best_opt.message),
            optimization_type=opt_type,
            risk_free_rate=self.risk_free_rate,
        )

    def max_sharpe_ratio(
        self,
        risk_free_rate: Optional[float] = None,
        custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Finds the Maximum Sharpe Ratio (Tangency) Portfolio.

        Optimization Formulation:
            max_w  (w^T mu - Rf) / sqrt(w^T Sigma w)
            s.t.   sum(w_i) = 1
                   l_i <= w_i <= u_i

        Parameters
        ----------
        risk_free_rate : float, optional
            Risk-free rate override. Defaults to self.risk_free_rate.
        custom_bounds : Sequence[Tuple[float, float]], optional
            Optional bounds override.

        Returns
        -------
        OptimizationResult
            Tangency portfolio result.
        """
        rf = self.risk_free_rate if risk_free_rate is None else float(risk_free_rate)
        bounds = custom_bounds if custom_bounds is not None else self.bounds

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = np.dot(w, self._mu)
            port_var = np.dot(w, np.dot(self._sigma, w))
            port_vol = np.sqrt(max(port_var, 1e-12))
            return -float((port_ret - rf) / port_vol)

        def neg_sharpe_jac(w: np.ndarray) -> np.ndarray:
            port_ret = np.dot(w, self._mu)
            port_var = np.dot(w, np.dot(self._sigma, w))
            port_vol = np.sqrt(max(port_var, 1e-12))
            sigma_w = np.dot(self._sigma, w)
            grad = - (self._mu / port_vol - ((port_ret - rf) * sigma_w) / (port_vol**3 + 1e-12))
            return grad

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]

        best_opt = None
        best_fun = float("inf")

        for x0 in self._initial_guesses():
            x0_proj = np.array([np.clip(x0[i], bounds[i][0], bounds[i][1]) for i in range(self.n_assets)])
            if np.sum(x0_proj) > 0:
                x0_proj = x0_proj / np.sum(x0_proj)
            else:
                x0_proj = np.ones(self.n_assets) / self.n_assets

            opt = optimize.minimize(
                neg_sharpe,
                x0_proj,
                method="SLSQP",
                jac=neg_sharpe_jac,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500},
            )
            if opt.success and opt.fun < best_fun:
                best_opt = opt
                best_fun = opt.fun

        if best_opt is None:
            best_opt = optimize.minimize(
                neg_sharpe,
                np.ones(self.n_assets) / self.n_assets,
                method="SLSQP",
                jac=neg_sharpe_jac,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )

        w_opt = best_opt.x
        w_opt = np.where(np.abs(w_opt) < 1e-7, 0.0, w_opt)
        w_opt = w_opt / np.sum(w_opt)

        ret, vol, sr = self.portfolio_performance(w_opt, risk_free_rate=rf)
        weights_series = pd.Series(w_opt, index=self.asset_names, name="weights")

        return OptimizationResult(
            weights=weights_series,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sr,
            success=bool(best_opt.success),
            message=str(best_opt.message),
            optimization_type="Maximum Sharpe Ratio (Tangency)",
            risk_free_rate=rf,
        )

    def efficient_return(
        self,
        target_return: float,
        custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Finds the minimum volatility portfolio that achieves at least target_return.

        Parameters
        ----------
        target_return : float
            Desired minimum expected return.
        custom_bounds : Sequence[Tuple[float, float]], optional
            Optional bounds override.

        Returns
        -------
        OptimizationResult
            Optimized target return portfolio.
        """
        return self.min_volatility(target_return=target_return, custom_bounds=custom_bounds)

    def efficient_risk(
        self,
        target_volatility: float,
        custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Finds the maximum return portfolio subject to a target volatility cap.

        Optimization Formulation:
            max_w  w^T mu
            s.t.   sqrt(w^T Sigma w) <= target_volatility
                   sum(w_i) = 1
                   l_i <= w_i <= u_i

        Parameters
        ----------
        target_volatility : float
            Maximum allowable portfolio volatility.
        custom_bounds : Sequence[Tuple[float, float]], optional
            Optional bounds override.

        Returns
        -------
        OptimizationResult
            Target volatility portfolio result.
        """
        bounds = custom_bounds if custom_bounds is not None else self.bounds

        def objective(w: np.ndarray) -> float:
            return -float(np.dot(w, self._mu))

        def jacobian(w: np.ndarray) -> np.ndarray:
            return -self._mu

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},
            {
                "type": "ineq",
                "fun": lambda w: target_volatility**2 - np.dot(w, np.dot(self._sigma, w)),
                "jac": lambda w: -2.0 * np.dot(self._sigma, w),
            },
        ]

        best_opt = None
        best_fun = float("inf")

        for x0 in self._initial_guesses():
            x0_proj = np.array([np.clip(x0[i], bounds[i][0], bounds[i][1]) for i in range(self.n_assets)])
            if np.sum(x0_proj) > 0:
                x0_proj = x0_proj / np.sum(x0_proj)
            else:
                x0_proj = np.ones(self.n_assets) / self.n_assets

            opt = optimize.minimize(
                objective,
                x0_proj,
                method="SLSQP",
                jac=jacobian,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500},
            )
            if opt.success and opt.fun < best_fun:
                best_opt = opt
                best_fun = opt.fun

        if best_opt is None:
            best_opt = optimize.minimize(
                objective,
                np.ones(self.n_assets) / self.n_assets,
                method="SLSQP",
                jac=jacobian,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )

        w_opt = best_opt.x
        w_opt = np.where(np.abs(w_opt) < 1e-7, 0.0, w_opt)
        w_opt = w_opt / np.sum(w_opt)

        ret, vol, sr = self.portfolio_performance(w_opt)
        weights_series = pd.Series(w_opt, index=self.asset_names, name="weights")

        return OptimizationResult(
            weights=weights_series,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sr,
            success=bool(best_opt.success),
            message=str(best_opt.message),
            optimization_type="Target Volatility Maximum Return",
            risk_free_rate=self.risk_free_rate,
        )

    def efficient_frontier(
        self,
        n_points: int = 50,
        risk_free_rate: Optional[float] = None,
        custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> EfficientFrontierResult:
        """Generates the complete Efficient Frontier curve from minimum variance to maximum return.

        Parameters
        ----------
        n_points : int, default=50
            Number of optimal frontier points to calculate.
        risk_free_rate : float, optional
            Risk-free rate override. Defaults to self.risk_free_rate.
        custom_bounds : Sequence[Tuple[float, float]], optional
            Optional bounds override.

        Returns
        -------
        EfficientFrontierResult
            Container with arrays of returns, volatilities, Sharpe ratios, and weights DataFrame.
        """
        rf = self.risk_free_rate if risk_free_rate is None else float(risk_free_rate)
        bounds = custom_bounds if custom_bounds is not None else self.bounds

        # 1. Anchor portfolio 1: Global Minimum Volatility
        min_vol_port = self.min_volatility(custom_bounds=bounds)

        # 2. Anchor portfolio 2: Tangency (Max Sharpe)
        max_sharpe_port = self.max_sharpe_ratio(risk_free_rate=rf, custom_bounds=bounds)

        # Determine target return spectrum
        min_ret = min_vol_port.expected_return

        # Maximum possible return under given bounds
        max_ret = float(np.max(self._mu))
        for i, b in enumerate(bounds):
            if b[1] == 1.0 and self._mu[i] == max_ret:
                break
        else:
            # Solve linear program for maximum return under bounds
            res_max = optimize.linprog(
                c=-self._mu,
                A_eq=np.ones((1, self.n_assets)),
                b_eq=[1.0],
                bounds=bounds,
                method="highs",
            )
            if res_max.success:
                max_ret = -res_max.fun

        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_returns: List[float] = []
        frontier_vols: List[float] = []
        frontier_sharpes: List[float] = []
        frontier_weights: List[np.ndarray] = []

        for r in target_returns:
            res = self.min_volatility(target_return=r, custom_bounds=bounds)
            if res.success or len(frontier_returns) == 0:
                frontier_returns.append(res.expected_return)
                frontier_vols.append(res.volatility)
                sr = (res.expected_return - rf) / (res.volatility + 1e-12)
                frontier_sharpes.append(sr)
                frontier_weights.append(res.weights.values)

        w_df = pd.DataFrame(frontier_weights, columns=self.asset_names, index=np.round(frontier_returns, 4))
        w_df.index.name = "Target_Return"

        return EfficientFrontierResult(
            returns=np.array(frontier_returns),
            volatilities=np.array(frontier_vols),
            sharpe_ratios=np.array(frontier_sharpes),
            weights=w_df,
            min_vol_portfolio=min_vol_port,
            max_sharpe_portfolio=max_sharpe_port,
            risk_free_rate=rf,
        )

    def simulate_random_portfolios(
        self,
        n_portfolios: int = 5000,
        risk_free_rate: Optional[float] = None,
        seed: Optional[int] = 42,
    ) -> SimulatedPortfoliosResult:
        """Generates random long-only portfolios via Dirichlet distribution for MPT cloud visualization.

        Parameters
        ----------
        n_portfolios : int, default=5000
            Number of random portfolios to generate.
        risk_free_rate : float, optional
            Risk-free rate for Sharpe ratio calculation.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        SimulatedPortfoliosResult
            Container with simulated portfolio returns, volatilities, and Sharpe ratios.
        """
        rf = self.risk_free_rate if risk_free_rate is None else float(risk_free_rate)
        if seed is not None:
            np.random.seed(seed)

        # Generate Dirichlet random weights on unit simplex sum(w)=1, w >= 0
        w_random = np.random.dirichlet(np.ones(self.n_assets), size=n_portfolios)

        # Also add one-hot single asset portfolios and equal-weight to ensure boundary inclusion
        w_single = np.eye(self.n_assets)
        w_eq = np.ones((1, self.n_assets)) / self.n_assets
        all_weights = np.vstack([w_random, w_single, w_eq])

        # Vectorized returns: W * mu (N x 1)
        rets = np.dot(all_weights, self._mu)

        # Vectorized volatilities: sqrt(sum((W * Sigma) * W, axis=1))
        w_sigma = np.dot(all_weights, self._sigma)
        vars_ = np.sum(w_sigma * all_weights, axis=1)
        vols = np.sqrt(np.maximum(vars_, 0.0))

        # Sharpe ratios
        sharpes = (rets - rf) / (vols + 1e-12)

        return SimulatedPortfoliosResult(
            returns=rets,
            volatilities=vols,
            sharpe_ratios=sharpes,
            weights=all_weights,
            asset_names=self.asset_names,
            risk_free_rate=rf,
        )

    def capital_allocation_line(
        self,
        n_points: int = 50,
        max_vol: Optional[float] = None,
        risk_free_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculates the Capital Allocation Line (CAL) connecting (0, Rf) through the Tangency portfolio.

        Equation:
            E[R_p] = R_f + Sharpe_tangency * Vol_p

        Parameters
        ----------
        n_points : int, default=50
            Number of points along the line.
        max_vol : float, optional
            Upper bound of volatility for the line. Defaults to 1.5 * Tangency Volatility.
        risk_free_rate : float, optional
            Risk-free rate override.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'volatilities' and 'returns' arrays along the CAL.
        """
        rf = self.risk_free_rate if risk_free_rate is None else float(risk_free_rate)
        tangency = self.max_sharpe_ratio(risk_free_rate=rf)

        tangency_vol = tangency.volatility
        tangency_sr = tangency.sharpe_ratio

        if max_vol is None:
            max_vol = float(tangency_vol * 1.6)

        vols = np.linspace(0.0, max_vol, n_points)
        rets = rf + tangency_sr * vols

        return {
            "volatilities": vols,
            "returns": rets,
            "tangency_portfolio": tangency.to_dict(),
            "risk_free_rate": rf,
            "sharpe_ratio": tangency_sr,
        }
