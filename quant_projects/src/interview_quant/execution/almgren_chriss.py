"""Almgren-Chriss Optimal Execution Framework (2000).

Computes closed-form trading trajectories minimizing expected execution shortfall and market risk variance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


@dataclass
class ExecutionTrajectoryResult:
    """Structured result containing the optimal Almgren-Chriss trading trajectory."""
    time_grid: np.ndarray             # Discrete timestamps t_j in [0, T]
    holdings: np.ndarray              # Remaining inventory x_j at each interval
    trade_sizes: np.ndarray           # Trade volume n_j = x_{j-1} - x_j executed in interval j
    trading_rates: np.ndarray         # Trading speed v_j = n_j / tau
    expected_shortfall: float         # E[x] expected cost of execution in currency
    variance_shortfall: float         # V[x] variance of execution cost
    std_shortfall: float              # sqrt(V[x]) standard deviation in currency
    utility: float                    # E[x] + lambda * V[x]
    half_life: float                  # 1 / kappa time to liquidate ~63.2% of position
    kappa: float                      # Decay curvature parameter
    params: Dict[str, float]

    def to_dataframe(self) -> pd.DataFrame:
        """Formats the trajectory as a pandas DataFrame."""
        n_steps = len(self.holdings)
        rows = []
        for j in range(n_steps):
            t = self.time_grid[j]
            x = self.holdings[j]
            n = self.trade_sizes[j] if j < len(self.trade_sizes) else 0.0
            v = self.trading_rates[j] if j < len(self.trading_rates) else 0.0
            rows.append({
                'Step': j,
                'Time': t,
                'Inventory': x,
                'Trade_Size': n,
                'Trading_Rate': v,
                'Pct_Remaining': x / self.params['total_shares'] if self.params['total_shares'] != 0 else 0.0,
            })
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        """Summary dictionary of key execution metrics."""
        return {
            'Total Shares': self.params['total_shares'],
            'Horizon (T)': self.params['horizon'],
            'Intervals (N)': self.params['n_intervals'],
            'Risk Aversion (lambda)': self.params['risk_aversion'],
            'Kappa': self.kappa,
            'Execution Half-Life': self.half_life,
            'Expected Shortfall ($)': self.expected_shortfall,
            'Shortfall Std Dev ($)': self.std_shortfall,
            'Utility ($)': self.utility,
        }


@dataclass
class ExecutionFrontierResult:
    """Efficient Frontier of Execution: Expected Cost vs Risk Variance across risk aversion lambda."""
    lambda_values: np.ndarray
    expected_shortfalls: np.ndarray
    std_shortfalls: np.ndarray
    variances: np.ndarray
    kappas: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Returns tabular efficient frontier."""
        return pd.DataFrame({
            'Risk_Aversion_Lambda': self.lambda_values,
            'Expected_Cost': self.expected_shortfalls,
            'Cost_Std_Dev': self.std_shortfalls,
            'Cost_Variance': self.variances,
            'Kappa': self.kappas,
        })


class AlmgrenChrissModel:
    """Almgren-Chriss (2000) Optimal Execution Framework.

    Solves the multi-period trade-off between market impact (delay cost / liquidity friction)
    and price volatility risk (holding risk):
        min_x E[x] + lambda * V[x]
    """

    def __init__(
        self,
        total_shares: float = 1_000_000.0,  # Total volume X_0 to liquidate (>0) or acquire
        horizon: float = 1.0,               # Time horizon T (e.g. 1.0 day or 5.0 hours)
        n_intervals: int = 20,              # Number of discrete trading intervals N
        volatility: float = 0.30,           # Daily or annualized asset volatility sigma
        temp_impact: float = 2.5e-6,        # Temporary market impact parameter eta ($ / share^2)
        perm_impact: float = 2.5e-7,        # Permanent market impact parameter gamma ($ / share^2)
        fixed_cost: float = 0.0,            # Fixed transaction fee / spread epsilon ($ / share)
        initial_price: float = 100.0,       # Initial mid/arrival price S_0
    ):
        if total_shares == 0:
            raise ValueError("total_shares must be non-zero.")
        if horizon <= 0:
            raise ValueError("horizon must be strictly positive.")
        if n_intervals < 1:
            raise ValueError("n_intervals must be at least 1.")
        if volatility < 0 or temp_impact < 0 or perm_impact < 0:
            raise ValueError("Market parameters cannot be negative.")

        self.total_shares = total_shares
        self.horizon = horizon
        self.n_intervals = n_intervals
        self.tau = horizon / n_intervals
        self.volatility = volatility
        self.temp_impact = temp_impact
        self.perm_impact = perm_impact
        self.fixed_cost = fixed_cost
        self.initial_price = initial_price

    def solve_trajectory(self, risk_aversion: float = 1e-6) -> ExecutionTrajectoryResult:
        """Computes the closed-form optimal Almgren-Chriss trading trajectory.

        Args:
            risk_aversion: Trader risk aversion lambda >= 0.
                           lambda -> 0 yields linear TWAP (risk-neutral).
                           lambda -> infinity yields immediate urgent liquidation.
        """
        X0 = self.total_shares
        T = self.horizon
        N = self.n_intervals
        tau = self.tau
        sigma = self.volatility
        eta = self.temp_impact
        gamma = self.perm_impact
        eps = self.fixed_cost

        time_grid = np.linspace(0.0, T, N + 1)

        if risk_aversion <= 1e-12 or eta <= 1e-12 or sigma <= 1e-12:
            # Linear TWAP execution (risk-neutral limit lambda -> 0)
            kappa = 0.0
            half_life = T / 2.0
            holdings = np.array([X0 * (1.0 - j / N) for j in range(N + 1)])
            trade_sizes = np.array([X0 / N for _ in range(N)])
            trading_rates = trade_sizes / tau
        else:
            # Discrete Almgren-Chriss formulation
            # 2*cosh(kappa*tau) = 2 + (tau^2 * lambda * sigma^2) / eta
            cosh_val = 1.0 + 0.5 * (tau**2 * risk_aversion * (sigma**2)) / eta
            kappa_discrete = np.arccosh(cosh_val) / tau
            kappa = kappa_discrete
            half_life = 1.0 / kappa if kappa > 0 else T

            # Analytical trajectory formula: x_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X_0
            sinh_kappa_T = np.sinh(kappa * T)
            holdings = np.array([
                (np.sinh(kappa * (T - t_j)) / sinh_kappa_T) * X0 for t_j in time_grid
            ])
            holdings[-1] = 0.0  # Force exact terminal liquidation

            # Trade size per interval: n_j = x_{j-1} - x_j
            trade_sizes = holdings[:-1] - holdings[1:]
            trading_rates = trade_sizes / tau

        # Compute Expected Implementation Shortfall E[x]
        # E[x] = 0.5 * gamma * X0^2 + eps * sum(|n_j|) + eta * sum(n_j^2 / tau)
        perm_cost = 0.5 * gamma * (X0**2)
        fixed_friction = eps * np.sum(np.abs(trade_sizes))
        temp_cost = eta * np.sum((trade_sizes**2) / tau)
        expected_shortfall = perm_cost + fixed_friction + temp_cost

        # Compute Variance of Execution Cost V[x]
        # V[x] = sigma^2 * tau * sum_{j=1}^N x_j^2  (or integral of x(t)^2 dt)
        variance_shortfall = (sigma**2) * tau * np.sum(holdings[1:]**2)
        std_shortfall = np.sqrt(variance_shortfall)
        utility = expected_shortfall + risk_aversion * variance_shortfall

        return ExecutionTrajectoryResult(
            time_grid=time_grid,
            holdings=holdings,
            trade_sizes=trade_sizes,
            trading_rates=trading_rates,
            expected_shortfall=expected_shortfall,
            variance_shortfall=variance_shortfall,
            std_shortfall=std_shortfall,
            utility=utility,
            half_life=half_life,
            kappa=kappa,
            params={
                'total_shares': X0,
                'horizon': T,
                'n_intervals': N,
                'risk_aversion': risk_aversion,
                'volatility': sigma,
                'temp_impact': eta,
                'perm_impact': gamma,
            }
        )

    def efficient_frontier(
        self,
        lambda_min: float = 1e-8,
        lambda_max: float = 1e-3,
        n_points: int = 50,
    ) -> ExecutionFrontierResult:
        """Traces the efficient frontier of optimal execution: Expected Cost vs Risk Variance."""
        lambda_values = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_points)
        exp_costs = []
        std_costs = []
        vars_costs = []
        kappas = []

        for lam in lambda_values:
            traj = self.solve_trajectory(risk_aversion=lam)
            exp_costs.append(traj.expected_shortfall)
            std_costs.append(traj.std_shortfall)
            vars_costs.append(traj.variance_shortfall)
            kappas.append(traj.kappa)

        return ExecutionFrontierResult(
            lambda_values=lambda_values,
            expected_shortfalls=np.array(exp_costs),
            std_shortfalls=np.array(std_costs),
            variances=np.array(vars_costs),
            kappas=np.array(kappas),
        )
