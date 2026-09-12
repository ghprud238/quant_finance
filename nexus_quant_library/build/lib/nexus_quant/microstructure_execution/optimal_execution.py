"""
Optimal Execution Frameworks: Almgren-Chriss (2000), TWAP, VWAP & Perold Implementation Shortfall.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class ExecutionTrajectory:
    """Output container for Almgren-Chriss execution schedule."""
    time_grid: np.ndarray
    holdings: np.ndarray
    trade_sizes: np.ndarray
    expected_shortfall: float
    std_shortfall: float
    variance_shortfall: float
    half_life: float
    risk_aversion: float


class AlmgrenChrissModel:
    """
    Almgren-Chriss (2000) Optimal Order Execution Engine.
    Solves for the optimal trading trajectory liquidating X_0 shares over horizon T
    to balance market impact costs and price volatility risk.
    """

    def __init__(
        self,
        total_shares: float,
        horizon: float = 1.0,           # e.g., 1 trading day
        n_intervals: int = 20,          # number of execution slices
        volatility: float = 0.30,       # asset annual volatility
        temp_impact: float = 2.5e-6,    # eta (temporary price impact parameter)
        perm_impact: float = 2.5e-7,    # gamma (permanent price impact parameter)
        fixed_cost: float = 0.001,      # epsilon (half-spread + fixed fee per share)
    ):
        self.total_shares = total_shares
        self.horizon = horizon
        self.n_intervals = n_intervals
        self.volatility = volatility
        self.temp_impact = temp_impact
        self.perm_impact = perm_impact
        self.fixed_cost = fixed_cost
        self.tau = horizon / n_intervals

    def solve_trajectory(self, risk_aversion: float = 1e-6) -> ExecutionTrajectory:
        """Solve analytical closed-form Almgren-Chriss trading schedule."""
        x0 = self.total_shares
        t_total = self.horizon
        n = self.n_intervals
        tau = self.tau
        lam = max(risk_aversion, 1e-12)
        sig = self.volatility
        eta = self.temp_impact
        gamma = self.perm_impact
        eps = self.fixed_cost

        # Urgency parameter kappa
        kappa_sq = (lam * (sig**2)) / eta
        kappa = np.sqrt(kappa_sq)

        t_grid = np.linspace(0.0, t_total, n + 1)
        holdings = np.zeros(n + 1)

        # Closed-form holdings trajectory: x_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X_0
        if kappa * t_total < 1e-4:
            # Linear TWAP limit as lambda -> 0
            holdings = x0 * (1.0 - t_grid / t_total)
        else:
            sinh_kt = np.sinh(kappa * t_total)
            for j in range(n + 1):
                holdings[j] = x0 * np.sinh(kappa * (t_total - t_grid[j])) / sinh_kt
            holdings[-1] = 0.0

        # Discrete trade slices n_j = x_{j-1} - x_j
        trade_sizes = -np.diff(holdings)

        # Expected Implementation Shortfall Cost
        perm_cost = 0.5 * gamma * (x0**2)
        fixed_cost_total = eps * np.sum(np.abs(trade_sizes))
        temp_cost = eta * np.sum((trade_sizes**2) / tau)
        exp_shortfall = perm_cost + fixed_cost_total + temp_cost

        # Variance of Shortfall
        var_shortfall = (sig**2) * tau * np.sum(holdings[:-1]**2)
        std_shortfall = np.sqrt(max(var_shortfall, 0.0))

        half_life = np.log(2.0) / max(kappa, 1e-6)

        return ExecutionTrajectory(
            time_grid=t_grid,
            holdings=holdings,
            trade_sizes=trade_sizes,
            expected_shortfall=float(exp_shortfall),
            std_shortfall=float(std_shortfall),
            variance_shortfall=float(var_shortfall),
            half_life=float(half_life),
            risk_aversion=risk_aversion,
        )

    def efficient_frontier(self, n_points: int = 40) -> pd.DataFrame:
        """Trace the efficient frontier of execution (Expected Cost vs Risk Std Dev)."""
        lambdas = np.logspace(-8, -4, n_points)
        e_costs = []
        v_risks = []
        half_lives = []

        for lam in lambdas:
            traj = self.solve_trajectory(risk_aversion=lam)
            e_costs.append(traj.expected_shortfall)
            v_risks.append(traj.std_shortfall)
            half_lives.append(traj.half_life)

        return pd.DataFrame({
            "Risk_Aversion_Lambda": lambdas,
            "Expected_Shortfall_USD": e_costs,
            "Shortfall_Std_Dev_USD": v_risks,
            "Half_Life_Days": half_lives,
        })


class BenchmarkExecutors:
    """Benchmark execution schedule generators: TWAP, VWAP, and POV."""

    @staticmethod
    def generate_twap(total_shares: float, n_intervals: int) -> np.ndarray:
        """Generate uniform Time-Weighted Average Price trade slices."""
        return np.full(n_intervals, total_shares / n_intervals)

    @staticmethod
    def generate_vwap(total_shares: float, volume_profile: np.ndarray) -> np.ndarray:
        """Generate Volume-Weighted Average Price trade slices based on historical volume curve."""
        norm_weights = volume_profile / np.sum(volume_profile)
        return total_shares * norm_weights

    @staticmethod
    def generate_pov(market_volume_series: np.ndarray, participation_rate: float = 0.10) -> np.ndarray:
        """Generate Percentage-of-Volume execution schedule."""
        return market_volume_series * participation_rate


class ImplementationShortfallAttributor:
    """
    Perold (1988) Implementation Shortfall Attribution.
    Decomposes trading cost into Delay Cost, Market Drift, Temporary Impact, Permanent Impact, and Commissions.
    """

    @classmethod
    def attribute(
        cls,
        total_shares: float,
        decision_price: float,
        arrival_price: float,
        terminal_price: float,
        trade_sizes: np.ndarray,
        execution_prices: np.ndarray,
        temp_impact_eta: float = 2.5e-6,
        perm_impact_gamma: float = 2.5e-7,
        side: str = "buy",
        commission_bps: float = 1.0,
    ) -> Dict[str, float]:
        """Decompose total implementation shortfall in dollars and basis points."""
        direction = 1.0 if side.lower() == "buy" else -1.0
        shares_filled = np.sum(trade_sizes)

        # 1. Total Benchmark Cost (Decision Price Notional)
        benchmark_notional = shares_filled * decision_price
        actual_execution_notional = np.sum(trade_sizes * execution_prices)
        total_shortfall_usd = direction * (actual_execution_notional - benchmark_notional)

        # 2. Delay Cost: (Arrival Price - Decision Price) * Shares
        delay_cost_usd = direction * (arrival_price - decision_price) * shares_filled

        # 3. Market Trend / Drift
        market_drift_usd = direction * (terminal_price - arrival_price) * (shares_filled - trade_sizes[0]) * 0.50

        # 4. Impact
        perm_impact_usd = 0.5 * perm_impact_gamma * (shares_filled**2)
        temp_impact_usd = temp_impact_eta * np.sum(trade_sizes**2)
        commissions_usd = actual_execution_notional * (commission_bps / 10000.0)

        total_bps = (total_shortfall_usd / benchmark_notional) * 10000.0

        return {
            "Total_Shortfall_USD": float(total_shortfall_usd),
            "Total_Shortfall_bps": float(total_bps),
            "Delay_Cost_USD": float(delay_cost_usd),
            "Market_Drift_USD": float(market_drift_usd),
            "Temporary_Impact_USD": float(temp_impact_usd),
            "Permanent_Impact_USD": float(perm_impact_usd),
            "Commissions_USD": float(commissions_usd),
            "Executed_Shares": float(shares_filled),
            "Average_Execution_Price": float(actual_execution_notional / shares_filled),
        }
