"""Benchmark Execution Algorithms (TWAP, VWAP, POV) and Implementation Shortfall Attribution."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd


@dataclass
class ExecutionBenchmarkResult:
    """Result container for execution benchmarks (TWAP, VWAP, POV)."""
    algorithm_name: str
    total_shares: float
    executed_shares: float
    horizon: float
    trade_schedule: np.ndarray        # Slices n_j executed per interval
    execution_prices: np.ndarray      # Realized or simulated prices P_j
    average_price: float              # Weighted average execution price
    arrival_price: float              # S_0 benchmark price
    vwap_market_price: float          # Market VWAP over execution window
    total_cost: float                 # Dollar implementation shortfall vs arrival price
    slippage_bps: float               # Shortfall in basis points (bps)
    vwap_slippage_bps: float          # Shortfall relative to market VWAP in bps
    execution_log: pd.DataFrame


class TWAPExecutor:
    """Time-Weighted Average Price (TWAP) Execution Algorithm.

    Splits the parent order into N uniform time-sliced child orders:
        n_j = X_0 / N
    """

    @staticmethod
    def generate_schedule(total_shares: float, n_intervals: int) -> np.ndarray:
        """Generates uniform time-sliced trade sizes."""
        if n_intervals <= 0:
            raise ValueError("n_intervals must be positive.")
        return np.full(n_intervals, total_shares / n_intervals)

    @staticmethod
    def simulate_execution(
        total_shares: float,
        price_path: np.ndarray,
        temp_impact_eta: float = 2.5e-6,
        perm_impact_gamma: float = 2.5e-7,
        side: str = 'sell',
    ) -> ExecutionBenchmarkResult:
        """Simulates execution of a TWAP schedule over a price trajectory."""
        n_intervals = len(price_path) - 1
        schedule = TWAPExecutor.generate_schedule(total_shares, n_intervals)
        arrival_price = price_path[0]

        exec_prices = []
        cum_impact = 0.0
        direction = -1.0 if side.lower() == 'sell' else +1.0

        for j in range(n_intervals):
            n_j = schedule[j]
            # Permanent impact updates mid-price: Delta P_perm = direction * gamma * n_j
            cum_impact += direction * perm_impact_gamma * n_j
            # Temporary impact on executed price: Delta P_temp = direction * eta * n_j
            effective_price = price_path[j+1] + cum_impact + direction * temp_impact_eta * n_j
            exec_prices.append(effective_price)

        exec_prices = np.array(exec_prices)
        avg_price = np.average(exec_prices, weights=schedule)

        # Dollar shortfall vs arrival price
        if side.lower() == 'sell':
            shortfall = (arrival_price - avg_price) * total_shares
        else:
            shortfall = (avg_price - arrival_price) * total_shares

        slippage_bps = (shortfall / (arrival_price * total_shares)) * 10000.0
        mkt_vwap = np.mean(price_path[1:])
        vwap_slippage_bps = ((mkt_vwap - avg_price) / arrival_price if side == 'sell' else (avg_price - mkt_vwap) / arrival_price) * 10000.0

        df_log = pd.DataFrame({
            'Step': np.arange(1, n_intervals + 1),
            'Market_Price': price_path[1:],
            'Trade_Size': schedule,
            'Execution_Price': exec_prices,
            'Cash_Flow': schedule * exec_prices,
        })

        return ExecutionBenchmarkResult(
            algorithm_name='TWAP',
            total_shares=total_shares,
            executed_shares=total_shares,
            horizon=float(n_intervals),
            trade_schedule=schedule,
            execution_prices=exec_prices,
            average_price=avg_price,
            arrival_price=arrival_price,
            vwap_market_price=mkt_vwap,
            total_cost=shortfall,
            slippage_bps=slippage_bps,
            vwap_slippage_bps=vwap_slippage_bps,
            execution_log=df_log,
        )


class VWAPExecutor:
    """Volume-Weighted Average Price (VWAP) Execution Algorithm.

    Allocates order volume proportionally to the expected intraday market volume curve:
        n_j = X_0 * (V_j / V_total)
    """

    @staticmethod
    def u_shaped_volume_profile(n_intervals: int = 20) -> np.ndarray:
        """Generates a canonical intraday U-shaped volume distribution (high at open/close, lower at midday)."""
        x = np.linspace(-1.0, 1.0, n_intervals)
        profile = 0.5 * (x**2) + 0.5
        return profile / np.sum(profile)

    @classmethod
    def generate_schedule(cls, total_shares: float, n_intervals: int, volume_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Generates volume-weighted slice schedule."""
        if volume_profile is None:
            volume_profile = cls.u_shaped_volume_profile(n_intervals)
        else:
            volume_profile = volume_profile / np.sum(volume_profile)
        return total_shares * volume_profile

    @classmethod
    def simulate_execution(
        cls,
        total_shares: float,
        price_path: np.ndarray,
        volume_profile: Optional[np.ndarray] = None,
        temp_impact_eta: float = 2.5e-6,
        perm_impact_gamma: float = 2.5e-7,
        side: str = 'sell',
    ) -> ExecutionBenchmarkResult:
        """Simulates execution of a VWAP schedule."""
        n_intervals = len(price_path) - 1
        schedule = cls.generate_schedule(total_shares, n_intervals, volume_profile)
        arrival_price = price_path[0]

        exec_prices = []
        cum_impact = 0.0
        direction = -1.0 if side.lower() == 'sell' else +1.0

        for j in range(n_intervals):
            n_j = schedule[j]
            cum_impact += direction * perm_impact_gamma * n_j
            effective_price = price_path[j+1] + cum_impact + direction * temp_impact_eta * n_j
            exec_prices.append(effective_price)

        exec_prices = np.array(exec_prices)
        avg_price = np.average(exec_prices, weights=schedule)

        if side.lower() == 'sell':
            shortfall = (arrival_price - avg_price) * total_shares
        else:
            shortfall = (avg_price - arrival_price) * total_shares

        slippage_bps = (shortfall / (arrival_price * total_shares)) * 10000.0
        mkt_vwap = np.average(price_path[1:], weights=schedule)
        vwap_slippage_bps = ((mkt_vwap - avg_price) / arrival_price if side == 'sell' else (avg_price - mkt_vwap) / arrival_price) * 10000.0

        df_log = pd.DataFrame({
            'Step': np.arange(1, n_intervals + 1),
            'Market_Price': price_path[1:],
            'Trade_Size': schedule,
            'Execution_Price': exec_prices,
            'Cash_Flow': schedule * exec_prices,
        })

        return ExecutionBenchmarkResult(
            algorithm_name='VWAP',
            total_shares=total_shares,
            executed_shares=total_shares,
            horizon=float(n_intervals),
            trade_schedule=schedule,
            execution_prices=exec_prices,
            average_price=avg_price,
            arrival_price=arrival_price,
            vwap_market_price=mkt_vwap,
            total_cost=shortfall,
            slippage_bps=slippage_bps,
            vwap_slippage_bps=vwap_slippage_bps,
            execution_log=df_log,
        )


class POVExecutor:
    """Percentage of Volume (POV) / Participation Rate Execution Algorithm.

    Participates at a fixed fraction alpha of market volume: n_j = alpha * V_{mkt, j}.
    """

    @staticmethod
    def generate_schedule(total_shares: float, market_volumes: np.ndarray, participation_rate: float = 0.10) -> np.ndarray:
        """Generates participation-capped trade schedule."""
        raw_sizes = participation_rate * market_volumes
        cum_sizes = np.cumsum(raw_sizes)
        schedule = []
        rem = total_shares

        for s in raw_sizes:
            traded = min(s, rem)
            schedule.append(traded)
            rem -= traded
            if rem <= 1e-8:
                break

        # Pad with zeros if done early
        while len(schedule) < len(market_volumes):
            schedule.append(0.0)

        return np.array(schedule)


class ImplementationShortfallAttributor:
    """Perrault (Perold 1988) Implementation Shortfall Cost Attribution Engine.

    Decomposes total execution slippage vs. arrival price benchmark into:
    1. Delay Cost (Lag between decision time and order release)
    2. Market Drift (Underlying asset price trend over execution window)
    3. Temporary Market Impact (Liquidity friction per trade slice)
    4. Permanent Market Impact (Information leakage / structural price shift)
    5. Fixed Fees / Commissions
    """

    @staticmethod
    def attribute_costs(
        total_shares: float,
        decision_price: float,
        arrival_price: float,
        terminal_price: float,
        trade_sizes: np.ndarray,
        execution_prices: np.ndarray,
        temp_impact_eta: float = 2.5e-6,
        perm_impact_gamma: float = 2.5e-7,
        commission_per_share: float = 0.005,
        side: str = 'sell',
    ) -> Dict[str, Any]:
        """Performs formal implementation shortfall cost decomposition."""
        side_mult = -1.0 if side.lower() == 'sell' else +1.0
        X0 = total_shares

        # 1. Total Implementation Shortfall vs Decision Price
        avg_exec_price = np.average(execution_prices, weights=trade_sizes)
        if side.lower() == 'sell':
            total_shortfall = (decision_price - avg_exec_price) * X0
        else:
            total_shortfall = (avg_exec_price - decision_price) * X0

        # 2. Delay Cost: (Arrival - Decision) * X0
        delay_cost = side_mult * (arrival_price - decision_price) * X0

        # 3. Market Drift (Exogenous price trend over execution)
        market_drift = side_mult * (terminal_price - arrival_price) * (X0 - np.sum(trade_sizes * (1.0 - np.arange(len(trade_sizes))/len(trade_sizes))))

        # 4. Permanent Impact
        permanent_impact = 0.5 * perm_impact_gamma * (X0**2)

        # 5. Temporary Impact
        temporary_impact = temp_impact_eta * np.sum(trade_sizes**2)

        # 6. Commissions & Fees
        commissions = commission_per_share * X0

        total_attributed = delay_cost + permanent_impact + temporary_impact + commissions

        return {
            'Total_Shortfall_Dollars': total_shortfall,
            'Total_Shortfall_bps': (total_shortfall / (decision_price * X0)) * 10000.0,
            'Delay_Cost_Dollars': delay_cost,
            'Delay_Cost_bps': (delay_cost / (decision_price * X0)) * 10000.0,
            'Temporary_Impact_Dollars': temporary_impact,
            'Temporary_Impact_bps': (temporary_impact / (decision_price * X0)) * 10000.0,
            'Permanent_Impact_Dollars': permanent_impact,
            'Permanent_Impact_bps': (permanent_impact / (decision_price * X0)) * 10000.0,
            'Commissions_Dollars': commissions,
            'Commissions_bps': (commissions / (decision_price * X0)) * 10000.0,
            'Average_Execution_Price': avg_exec_price,
            'Decision_Price': decision_price,
            'Arrival_Price': arrival_price,
        }
