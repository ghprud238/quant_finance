"""Impermanent Loss & Loss-Versus-Rebalancing (LVR) Quantitative Framework (Project 42).

Implements:
1. Standard CFMM Impermanent Loss & Concentrated Liquidity (v3) IL functions.
2. Milionis, Moallemi, Roughgarden, Adams (2022) Loss-Versus-Rebalancing (LVR) Model.
3. Arbitrageur extraction dynamics, continuous LVR integral, and LP Net Profitability.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class LVRSimulationResult:
    """Results of an LVR and LP Profitability simulation."""
    timestamps: pd.DatetimeIndex
    spot_prices: np.ndarray
    cumulative_lvr_usd: np.ndarray
    cumulative_fees_usd: np.ndarray
    cumulative_net_pnl_usd: np.ndarray
    impermanent_loss_usd: float
    total_lvr_usd: float
    total_fee_revenue_usd: float
    net_lp_pnl_usd: float
    lvr_to_volume_bps: float
    breakeven_volatility_ann: float
    summary_table: pd.DataFrame


@dataclass
class LPProfitabilityReport:
    """Detailed LP Return & Risk Breakdown."""
    initial_capital_usd: float
    terminal_lp_value_usd: float
    terminal_hodl_value_usd: float
    impermanent_loss_usd: float
    impermanent_loss_pct: float
    total_lvr_usd: float
    fee_revenue_usd: float
    gas_costs_usd: float
    net_pnl_usd: float
    net_roi_pct: float
    annualized_sharpe: float
    max_drawdown_pct: float


# =========================================================================
# 1. IMPERMANENT LOSS CALCULATOR (v2 & v3)
# =========================================================================

class ImpermanentLossCalculator:
    """Analytical Impermanent Loss engine for full-range (v2) and concentrated (v3) AMMs."""

    @staticmethod
    def calculate_v2_impermanent_loss(price_ratio_k: float) -> float:
        """Standard Uniswap v2 Impermanent Loss formula:
        
        IL(k) = (2 * sqrt(k)) / (1 + k) - 1
        where k = P_t / P_0 is the relative price ratio.
        """
        if price_ratio_k <= 0:
            raise ValueError("Price ratio k must be strictly positive.")
        sqrt_k = np.sqrt(price_ratio_k)
        return (2.0 * sqrt_k) / (1.0 + price_ratio_k) - 1.0

    @classmethod
    def calculate_v3_impermanent_loss(
        cls,
        price_current: float,
        price_initial: float,
        price_lower: float,
        price_upper: float,
    ) -> float:
        """Concentrated Liquidity (Uniswap v3) Impermanent Loss within range [P_a, P_b].
        
        Compares the value of the concentrated LP position against HODLing the initial tokens.
        """
        if price_lower >= price_upper:
            raise ValueError("price_lower must be strictly less than price_upper.")
        if price_current <= 0 or price_initial <= 0:
            raise ValueError("Prices must be strictly positive.")
            
        sqrt_p0 = np.sqrt(price_initial)
        sqrt_p = np.sqrt(price_current)
        sqrt_a = np.sqrt(price_lower)
        sqrt_b = np.sqrt(price_upper)
        
        # Initial deposit amounts for L = 1.0
        if sqrt_p0 <= sqrt_a:
            x0 = (sqrt_b - sqrt_a) / (sqrt_a * sqrt_b)
            y0 = 0.0
        elif sqrt_p0 >= sqrt_b:
            x0 = 0.0
            y0 = sqrt_b - sqrt_a
        else:
            x0 = (sqrt_b - sqrt_p0) / (sqrt_p0 * sqrt_b)
            y0 = sqrt_p0 - sqrt_a
            
        # Value of HODL portfolio at current price P
        v_hodl = x0 * price_current + y0
        
        # Value of Concentrated LP portfolio at current price P
        if sqrt_p <= sqrt_a:
            xt = (sqrt_b - sqrt_a) / (sqrt_a * sqrt_b)
            yt = 0.0
        elif sqrt_p >= sqrt_b:
            xt = 0.0
            yt = sqrt_b - sqrt_a
        else:
            xt = (sqrt_b - sqrt_p) / (sqrt_p * sqrt_b)
            yt = sqrt_p - sqrt_a
            
        v_lp = xt * price_current + yt
        
        if v_hodl == 0:
            return 0.0
        return (v_lp / v_hodl) - 1.0


# =========================================================================
# 2. LOSS-VERSUS-REBALANCING (LVR) MODEL
# =========================================================================

class LossVersusRebalancingEngine:
    """Milionis, Moallemi, Roughgarden, Adams (2022) Loss-Versus-Rebalancing (LVR) Engine.
    
    LVR quantifies the unhedgeable adverse selection cost suffered by passive AMM LPs
    to external arbitrageurs due to stale on-chain quotes relative to continuous reference markets.
    """

    def __init__(
        self,
        pool_type: str = "v2",  # v2 or v3
        fee_rate: float = 0.0030,
        gas_cost_per_rebalance: float = 2.50,
    ):
        self.pool_type = pool_type.lower()
        self.fee_rate = float(fee_rate)
        self.gas_cost = float(gas_cost_per_rebalance)

    @classmethod
    def continuous_lvr_integral(
        cls,
        spot_prices: np.ndarray,
        liquidity_L: np.ndarray,
        volatility_ann: float,
        dt_years: float,
    ) -> np.ndarray:
        """Computes continuous-time theoretical LVR:
        
        dLVR_t = (sigma^2 / 8) * S_t * L_t * dt
        LVR_t = integral_0^t (sigma^2 / 8) * S_u * L_u du
        """
        sigma_sq = volatility_ann ** 2
        instantaneous_lvr = (sigma_sq / 8.0) * spot_prices * liquidity_L * dt_years
        return np.cumsum(instantaneous_lvr)

    def simulate_lp_performance(
        self,
        price_series: pd.Series,
        volume_series: pd.Series,
        initial_capital_usd: float = 100_000.0,
        price_lower: Optional[float] = None,
        price_upper: Optional[float] = None,
        volatility_ann: Optional[float] = None,
    ) -> LVRSimulationResult:
        """Simulates path-dependent LP returns, arbitrageur LVR extraction, and fee generation."""
        prices = price_series.values
        volumes = volume_series.values
        timestamps = price_series.index
        n_steps = len(prices)
        
        p0 = prices[0]
        p_a = price_lower if price_lower is not None else p0 * 0.70
        p_b = price_upper if price_upper is not None else p0 * 1.30
        
        # Estimate empirical volatility if not provided
        if volatility_ann is None:
            returns = np.diff(np.log(prices))
            volatility_ann = float(np.std(returns) * np.sqrt(365 * 24))  # assuming hourly
            
        dt_years = (1.0 / (365 * 24)) if len(prices) > 365 else (1.0 / 365)
        
        # Calculate active liquidity L from initial capital
        # Initial 50/50 split at P0
        if self.pool_type == "v2":
            x0 = (initial_capital_usd / 2.0) / p0
            y0 = initial_capital_usd / 2.0
            L = np.sqrt(x0 * y0)
            liquidity_arr = np.full(n_steps, L)
        else:
            # v3 concentrated
            sqrt_p0 = np.sqrt(p0)
            sqrt_a = np.sqrt(p_a)
            sqrt_b = np.sqrt(p_b)
            # L = v0 / (2 * sqrt(P0) - P0/sqrt(P_b) - sqrt(P_a))
            denom = (2.0 * sqrt_p0 - p0 / sqrt_b - sqrt_a)
            L = initial_capital_usd / denom if denom > 0 else initial_capital_usd / (2.0 * sqrt_p0)
            
            # Liquidity is active only while P in [P_a, P_b]
            in_range = (prices >= p_a) & (prices <= p_b)
            liquidity_arr = np.where(in_range, L, 0.0)
            
        # 1. Theoretical Continuous LVR
        cumulative_lvr = self.continuous_lvr_integral(prices, liquidity_arr, volatility_ann, dt_years)
        
        # 2. Fee Revenue Generation
        # Assume pool captures fee_rate * volume_share
        fee_revenue_steps = volumes * self.fee_rate * (liquidity_arr / (liquidity_arr + 1e-6)) * 0.05
        cumulative_fees = np.cumsum(fee_revenue_steps)
        
        # 3. Net LP P&L = Fees - LVR
        cumulative_net_pnl = cumulative_fees - cumulative_lvr
        
        total_lvr = float(cumulative_lvr[-1])
        total_fees = float(cumulative_fees[-1])
        net_pnl = float(cumulative_net_pnl[-1])
        
        # Impermanent Loss at terminal price
        p_terminal = prices[-1]
        if self.pool_type == "v2":
            il_pct = ImpermanentLossCalculator.calculate_v2_impermanent_loss(p_terminal / p0)
        else:
            il_pct = ImpermanentLossCalculator.calculate_v3_impermanent_loss(p_terminal, p0, p_a, p_b)
        il_usd = il_pct * initial_capital_usd
        
        total_volume = np.sum(volumes)
        lvr_bps = (total_lvr / total_volume * 10000.0) if total_volume > 0 else 0.0
        
        # Breakeven volatility: sigma_be = sqrt(8 * Fees / (S * L * T))
        total_s_l_t = np.sum(prices * liquidity_arr * dt_years)
        if total_s_l_t > 0:
            breakeven_vol = float(np.sqrt(8.0 * total_fees / total_s_l_t))
        else:
            breakeven_vol = 0.0
            
        summary_metrics = {
            "Pool Type": self.pool_type.upper(),
            "Initial Capital ($)": f"${initial_capital_usd:,.2f}",
            "Terminal Spot Price ($)": f"${p_terminal:,.2f} ({p_terminal/p0 - 1.0:+.2%})",
            "Impermanent Loss (IL) ($)": f"${il_usd:,.2f} ({il_pct:+.2%})",
            "Total LVR (Adverse Selection) ($)": f"${total_lvr:,.2f}",
            "Total Fee Revenue ($)": f"${total_fees:,.2f}",
            "Net LP Alpha / P&L ($)": f"${net_pnl:,.2f}",
            "LVR / Volume Drag (bps)": f"{lvr_bps:.2f} bps",
            "Annualized Asset Volatility": f"{volatility_ann:.2%}",
            "Breakeven Volatility": f"{breakeven_vol:.2%}",
            "LP Profitable?": "YES" if net_pnl > 0 else "NO (LVR exceeded Fees)",
        }
        summary_df = pd.DataFrame(list(summary_metrics.items()), columns=["Metric", "Value"])
        
        return LVRSimulationResult(
            timestamps=timestamps,
            spot_prices=prices,
            cumulative_lvr_usd=cumulative_lvr,
            cumulative_fees_usd=cumulative_fees,
            cumulative_net_pnl_usd=cumulative_net_pnl,
            impermanent_loss_usd=il_usd,
            total_lvr_usd=total_lvr,
            total_fee_revenue_usd=total_fees,
            net_lp_pnl_usd=net_pnl,
            lvr_to_volume_bps=lvr_bps,
            breakeven_volatility_ann=breakeven_vol,
            summary_table=summary_df,
        )
