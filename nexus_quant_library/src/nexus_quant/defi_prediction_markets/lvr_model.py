"""
Impermanent Loss & Loss-Versus-Rebalancing (LVR) Modeling (Milionis et al. 2022).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


class ImpermanentLossCalculator:
    """Computes traditional CFMM and Concentrated Liquidity Impermanent Loss."""

    @staticmethod
    def standard_cfmm_il(price_ratio: float) -> float:
        """Standard Uniswap v2 IL: 2*sqrt(k)/(1+k) - 1."""
        k = float(price_ratio)
        return float(2.0 * np.sqrt(k) / (1.0 + k) - 1.0)

    @staticmethod
    def concentrated_liquidity_il(p0: float, pt: float, pa: float, pb: float) -> float:
        """Concentrated liquidity impermanent loss within [pa, pb]."""
        sqrt_p0, sqrt_pt, sqrt_pa, sqrt_pb = np.sqrt(p0), np.sqrt(pt), np.sqrt(pa), np.sqrt(pb)

        # Value of LP position at pt
        if pt <= pa:
            v_lp = pt * (1.0 / sqrt_pa - 1.0 / sqrt_pb)
        elif pt >= pb:
            v_lp = sqrt_pb - sqrt_pa
        else:
            v_lp = pt * (1.0 / sqrt_pt - 1.0 / sqrt_pb) + (sqrt_pt - sqrt_pa)

        # Value of HODL position
        if p0 <= pa:
            x0 = (1.0 / sqrt_pa - 1.0 / sqrt_pb)
            y0 = 0.0
        elif p0 >= pb:
            x0 = 0.0
            y0 = (sqrt_pb - sqrt_pa)
        else:
            x0 = (1.0 / sqrt_p0 - 1.0 / sqrt_pb)
            y0 = (sqrt_p0 - sqrt_pa)
        v_hodl = x0 * pt + y0

        return float(v_lp / v_hodl - 1.0) if v_hodl > 0 else 0.0


@dataclass
class LPSimulationResult:
    total_fee_revenue_usd: float
    total_lvr_usd: float
    total_impermanent_loss_usd: float
    net_lp_profit_usd: float
    breakeven_volatility_annual: float
    realized_volatility_annual: float
    is_lp_profitable: bool
    summary_table: pd.DataFrame


class LossVersusRebalancingEngine:
    """Milionis-Moallemi-Roughgarden Loss-Versus-Rebalancing (LVR) Adverse Selection Engine."""

    def __init__(self, pool_type: str = "v3", fee_rate: float = 0.0030):
        self.pool_type = pool_type.lower()
        self.fee_rate = fee_rate

    def compute_continuous_lvr(self, price_series: pd.Series, liquidity: float, annual_vol: float) -> float:
        """Continuous-time instantaneous LVR: integral(sigma^2 / 8 * S_t * L dt)."""
        dt = 1.0 / (252.0 * 24.0)  # hourly steps
        daily_lvr_rate = (annual_vol ** 2) / 8.0
        lvr_sum = np.sum(price_series * liquidity * daily_lvr_rate * dt)
        return float(lvr_sum)

    def simulate_lp_performance(
        self,
        price_series: pd.Series,
        volume_series: pd.Series,
        initial_capital_usd: float = 100_000.0,
        price_lower: float = 2500.0,
        price_upper: float = 3500.0,
    ) -> LPSimulationResult:
        p0 = price_series.iloc[0]
        pt = price_series.iloc[-1]
        
        # Realized annual volatility
        ret = price_series.pct_change().dropna()
        realized_vol = float(ret.std() * np.sqrt(252.0 * 24.0))

        # Fee revenue: pool captures 0.30% fee on volume proportional to LP market share
        total_pool_volume = float(volume_series.sum())
        lp_pool_share = initial_capital_usd / 10_000_000.0  # assume $10M pool TVL
        fee_revenue = total_pool_volume * self.fee_rate * lp_pool_share

        # LVR calculation
        multiplier = 1.0 / (1.0 - np.sqrt(price_lower / price_upper))
        effective_liquidity = (initial_capital_usd / (2.0 * np.sqrt(p0))) * multiplier
        lvr = self.compute_continuous_lvr(price_series, effective_liquidity, realized_vol)

        # Standard IL
        il_pct = ImpermanentLossCalculator.concentrated_liquidity_il(p0, pt, price_lower, price_upper)
        il_usd = abs(il_pct) * initial_capital_usd

        net_profit = fee_revenue - lvr
        sigma_be = float(np.sqrt(8.0 * fee_revenue / max(np.sum(price_series * effective_liquidity * (1.0 / (252.0 * 24.0))), 1e-4)))

        summary = pd.DataFrame([
            {"Metric": "Initial LP Capital ($)", "Value": f"${initial_capital_usd:,.2f}"},
            {"Metric": "Fee Revenue Earned ($)", "Value": f"${fee_revenue:,.2f}"},
            {"Metric": "LVR Adverse Selection Cost ($)", "Value": f"${lvr:,.2f}"},
            {"Metric": "Standard Impermanent Loss ($)", "Value": f"${il_usd:,.2f}"},
            {"Metric": "Net LP Profit / Loss ($)", "Value": f"${net_profit:+,.2f}"},
            {"Metric": "Realized Annual Volatility", "Value": f"{realized_vol:.2%}"},
            {"Metric": "Breakeven Volatility Threshold", "Value": f"{sigma_be:.2%}"},
            {"Metric": "LP Alpha Regime", "Value": "PROFITABLE (Fees > LVR)" if net_profit > 0 else "DRAINED (LVR > Fees)"},
        ])

        return LPSimulationResult(
            total_fee_revenue_usd=fee_revenue,
            total_lvr_usd=lvr,
            total_impermanent_loss_usd=il_usd,
            net_lp_profit_usd=net_profit,
            breakeven_volatility_annual=sigma_be,
            realized_volatility_annual=realized_vol,
            is_lp_profitable=net_profit > 0,
            summary_table=summary,
        )
