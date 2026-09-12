"""
Crypto Perpetual Futures 8-Hour Funding Rate Mechanics & Cash-and-Carry Basis Trading.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BasisTradeResult:
    initial_capital_usd: float
    final_equity_usd: float
    total_return_pct: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown_pct: float
    cumulative_funding_collected_usd: float
    net_funding_yield_apy: float
    peak_margin_utilization_pct: float
    margin_calls_triggered: int
    equity_curve: pd.Series

    def summary(self) -> str:
        return f"""Cash-and-Carry Delta-Neutral Performance:
  - Total Return:                    {self.total_return_pct:+.2%} (CAGR: {self.cagr:+.2%})
  - Annualized Volatility:           {self.volatility:.2%}
  - Sharpe Ratio (Rf=2%):            {self.sharpe_ratio:.2f}
  - Maximum Drawdown:                {self.max_drawdown_pct:.2%}
  - Net Funding Collected:           ${self.cumulative_funding_collected_usd:+,.2f}
  - Realized Funding APY:            {self.net_funding_yield_apy:+.2%}
  - Margin Calls:                    {self.margin_calls_triggered}"""


class PerpetualFundingEngine:
    """Simulates 8-hour perpetual futures funding rate dynamics."""

    @staticmethod
    def calculate_funding_rate(premium_index: float, interest_rate: float = 0.0003) -> float:
        """Standard 8-hour funding rate formula with 0.05% interest clamp and 0.75% absolute cap."""
        clamped_interest = np.clip(interest_rate - premium_index, -0.0005, 0.0005)
        return float(np.clip(premium_index + clamped_interest, -0.0075, 0.0075))


class CashAndCarryBasisTrader:
    """Delta-Neutral Spot-Perpetual Basis Trading with Margin Monitoring."""

    def __init__(
        self,
        initial_capital_usd: float = 1_000_000.0,
        spot_allocation_pct: float = 0.50,
        staking_yield_apy: float = 0.035,
        maintenance_margin_rate: float = 0.05,
    ):
        self.initial_capital_usd = initial_capital_usd
        self.spot_allocation_pct = spot_allocation_pct
        self.staking_yield_apy = staking_yield_apy
        self.maintenance_margin_rate = maintenance_margin_rate

    def backtest(self, market_df: pd.DataFrame) -> BasisTradeResult:
        df = market_df.copy()
        n = len(df)
        spot_cap = self.initial_capital_usd * self.spot_allocation_pct
        margin_cash = self.initial_capital_usd * (1.0 - self.spot_allocation_pct)

        p0 = df["spot_price"].iloc[0]
        n_units = spot_cap / p0

        equity_curve = []
        total_funding_collected = 0.0
        margin_calls = 0
        peak_margin_util = 0.0

        for i in range(n):
            row = df.iloc[i]
            p_spot = row["spot_price"]
            p_perp = row.get("perp_price", p_spot)
            funding_rate = row["funding_rate"]

            # 8-hour funding cashflow: Short perp receives funding when rate > 0
            funding_pnl = n_units * p_perp * funding_rate
            total_funding_collected += funding_pnl

            # Staking yield on spot (pro-rated 8h: 1 / (365*3))
            staking_pnl = n_units * p_spot * (self.staking_yield_apy / (365.0 * 3.0))

            # Short perp unrealized PnL: -(p_perp - p0) * n_units
            perp_unrealized_pnl = -(p_perp - p0) * n_units
            spot_val = n_units * p_spot

            margin_cash += (funding_pnl + staking_pnl)
            total_equity = spot_val + perp_unrealized_pnl + margin_cash
            equity_curve.append(total_equity)

            # Margin utilization
            perp_notional = n_units * p_perp
            effective_margin = margin_cash + perp_unrealized_pnl
            margin_ratio = effective_margin / max(perp_notional, 1.0)
            if margin_ratio > 0:
                utilization = 1.0 / margin_ratio
                peak_margin_util = max(peak_margin_util, utilization)

            if margin_ratio <= self.maintenance_margin_rate:
                margin_calls += 1

        eq_series = pd.Series(equity_curve, index=df.index if "spot_price" in df else None)
        total_ret = (eq_series.iloc[-1] - self.initial_capital_usd) / self.initial_capital_usd
        periods_per_year = 365.0 * 3.0
        cagr = float((eq_series.iloc[-1] / self.initial_capital_usd) ** (periods_per_year / n) - 1.0)
        
        returns_8h = eq_series.pct_change().dropna()
        vol_ann = float(returns_8h.std() * np.sqrt(periods_per_year))
        sharpe = float((cagr - 0.02) / vol_ann) if vol_ann > 0 else 0.0

        peaks = eq_series.cummax()
        drawdowns = (eq_series - peaks) / peaks
        max_dd = float(drawdowns.min())

        funding_apy = (total_funding_collected / self.initial_capital_usd) * (periods_per_year / n)

        return BasisTradeResult(
            initial_capital_usd=self.initial_capital_usd,
            final_equity_usd=float(eq_series.iloc[-1]),
            total_return_pct=float(total_ret),
            cagr=cagr,
            volatility=vol_ann,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            cumulative_funding_collected_usd=total_funding_collected,
            net_funding_yield_apy=float(funding_apy),
            peak_margin_utilization_pct=float(peak_margin_util * 100.0),
            margin_calls_triggered=margin_calls,
            equity_curve=eq_series,
        )
