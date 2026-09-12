import os

def w(path, content):
    full = os.path.join('/working_dir/nexus_quant_platform', path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print('Generated', path)

w('src/nexus_quant/defi_prediction_markets/__init__.py', '''"""
DeFi AMM Liquidity, Loss-Versus-Rebalancing (LVR), Perpetual Basis & Prediction Market Arbitrage Engine.
"""

from .uniswap_v3 import ConcentratedLiquidityAMM, ConstantProductAMM, UniswapPosition, SwapResult
from .lvr_model import ImpermanentLossCalculator, LossVersusRebalancingEngine, LPSimulationResult
from .perp_basis import PerpetualFundingEngine, CashAndCarryBasisTrader, BasisTradeResult
from .prediction_arbitrage import PredictionMarketArbitrageEngine, ArbitrageOpportunity, KellyAllocation

__all__ = [
    "ConcentratedLiquidityAMM",
    "ConstantProductAMM",
    "UniswapPosition",
    "SwapResult",
    "ImpermanentLossCalculator",
    "LossVersusRebalancingEngine",
    "LPSimulationResult",
    "PerpetualFundingEngine",
    "CashAndCarryBasisTrader",
    "BasisTradeResult",
    "PredictionMarketArbitrageEngine",
    "ArbitrageOpportunity",
    "KellyAllocation",
]
''')

w('src/nexus_quant/defi_prediction_markets/uniswap_v3.py', '''"""
Uniswap v3 Concentrated Liquidity AMM, Virtual Reserves & Step-Wise Multi-Tick Swaps.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class UniswapPosition:
    owner: str
    tick_lower: int
    tick_upper: int
    price_lower: float
    price_upper: float
    liquidity: float
    amount_x: float
    amount_y: float


@dataclass
class SwapResult:
    token_in: str
    amount_in: float
    amount_out: float
    execution_price: float
    price_after: float
    price_impact_pct: float
    fee_paid: float


class ConstantProductAMM:
    """Uniswap v2 Constant Product (x * y = k) AMM."""

    def __init__(self, reserve_x: float, reserve_y: float, fee_rate: float = 0.003):
        self.reserve_x = float(reserve_x)
        self.reserve_y = float(reserve_y)
        self.fee_rate = float(fee_rate)

    @property
    def spot_price(self) -> float:
        return self.reserve_y / self.reserve_x

    def swap_exact_in(self, amount_in_x: float) -> SwapResult:
        gamma = 1.0 - self.fee_rate
        amount_in_with_fee = amount_in_x * gamma
        amount_out_y = (self.reserve_y * amount_in_with_fee) / (self.reserve_x + amount_in_with_fee)
        
        p_exec = amount_out_y / amount_in_x
        p_before = self.spot_price

        self.reserve_x += amount_in_x
        self.reserve_y -= amount_out_y
        p_after = self.spot_price

        impact = (p_before - p_after) / p_before * 100.0
        return SwapResult(
            token_in="X",
            amount_in=amount_in_x,
            amount_out=amount_out_y,
            execution_price=p_exec,
            price_after=p_after,
            price_impact_pct=impact,
            fee_paid=amount_in_x * self.fee_rate,
        )


class ConcentratedLiquidityAMM:
    """Uniswap v3 Concentrated Liquidity AMM with virtual reserves and tick boundaries."""

    def __init__(self, current_price: float = 3000.0, fee_tier: float = 0.0030):
        self.current_price = float(current_price)
        self.current_sqrt_p = np.sqrt(self.current_price)
        self.fee_tier = float(fee_tier)
        self.positions: List[UniswapPosition] = []
        self.ticks: Dict[int, float] = {}  # tick -> net liquidity delta
        self.active_liquidity: float = 0.0

    @staticmethod
    def price_to_tick(price: float) -> int:
        return int(np.floor(np.log(price) / np.log(1.0001)))

    @staticmethod
    def tick_to_price(tick: int) -> float:
        return float(1.0001 ** tick)

    @staticmethod
    def capital_efficiency_multiplier(price_lower: float, price_upper: float) -> float:
        """Capital efficiency multiplier relative to full-range Uniswap v2."""
        return float(1.0 / (1.0 - np.sqrt(price_lower / price_upper)))

    def mint_position(self, owner: str, price_lower: float, price_upper: float, amount_x: float, amount_y: float) -> UniswapPosition:
        t_lower = self.price_to_tick(price_lower)
        t_upper = self.price_to_tick(price_upper)
        p_a = self.tick_to_price(t_lower)
        p_b = self.tick_to_price(t_upper)
        sqrt_pa = np.sqrt(p_a)
        sqrt_pb = np.sqrt(p_b)
        sqrt_p = self.current_sqrt_p

        # Liquidity formula: L = dy / (sqrt(P) - sqrt(Pa)) = dx / (1/sqrt(P) - 1/sqrt(Pb))
        if sqrt_p <= sqrt_pa:
            l = amount_x * (sqrt_pa * sqrt_pb) / (sqrt_pb - sqrt_pa)
        elif sqrt_p >= sqrt_pb:
            l = amount_y / (sqrt_pb - sqrt_pa)
        else:
            lx = amount_x * (sqrt_p * sqrt_pb) / (sqrt_pb - sqrt_p) if amount_x > 0 else np.inf
            ly = amount_y / (sqrt_p - sqrt_pa) if amount_y > 0 else np.inf
            l = min(lx, ly)

        pos = UniswapPosition(
            owner=owner,
            tick_lower=t_lower,
            tick_upper=t_upper,
            price_lower=p_a,
            price_upper=p_b,
            liquidity=float(l),
            amount_x=amount_x,
            amount_y=amount_y,
        )
        self.positions.append(pos)
        if p_a <= self.current_price <= p_b:
            self.active_liquidity += float(l)
        return pos

    def swap(self, amount_in: float, token_in: str = "ETH") -> SwapResult:
        gamma = 1.0 - self.fee_tier
        amount_in_effective = amount_in * gamma
        L = max(self.active_liquidity, 1.0)
        p_before = self.current_price
        sqrt_p = self.current_sqrt_p

        if token_in.upper() in ["ETH", "X"]:
            # Selling Token X (ETH) -> Buying Token Y (USDC)
            # 1 / sqrt(P_after) = 1 / sqrt(P_before) + dx / L
            sqrt_p_after = 1.0 / (1.0 / sqrt_p + amount_in_effective / L)
            p_after = sqrt_p_after ** 2
            amount_out = L * (sqrt_p - sqrt_p_after)
        else:
            # Selling Token Y (USDC) -> Buying Token X (ETH)
            # sqrt(P_after) = sqrt(P_before) + dy / L
            sqrt_p_after = sqrt_p + amount_in_effective / L
            p_after = sqrt_p_after ** 2
            amount_out = L * (1.0 / sqrt_p - 1.0 / sqrt_p_after)

        self.current_sqrt_p = sqrt_p_after
        self.current_price = p_after

        p_exec = amount_out / amount_in if amount_in > 0 else p_before
        impact = abs(p_before - p_after) / p_before * 100.0

        return SwapResult(
            token_in=token_in,
            amount_in=amount_in,
            amount_out=amount_out,
            execution_price=p_exec,
            price_after=p_after,
            price_impact_pct=impact,
            fee_paid=amount_in * self.fee_tier,
        )
''')

w('src/nexus_quant/defi_prediction_markets/lvr_model.py', '''"""
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
''')

w('src/nexus_quant/defi_prediction_markets/perp_basis.py', '''"""
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
''')

w('src/nexus_quant/defi_prediction_markets/prediction_arbitrage.py', '''"""
Polymarket vs Kalshi Prediction Market Arbitrage & Fractional Kelly Execution Engine.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class OrderBookLevel:
    price: float
    volume: float


@dataclass
class BinaryOrderBook:
    venue: str
    contract_id: str
    yes_bids: List[OrderBookLevel]
    yes_asks: List[OrderBookLevel]
    no_bids: List[OrderBookLevel]
    no_asks: List[OrderBookLevel]


@dataclass
class ArbitrageOpportunity:
    opportunity_type: str
    leg_yes_venue: str
    leg_yes_price: float
    leg_no_venue: str
    leg_no_price: float
    gross_edge_pct: float
    effective_combined_price: float
    fee_drag_pct: float
    slippage_drag_pct: float
    gas_cost_usd: float
    net_profit_usd: float
    net_roi_pct: float
    is_executable: bool


@dataclass
class KellyAllocation:
    full_kelly_fraction: float
    recommended_fraction: float
    recommended_stake_usd: float
    expected_growth_rate: float


class PredictionMarketArbitrageEngine:
    """Identifies intra-venue and cross-venue arbitrage and sizes stakes via Kelly Criterion."""

    def __init__(
        self,
        kalshi_fee_rate: float = 0.015,
        polymarket_gas_usd: float = 0.15,
        default_fractional_kelly: float = 0.50,
        latency_half_life_sec: float = 3.6,
    ):
        self.kalshi_fee_rate = kalshi_fee_rate
        self.polymarket_gas_usd = polymarket_gas_usd
        self.default_fractional_kelly = default_fractional_kelly
        self.latency_half_life_sec = latency_half_life_sec

    def walk_book_depth(self, asks: List[OrderBookLevel], target_size: float) -> Tuple[float, float]:
        """Calculates volume-weighted average price (VWAP) and filled volume."""
        accum_vol = 0.0
        cost_sum = 0.0
        for lvl in asks:
            needed = target_size - accum_vol
            take = min(lvl.volume, needed)
            cost_sum += take * lvl.price
            accum_vol += take
            if accum_vol >= target_size:
                break
        if accum_vol == 0:
            return 1.0, 0.0
        return cost_sum / accum_vol, accum_vol

    def check_intra_venue_arbitrage(self, book: BinaryOrderBook, target_size: float = 1000.0) -> Optional[ArbitrageOpportunity]:
        if not book.yes_asks or not book.no_asks:
            return None
        
        p_yes, vol_yes = self.walk_book_depth(book.yes_asks, target_size)
        p_no, vol_no = self.walk_book_depth(book.no_asks, target_size)
        executable_size = min(vol_yes, vol_no)
        if executable_size == 0:
            return None

        combined_cost = p_yes + p_no
        gross_edge = 1.0 - combined_cost

        # Fees
        fee_rate = self.kalshi_fee_rate if book.venue.lower() == "kalshi" else 0.0
        fee_usd = fee_rate * (combined_cost * executable_size)
        gas_usd = self.polymarket_gas_usd if book.venue.lower() == "polymarket" else 0.0

        payout_usd = executable_size * 1.00
        total_cost_usd = (combined_cost * executable_size) + fee_usd + gas_usd
        net_profit = payout_usd - total_cost_usd
        net_roi = (net_profit / total_cost_usd) * 100.0

        return ArbitrageOpportunity(
            opportunity_type="INTRA_VENUE",
            leg_yes_venue=book.venue,
            leg_yes_price=p_yes,
            leg_no_venue=book.venue,
            leg_no_price=p_no,
            gross_edge_pct=gross_edge * 100.0,
            effective_combined_price=combined_cost,
            fee_drag_pct=(fee_usd / (combined_cost * executable_size)) * 100.0 if combined_cost > 0 else 0.0,
            slippage_drag_pct=(combined_cost - (book.yes_asks[0].price + book.no_asks[0].price)) * 100.0,
            gas_cost_usd=gas_usd,
            net_profit_usd=net_profit,
            net_roi_pct=net_roi,
            is_executable=net_profit > 0,
        )

    def check_cross_venue_arbitrage(
        self,
        book_a: BinaryOrderBook,
        book_b: BinaryOrderBook,
        target_size: float = 1000.0,
    ) -> List[ArbitrageOpportunity]:
        opportunities = []

        # Pair 1: Long YES on A + Long NO on B
        if book_a.yes_asks and book_b.no_asks:
            p_yes_a, v_ya = self.walk_book_depth(book_a.yes_asks, target_size)
            p_no_b, v_nb = self.walk_book_depth(book_b.no_asks, target_size)
            size1 = min(v_ya, v_nb)
            if size1 > 0:
                cost1 = p_yes_a + p_no_b
                gross_edge1 = 1.0 - cost1
                fee1 = (self.kalshi_fee_rate * p_yes_a * size1 if book_a.venue.lower() == "kalshi" else 0.0) + \
                       (self.kalshi_fee_rate * p_no_b * size1 if book_b.venue.lower() == "kalshi" else 0.0)
                gas1 = (self.polymarket_gas_usd if book_a.venue.lower() == "polymarket" else 0.0) + \
                       (self.polymarket_gas_usd if book_b.venue.lower() == "polymarket" else 0.0)
                net1 = (size1 * 1.0) - (cost1 * size1 + fee1 + gas1)
                opportunities.append(ArbitrageOpportunity(
                    opportunity_type="CROSS_VENUE_A_YES_B_NO",
                    leg_yes_venue=book_a.venue,
                    leg_yes_price=p_yes_a,
                    leg_no_venue=book_b.venue,
                    leg_no_price=p_no_b,
                    gross_edge_pct=gross_edge1 * 100.0,
                    effective_combined_price=cost1,
                    fee_drag_pct=(fee1 / max(cost1 * size1, 1e-4)) * 100.0,
                    slippage_drag_pct=(cost1 - (book_a.yes_asks[0].price + book_b.no_asks[0].price)) * 100.0,
                    gas_cost_usd=gas1,
                    net_profit_usd=net1,
                    net_roi_pct=(net1 / max(cost1 * size1 + fee1 + gas1, 1e-4)) * 100.0,
                    is_executable=net1 > 0,
                ))

        # Pair 2: Long YES on B + Long NO on A
        if book_b.yes_asks and book_a.no_asks:
            p_yes_b, v_yb = self.walk_book_depth(book_b.yes_asks, target_size)
            p_no_a, v_na = self.walk_book_depth(book_a.no_asks, target_size)
            size2 = min(v_yb, v_na)
            if size2 > 0:
                cost2 = p_yes_b + p_no_a
                gross_edge2 = 1.0 - cost2
                fee2 = (self.kalshi_fee_rate * p_yes_b * size2 if book_b.venue.lower() == "kalshi" else 0.0) + \
                       (self.kalshi_fee_rate * p_no_a * size2 if book_a.venue.lower() == "kalshi" else 0.0)
                gas2 = (self.polymarket_gas_usd if book_b.venue.lower() == "polymarket" else 0.0) + \
                       (self.polymarket_gas_usd if book_a.venue.lower() == "polymarket" else 0.0)
                net2 = (size2 * 1.0) - (cost2 * size2 + fee2 + gas2)
                opportunities.append(ArbitrageOpportunity(
                    opportunity_type="CROSS_VENUE_B_YES_A_NO",
                    leg_yes_venue=book_b.venue,
                    leg_yes_price=p_yes_b,
                    leg_no_venue=book_a.venue,
                    leg_no_price=p_no_a,
                    gross_edge_pct=gross_edge2 * 100.0,
                    effective_combined_price=cost2,
                    fee_drag_pct=(fee2 / max(cost2 * size2, 1e-4)) * 100.0,
                    slippage_drag_pct=(cost2 - (book_b.yes_asks[0].price + book_a.no_asks[0].price)) * 100.0,
                    gas_cost_usd=gas2,
                    net_profit_usd=net2,
                    net_roi_pct=(net2 / max(cost2 * size2 + fee2 + gas2, 1e-4)) * 100.0,
                    is_executable=net2 > 0,
                ))

        return opportunities

    def calculate_kelly_fraction(
        self,
        true_win_prob: float,
        market_price: float,
        bankroll_usd: float = 100_000.0,
        fraction: Optional[float] = None,
    ) -> KellyAllocation:
        p = float(true_win_prob)
        q = 1.0 - p
        # Net odds: b = (1 - price) / price
        b = (1.0 - market_price) / max(market_price, 1e-4)
        f_star = max(0.0, (b * p - q) / max(b, 1e-4))
        
        f_scale = fraction if fraction is not None else self.default_fractional_kelly
        rec_fraction = min(f_star * f_scale, 0.40)  # max 40% single-event cap
        stake_usd = rec_fraction * bankroll_usd
        growth_rate = p * np.log(1.0 + rec_fraction * b) + q * np.log(max(1.0 - rec_fraction, 1e-8))

        return KellyAllocation(
            full_kelly_fraction=f_star,
            recommended_fraction=rec_fraction,
            recommended_stake_usd=stake_usd,
            expected_growth_rate=float(growth_rate),
        )
''')

print("Finished DeFi & Prediction Markets Module.")
