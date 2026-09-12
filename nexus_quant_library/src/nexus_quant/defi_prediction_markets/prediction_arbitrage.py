"""
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
                fee1 = (self.kalshi_fee_rate * p_yes_a * size1 if book_a.venue.lower() == "kalshi" else 0.0) +                        (self.kalshi_fee_rate * p_no_b * size1 if book_b.venue.lower() == "kalshi" else 0.0)
                gas1 = (self.polymarket_gas_usd if book_a.venue.lower() == "polymarket" else 0.0) +                        (self.polymarket_gas_usd if book_b.venue.lower() == "polymarket" else 0.0)
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
                fee2 = (self.kalshi_fee_rate * p_yes_b * size2 if book_b.venue.lower() == "kalshi" else 0.0) +                        (self.kalshi_fee_rate * p_no_a * size2 if book_a.venue.lower() == "kalshi" else 0.0)
                gas2 = (self.polymarket_gas_usd if book_b.venue.lower() == "polymarket" else 0.0) +                        (self.polymarket_gas_usd if book_a.venue.lower() == "polymarket" else 0.0)
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
