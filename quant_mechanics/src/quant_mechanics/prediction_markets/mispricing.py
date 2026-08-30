"""Prediction Market Arbitrage, L2 Depth Walking, and Kelly Execution Engine."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from quant_mechanics.data.loader import BinaryContractOrderBook, OrderBookLevel


class ArbitrageType(Enum):
    INTRA_VENUE = "INTRA_VENUE"
    CROSS_VENUE = "CROSS_VENUE"


@dataclass
class ExecutionResult:
    """Detailed breakdown of order book walking and cost attribution."""
    requested_size: float
    executed_size: float
    effective_price: float
    best_available_price: float
    slippage_bps: float
    total_cost_usd: float
    venue_fees_usd: float
    gas_cost_usd: float
    is_fully_filled: bool


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity with complete financial economics."""
    contract_id: str
    arb_type: ArbitrageType
    leg_yes_venue: str
    leg_no_venue: str
    leg_yes_price: float
    leg_no_price: float
    gross_combined_price: float
    gross_edge_pct: float
    max_executable_size: float
    depth_weighted_yes_price: float
    depth_weighted_no_price: float
    effective_combined_price: float
    net_profit_usd: float
    net_roi_pct: float
    total_capital_required: float
    yes_execution: ExecutionResult
    no_execution: ExecutionResult
    is_profitable: bool


@dataclass
class KellyAllocationResult:
    """Kelly criterion sizing recommendations."""
    full_kelly_fraction: float
    fractional_kelly_multiplier: float
    recommended_fraction: float
    recommended_stake_usd: float
    expected_growth_rate: float
    win_probability: float
    net_odds: float
    bankroll_usd: float


class PredictionMarketArbitrageEngine:
    """Institutional execution and arbitrage engine for binary prediction markets."""

    def __init__(
        self,
        default_fractional_kelly: float = 0.50,
        default_latency_half_life_sec: float = 2.50,
        min_net_roi_threshold: float = 0.002,  # 20 bps minimum net edge
    ):
        self.fractional_kelly = default_fractional_kelly
        self.latency_half_life_sec = default_latency_half_life_sec
        self.min_net_roi_threshold = min_net_roi_threshold

    @staticmethod
    def walk_order_book(
        levels: List[OrderBookLevel],
        target_size: float,
        fee_rate: float = 0.0,
        gas_cost_usd: float = 0.0,
    ) -> ExecutionResult:
        """Walks the L2 order book depth ladder to calculate average execution price and slippage."""
        if not levels or target_size <= 0:
            return ExecutionResult(
                requested_size=target_size,
                executed_size=0.0,
                effective_price=0.0,
                best_available_price=0.0,
                slippage_bps=0.0,
                total_cost_usd=gas_cost_usd,
                venue_fees_usd=0.0,
                gas_cost_usd=gas_cost_usd,
                is_fully_filled=False,
            )

        best_price = levels[0].price
        remaining = target_size
        total_dollars = 0.0
        executed_qty = 0.0

        for lvl in levels:
            if remaining <= 0:
                break
            fill = min(remaining, lvl.size)
            total_dollars += fill * lvl.price
            executed_qty += fill
            remaining -= fill

        if executed_qty == 0:
            return ExecutionResult(
                requested_size=target_size,
                executed_size=0.0,
                effective_price=0.0,
                best_available_price=best_price,
                slippage_bps=0.0,
                total_cost_usd=gas_cost_usd,
                venue_fees_usd=0.0,
                gas_cost_usd=gas_cost_usd,
                is_fully_filled=False,
            )

        eff_price = total_dollars / executed_qty
        slippage_bps = ((eff_price - best_price) / max(best_price, 1e-6)) * 10_000.0
        venue_fees = total_dollars * fee_rate
        total_cost = total_dollars + venue_fees + gas_cost_usd

        return ExecutionResult(
            requested_size=target_size,
            executed_size=executed_qty,
            effective_price=eff_price,
            best_available_price=best_price,
            slippage_bps=slippage_bps,
            total_cost_usd=total_cost,
            venue_fees_usd=venue_fees,
            gas_cost_usd=gas_cost_usd,
            is_fully_filled=(remaining <= 1e-6),
        )

    def check_intra_venue_arbitrage(
        self,
        book: BinaryContractOrderBook,
        target_size: Optional[float] = None,
    ) -> Optional[ArbitrageOpportunity]:
        """Evaluates intra-venue arbitrage (buying Yes Ask + No Ask on the same venue)."""
        if not book.yes_asks or not book.no_asks:
            return None

        best_yes = book.yes_asks[0].price
        best_no = book.no_asks[0].price
        gross_comb = best_yes + best_no

        if gross_comb >= 1.0:
            return None

        # Determine executable capacity
        max_size_yes = sum(lvl.size for lvl in book.yes_asks)
        max_size_no = sum(lvl.size for lvl in book.no_asks)
        available_cap = min(max_size_yes, max_size_no)

        size_to_test = target_size if target_size is not None else min(available_cap, 5000.0)
        if size_to_test <= 0:
            return None

        exec_yes = self.walk_order_book(book.yes_asks, size_to_test, book.fee_rate, book.gas_cost_usd / 2.0)
        exec_no = self.walk_order_book(book.no_asks, size_to_test, book.fee_rate, book.gas_cost_usd / 2.0)

        actual_size = min(exec_yes.executed_size, exec_no.executed_size)
        if actual_size <= 0:
            return None

        eff_comb = exec_yes.effective_price + exec_no.effective_price
        total_spent = exec_yes.total_cost_usd + exec_no.total_cost_usd
        guaranteed_payout = actual_size * 1.0  # Exactly one of Yes/No pays $1.00
        net_profit = guaranteed_payout - total_spent
        net_roi = net_profit / max(total_spent, 1e-6)

        return ArbitrageOpportunity(
            contract_id=book.contract_id,
            arb_type=ArbitrageType.INTRA_VENUE,
            leg_yes_venue=book.venue,
            leg_no_venue=book.venue,
            leg_yes_price=best_yes,
            leg_no_price=best_no,
            gross_combined_price=gross_comb,
            gross_edge_pct=(1.0 - gross_comb) * 100.0,
            max_executable_size=actual_size,
            depth_weighted_yes_price=exec_yes.effective_price,
            depth_weighted_no_price=exec_no.effective_price,
            effective_combined_price=eff_comb,
            net_profit_usd=net_profit,
            net_roi_pct=net_roi * 100.0,
            total_capital_required=total_spent,
            yes_execution=exec_yes,
            no_execution=exec_no,
            is_profitable=(net_roi >= self.min_net_roi_threshold),
        )

    def check_cross_venue_arbitrage(
        self,
        book_a: BinaryContractOrderBook,
        book_b: BinaryContractOrderBook,
        target_size: Optional[float] = None,
    ) -> List[ArbitrageOpportunity]:
        """Evaluates cross-venue arbitrage across two competing venues."""
        opportunities = []

        # Combo 1: Buy Yes on A + Buy No on B
        if book_a.yes_asks and book_b.no_asks:
            best_yes_a = book_a.yes_asks[0].price
            best_no_b = book_b.no_asks[0].price
            comb_1 = best_yes_a + best_no_b

            if comb_1 < 1.0:
                max_cap = min(sum(l.size for l in book_a.yes_asks), sum(l.size for l in book_b.no_asks))
                size = target_size if target_size is not None else min(max_cap, 5000.0)

                exec_yes = self.walk_order_book(book_a.yes_asks, size, book_a.fee_rate, book_a.gas_cost_usd)
                exec_no = self.walk_order_book(book_b.no_asks, size, book_b.fee_rate, book_b.gas_cost_usd)

                act_size = min(exec_yes.executed_size, exec_no.executed_size)
                if act_size > 0:
                    tot_spent = exec_yes.total_cost_usd + exec_no.total_cost_usd
                    net_profit = (act_size * 1.0) - tot_spent
                    net_roi = net_profit / max(tot_spent, 1e-6)

                    opportunities.append(
                        ArbitrageOpportunity(
                            contract_id=book_a.contract_id,
                            arb_type=ArbitrageType.CROSS_VENUE,
                            leg_yes_venue=book_a.venue,
                            leg_no_venue=book_b.venue,
                            leg_yes_price=best_yes_a,
                            leg_no_price=best_no_b,
                            gross_combined_price=comb_1,
                            gross_edge_pct=(1.0 - comb_1) * 100.0,
                            max_executable_size=act_size,
                            depth_weighted_yes_price=exec_yes.effective_price,
                            depth_weighted_no_price=exec_no.effective_price,
                            effective_combined_price=exec_yes.effective_price + exec_no.effective_price,
                            net_profit_usd=net_profit,
                            net_roi_pct=net_roi * 100.0,
                            total_capital_required=tot_spent,
                            yes_execution=exec_yes,
                            no_execution=exec_no,
                            is_profitable=(net_roi >= self.min_net_roi_threshold),
                        )
                    )

        # Combo 2: Buy Yes on B + Buy No on A
        if book_b.yes_asks and book_a.no_asks:
            best_yes_b = book_b.yes_asks[0].price
            best_no_a = book_a.no_asks[0].price
            comb_2 = best_yes_b + best_no_a

            if comb_2 < 1.0:
                max_cap = min(sum(l.size for l in book_b.yes_asks), sum(l.size for l in book_a.no_asks))
                size = target_size if target_size is not None else min(max_cap, 5000.0)

                exec_yes = self.walk_order_book(book_b.yes_asks, size, book_b.fee_rate, book_b.gas_cost_usd)
                exec_no = self.walk_order_book(book_a.no_asks, size, book_a.fee_rate, book_a.gas_cost_usd)

                act_size = min(exec_yes.executed_size, exec_no.executed_size)
                if act_size > 0:
                    tot_spent = exec_yes.total_cost_usd + exec_no.total_cost_usd
                    net_profit = (act_size * 1.0) - tot_spent
                    net_roi = net_profit / max(tot_spent, 1e-6)

                    opportunities.append(
                        ArbitrageOpportunity(
                            contract_id=book_a.contract_id,
                            arb_type=ArbitrageType.CROSS_VENUE,
                            leg_yes_venue=book_b.venue,
                            leg_no_venue=book_a.venue,
                            leg_yes_price=best_yes_b,
                            leg_no_price=best_no_a,
                            gross_combined_price=comb_2,
                            gross_edge_pct=(1.0 - comb_2) * 100.0,
                            max_executable_size=act_size,
                            depth_weighted_yes_price=exec_yes.effective_price,
                            depth_weighted_no_price=exec_no.effective_price,
                            effective_combined_price=exec_yes.effective_price + exec_no.effective_price,
                            net_profit_usd=net_profit,
                            net_roi_pct=net_roi * 100.0,
                            total_capital_required=tot_spent,
                            yes_execution=exec_yes,
                            no_execution=exec_no,
                            is_profitable=(net_roi >= self.min_net_roi_threshold),
                        )
                    )

        return opportunities

    def calculate_kelly_fraction(
        self,
        true_win_prob: float,
        market_price: float,
        bankroll_usd: float = 100_000.0,
        fractional_multiplier: Optional[float] = None,
    ) -> KellyAllocationResult:
        """Calculates exact Continuous and Discrete Kelly Criterion allocation."""
        mult = fractional_multiplier if fractional_multiplier is not None else self.fractional_kelly
        p = np.clip(true_win_prob, 0.001, 0.999)
        price = np.clip(market_price, 0.01, 0.99)
        
        # Net odds b = (payout - price) / price = (1 - price) / price
        b = (1.0 - price) / price
        q = 1.0 - p
        
        # Full Kelly f* = (b*p - q) / b = (p - price) / (1 - price)
        full_kelly = (b * p - q) / b
        full_kelly = float(np.clip(full_kelly, 0.0, 1.0))
        
        recommended_f = float(np.clip(full_kelly * mult, 0.0, 1.0))
        stake = recommended_f * bankroll_usd
        
        # Expected geometric growth rate g = p * ln(1 + f*b) + (1-p) * ln(1 - f)
        if recommended_f > 0:
            growth = p * np.log(1.0 + recommended_f * b) + q * np.log(max(1e-6, 1.0 - recommended_f))
        else:
            growth = 0.0

        return KellyAllocationResult(
            full_kelly_fraction=full_kelly,
            fractional_kelly_multiplier=mult,
            recommended_fraction=recommended_f,
            recommended_stake_usd=stake,
            expected_growth_rate=float(growth),
            win_probability=p,
            net_odds=float(b),
            bankroll_usd=bankroll_usd,
        )

    def calculate_latency_alpha_decay(
        self,
        initial_edge_pct: float,
        elapsed_sec: float,
        half_life_sec: Optional[float] = None,
    ) -> float:
        """Computes exponential alpha decay of executable edge over latency interval."""
        hl = half_life_sec if half_life_sec is not None else self.latency_half_life_sec
        tau = hl / np.log(2.0)
        decayed_edge = initial_edge_pct * np.exp(-elapsed_sec / max(tau, 1e-4))
        return float(decayed_edge)
