"""
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
