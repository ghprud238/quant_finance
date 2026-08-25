"""Constant Function Market Makers & Uniswap v2/v3 Concentrated Liquidity Engine (Project 41).

Implements:
1. Uniswap v2 Constant Product Market Maker (x * y = k) with fee dynamics and price impact.
2. Uniswap v3 Concentrated Liquidity AMM with tick intervals, virtual reserves, and capital efficiency.
3. Curve Stableswap Invariant with Amplification coefficient A.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class SwapResult:
    """Result of an AMM swap execution."""
    amount_in: float
    amount_out: float
    token_in: str
    token_out: str
    fee_paid: float
    price_before: float
    price_after: float
    execution_price: float
    price_impact_pct: float
    new_reserve_x: float
    new_reserve_y: float


@dataclass
class PositionV3:
    """Uniswap v3 LP Position."""
    position_id: str
    owner: str
    lower_tick: int
    upper_tick: int
    price_lower: float
    price_upper: float
    liquidity: float
    amount_x: float
    amount_y: float
    fee_growth_inside_x: float = 0.0
    fee_growth_inside_y: float = 0.0
    uncollected_fees_x: float = 0.0
    uncollected_fees_y: float = 0.0


@dataclass
class TickInfo:
    """Information stored at a discrete Uniswap v3 tick."""
    tick_idx: int
    price: float
    sqrt_price: float
    liquidity_gross: float = 0.0
    liquidity_net: float = 0.0  # +L when crossed left-to-right, -L right-to-left
    fee_growth_outside_x: float = 0.0
    fee_growth_outside_y: float = 0.0


# =========================================================================
# 1. UNISWAP V2 CONSTANT PRODUCT AMM (x * y = k)
# =========================================================================

class ConstantProductAMM:
    """Uniswap v2 Constant Product Automated Market Maker (x * y = k)."""
    
    def __init__(
        self,
        reserve_x: float = 1000.0,
        reserve_y: float = 3000000.0,
        token_x_symbol: str = "ETH",
        token_y_symbol: str = "USDC",
        fee_rate: float = 0.0030,  # 0.30% standard Uniswap v2 fee
    ):
        if reserve_x <= 0 or reserve_y <= 0:
            raise ValueError("Reserves must be strictly positive.")
        self.reserve_x = float(reserve_x)
        self.reserve_y = float(reserve_y)
        self.token_x = token_x_symbol
        self.token_y = token_y_symbol
        self.fee_rate = float(fee_rate)
        self.total_lp_shares = np.sqrt(self.reserve_x * self.reserve_y)
        self.k = self.reserve_x * self.reserve_y
        
    @property
    def spot_price_y_per_x(self) -> float:
        """Instantaneous marginal spot price of Token X in terms of Token Y (e.g. USDC per ETH)."""
        return self.reserve_y / self.reserve_x
        
    @property
    def spot_price_x_per_y(self) -> float:
        """Instantaneous marginal spot price of Token Y in terms of Token X."""
        return self.reserve_x / self.reserve_y

    def get_amount_out(self, amount_in: float, token_in: str) -> Tuple[float, float]:
        """Calculates exact output amount and fee paid given input amount and fee factor gamma = 1 - f.
        
        Formula:
            Delta_y = (y * gamma * Delta_x) / (x + gamma * Delta_x)
        """
        if amount_in <= 0:
            raise ValueError("Amount in must be positive.")
            
        gamma = 1.0 - self.fee_rate
        amount_in_with_fee = amount_in * gamma
        
        if token_in == self.token_x:
            numerator = amount_in_with_fee * self.reserve_y
            denominator = self.reserve_x + amount_in_with_fee
            amount_out = numerator / denominator
            fee_paid = amount_in * self.fee_rate
        elif token_in == self.token_y:
            numerator = amount_in_with_fee * self.reserve_x
            denominator = self.reserve_y + amount_in_with_fee
            amount_out = numerator / denominator
            fee_paid = amount_in * self.fee_rate
        else:
            raise ValueError(f"Unknown token {token_in}. Pool contains {self.token_x} and {self.token_y}.")
            
        return amount_out, fee_paid

    def swap(self, amount_in: float, token_in: str) -> SwapResult:
        """Executes a swap on the Constant Product pool and updates state."""
        p_before = self.spot_price_y_per_x
        amount_out, fee_paid = self.get_amount_out(amount_in, token_in)
        
        if token_in == self.token_x:
            token_out = self.token_y
            self.reserve_x += amount_in
            self.reserve_y -= amount_out
            exec_price = amount_out / amount_in  # USDC per ETH received
            impact_pct = (p_before - exec_price) / p_before * 100.0
        else:
            token_out = self.token_x
            self.reserve_y += amount_in
            self.reserve_x -= amount_out
            exec_price = amount_in / amount_out  # Effective USDC paid per ETH
            impact_pct = (exec_price - p_before) / p_before * 100.0
            
        p_after = self.spot_price_y_per_x
        self.k = self.reserve_x * self.reserve_y
        
        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            token_in=token_in,
            token_out=token_out,
            fee_paid=fee_paid,
            price_before=p_before,
            price_after=p_after,
            execution_price=exec_price,
            price_impact_pct=impact_pct,
            new_reserve_x=self.reserve_x,
            new_reserve_y=self.reserve_y,
        )

    def add_liquidity(self, amount_x: float, amount_y: float) -> Tuple[float, float, float]:
        """Adds liquidity to the pool preserving the price ratio, minting LP tokens."""
        if amount_x <= 0 or amount_y <= 0:
            raise ValueError("Amounts must be positive.")
            
        optimal_y = amount_x * (self.reserve_y / self.reserve_x)
        if amount_y < optimal_y * 0.999:
            raise ValueError(f"Insufficient {self.token_y} provided. Required: {optimal_y:.4f}, Provided: {amount_y:.4f}")
            
        shares_minted = self.total_lp_shares * (amount_x / self.reserve_x)
        self.reserve_x += amount_x
        self.reserve_y += optimal_y
        self.total_lp_shares += shares_minted
        self.k = self.reserve_x * self.reserve_y
        return shares_minted, amount_x, optimal_y

    def remove_liquidity(self, shares: float) -> Tuple[float, float]:
        """Burns LP shares to withdraw proportional reserves."""
        if shares <= 0 or shares > self.total_lp_shares:
            raise ValueError("Invalid shares amount.")
        share_ratio = shares / self.total_lp_shares
        amount_x = self.reserve_x * share_ratio
        amount_y = self.reserve_y * share_ratio
        
        self.reserve_x -= amount_x
        self.reserve_y -= amount_y
        self.total_lp_shares -= shares
        self.k = self.reserve_x * self.reserve_y
        return amount_x, amount_y


# =========================================================================
# 2. UNISWAP V3 CONCENTRATED LIQUIDITY AMM
# =========================================================================

class ConcentratedLiquidityAMM:
    """Uniswap v3 Concentrated Liquidity Automated Market Maker.
    
    Operates on tick space p(i) = 1.0001^i with virtual reserves and step-wise tick crossing.
    """
    
    def __init__(
        self,
        current_price: float = 3000.0,
        fee_tier: float = 0.0030,  # 0.30%
        token_x_symbol: str = "ETH",
        token_y_symbol: str = "USDC",
    ):
        self.current_price = float(current_price)
        self.sqrt_price = np.sqrt(self.current_price)
        self.current_tick = self.price_to_tick(self.current_price)
        self.fee_tier = float(fee_tier)
        self.token_x = token_x_symbol
        self.token_y = token_y_symbol
        
        # State
        self.liquidity = 0.0  # Currently active in-range liquidity L
        self.ticks: Dict[int, TickInfo] = {}
        self.positions: Dict[str, PositionV3] = {}
        self.fee_growth_global_x = 0.0
        self.fee_growth_global_y = 0.0

    @staticmethod
    def price_to_tick(price: float) -> int:
        """Converts price to the nearest integer tick: i = floor(ln(P) / ln(1.0001))."""
        return int(np.floor(np.log(price) / np.log(1.0001)))

    @staticmethod
    def tick_to_price(tick: int) -> float:
        """Converts integer tick to price: P = 1.0001^i."""
        return 1.0001 ** tick

    @staticmethod
    def tick_to_sqrt_price(tick: int) -> float:
        """Converts integer tick to sqrt(Price) = 1.0001^(i/2)."""
        return 1.0001 ** (tick / 2.0)

    @classmethod
    def calculate_liquidity(
        cls,
        amount_x: float,
        amount_y: float,
        price_current: float,
        price_lower: float,
        price_upper: float,
    ) -> float:
        """Computes active liquidity L from deposit amounts according to Uniswap v3 formulas."""
        if price_lower >= price_upper:
            raise ValueError("price_lower must be strictly less than price_upper.")
            
        sqrt_p = np.sqrt(price_current)
        sqrt_a = np.sqrt(price_lower)
        sqrt_b = np.sqrt(price_upper)
        
        if sqrt_p <= sqrt_a:
            # Current price below range: position is 100% Token X
            # L = Delta_x * (sqrt(P_a) * sqrt(P_b)) / (sqrt(P_b) - sqrt(P_a))
            l_x = amount_x * (sqrt_a * sqrt_b) / (sqrt_b - sqrt_a)
            return l_x
        elif sqrt_p >= sqrt_b:
            # Current price above range: position is 100% Token Y
            # L = Delta_y / (sqrt(P_b) - sqrt(P_a))
            l_y = amount_y / (sqrt_b - sqrt_a)
            return l_y
        else:
            # In range: position holds both tokens
            l_x = amount_x * (sqrt_p * sqrt_b) / (sqrt_b - sqrt_p) if amount_x > 0 else float("inf")
            l_y = amount_y / (sqrt_p - sqrt_a) if amount_y > 0 else float("inf")
            return min(l_x, l_y)

    @classmethod
    def calculate_amounts_for_liquidity(
        cls,
        liquidity: float,
        price_current: float,
        price_lower: float,
        price_upper: float,
    ) -> Tuple[float, float]:
        """Computes real Token X and Token Y balances for given liquidity L and price range."""
        sqrt_p = np.sqrt(price_current)
        sqrt_a = np.sqrt(price_lower)
        sqrt_b = np.sqrt(price_upper)
        
        if sqrt_p <= sqrt_a:
            amount_x = liquidity * (sqrt_b - sqrt_a) / (sqrt_a * sqrt_b)
            amount_y = 0.0
        elif sqrt_p >= sqrt_b:
            amount_x = 0.0
            amount_y = liquidity * (sqrt_b - sqrt_a)
        else:
            amount_x = liquidity * (sqrt_b - sqrt_p) / (sqrt_p * sqrt_b)
            amount_y = liquidity * (sqrt_p - sqrt_a)
            
        return amount_x, amount_y

    @classmethod
    def capital_efficiency_multiplier(
        cls, price_lower: float, price_upper: float
    ) -> float:
        """Calculates capital efficiency multiplier relative to full-range Uniswap v2:
        
        Multiplier = 1 / (1 - sqrt(P_a / P_b))
        """
        if price_lower >= price_upper:
            raise ValueError("price_lower must be strictly less than price_upper.")
        ratio = np.sqrt(price_lower / price_upper)
        return 1.0 / (1.0 - ratio)

    def mint_position(
        self,
        owner: str,
        price_lower: float,
        price_upper: float,
        amount_x: float,
        amount_y: float,
        position_id: Optional[str] = None,
    ) -> PositionV3:
        """Mints a concentrated liquidity position across [price_lower, price_upper]."""
        lower_tick = self.price_to_tick(price_lower)
        upper_tick = self.price_to_tick(price_upper)
        
        # Exact tick boundary prices
        p_a = self.tick_to_price(lower_tick)
        p_b = self.tick_to_price(upper_tick)
        
        L = self.calculate_liquidity(amount_x, amount_y, self.current_price, p_a, p_b)
        real_x, real_y = self.calculate_amounts_for_liquidity(L, self.current_price, p_a, p_b)
        
        if position_id is None:
            position_id = f"POS_{owner}_{lower_tick}_{upper_tick}"
            
        # Update tick net liquidity
        if lower_tick not in self.ticks:
            self.ticks[lower_tick] = TickInfo(
                tick_idx=lower_tick,
                price=p_a,
                sqrt_price=np.sqrt(p_a),
            )
        if upper_tick not in self.ticks:
            self.ticks[upper_tick] = TickInfo(
                tick_idx=upper_tick,
                price=p_b,
                sqrt_price=np.sqrt(p_b),
            )
            
        self.ticks[lower_tick].liquidity_gross += L
        self.ticks[lower_tick].liquidity_net += L  # Entered from below
        
        self.ticks[upper_tick].liquidity_gross += L
        self.ticks[upper_tick].liquidity_net -= L  # Exited from below
        
        # If currently active, add to active liquidity
        if lower_tick <= self.current_tick < upper_tick:
            self.liquidity += L
            
        pos = PositionV3(
            position_id=position_id,
            owner=owner,
            lower_tick=lower_tick,
            upper_tick=upper_tick,
            price_lower=p_a,
            price_upper=p_b,
            liquidity=L,
            amount_x=real_x,
            amount_y=real_y,
        )
        self.positions[position_id] = pos
        return pos

    def swap(self, amount_in: float, token_in: str) -> SwapResult:
        """Executes a step-wise swap crossing active tick boundaries with virtual reserves."""
        if amount_in <= 0:
            raise ValueError("amount_in must be strictly positive.")
            
        zero_for_one = (token_in == self.token_x)
        gamma = 1.0 - self.fee_tier
        
        amount_specified_remaining = amount_in
        amount_calculated = 0.0
        fee_paid_total = 0.0
        price_before = self.current_price
        
        # Sort ticks for sequential crossing
        sorted_ticks = sorted(self.ticks.keys())
        
        while amount_specified_remaining > 1e-9 and self.liquidity > 0:
            current_sqrt_p = self.sqrt_price
            
            # Find next initialized tick target
            if zero_for_one:
                # Selling X -> Price decreases -> Next tick is below current
                eligible_ticks = [t for t in sorted_ticks if t < self.current_tick]
                next_tick = eligible_ticks[-1] if eligible_ticks else self.current_tick - 500
                next_sqrt_p = self.tick_to_sqrt_price(next_tick)
                
                # Max amount X needed to reach next tick: Delta_x = L * (1/sqrt(P_next) - 1/sqrt(P_curr))
                max_dx = self.liquidity * (1.0 / next_sqrt_p - 1.0 / current_sqrt_p)
                amount_in_with_fee = amount_specified_remaining * gamma
                
                if amount_in_with_fee >= max_dx:
                    # Cross the tick fully
                    step_amount_in = max_dx / gamma
                    step_amount_out = self.liquidity * (current_sqrt_p - next_sqrt_p)
                    step_fee = step_amount_in * self.fee_tier
                    
                    self.sqrt_price = next_sqrt_p
                    self.current_tick = next_tick
                    self.current_price = next_sqrt_p ** 2
                    
                    # Update active liquidity (subtract liquidity_net because crossing right-to-left)
                    if next_tick in self.ticks:
                        self.liquidity -= self.ticks[next_tick].liquidity_net
                        
                    amount_specified_remaining -= step_amount_in
                    amount_calculated += step_amount_out
                    fee_paid_total += step_fee
                else:
                    # Partial step within current tick interval
                    # New sqrt_p: 1 / sqrt_p_new = 1 / sqrt_p_curr + Delta_x_fee / L
                    target_inv_sqrt = (1.0 / current_sqrt_p) + (amount_in_with_fee / self.liquidity)
                    new_sqrt_p = 1.0 / target_inv_sqrt
                    step_amount_out = self.liquidity * (current_sqrt_p - new_sqrt_p)
                    step_fee = amount_specified_remaining * self.fee_tier
                    
                    self.sqrt_price = new_sqrt_p
                    self.current_price = new_sqrt_p ** 2
                    self.current_tick = self.price_to_tick(self.current_price)
                    
                    amount_calculated += step_amount_out
                    fee_paid_total += step_fee
                    amount_specified_remaining = 0.0
            else:
                # Buying X (Selling Y) -> Price increases -> Next tick is above current
                eligible_ticks = [t for t in sorted_ticks if t > self.current_tick]
                next_tick = eligible_ticks[0] if eligible_ticks else self.current_tick + 500
                next_sqrt_p = self.tick_to_sqrt_price(next_tick)
                
                # Max amount Y needed to reach next tick: Delta_y = L * (sqrt(P_next) - sqrt(P_curr))
                max_dy = self.liquidity * (next_sqrt_p - current_sqrt_p)
                amount_in_with_fee = amount_specified_remaining * gamma
                
                if amount_in_with_fee >= max_dy:
                    step_amount_in = max_dy / gamma
                    step_amount_out = self.liquidity * (1.0 / current_sqrt_p - 1.0 / next_sqrt_p)
                    step_fee = step_amount_in * self.fee_tier
                    
                    self.sqrt_price = next_sqrt_p
                    self.current_tick = next_tick
                    self.current_price = next_sqrt_p ** 2
                    
                    # Update active liquidity (add liquidity_net because crossing left-to-right)
                    if next_tick in self.ticks:
                        self.liquidity += self.ticks[next_tick].liquidity_net
                        
                    amount_specified_remaining -= step_amount_in
                    amount_calculated += step_amount_out
                    fee_paid_total += step_fee
                else:
                    new_sqrt_p = current_sqrt_p + (amount_in_with_fee / self.liquidity)
                    step_amount_out = self.liquidity * (1.0 / current_sqrt_p - 1.0 / new_sqrt_p)
                    step_fee = amount_specified_remaining * self.fee_tier
                    
                    self.sqrt_price = new_sqrt_p
                    self.current_price = new_sqrt_p ** 2
                    self.current_tick = self.price_to_tick(self.current_price)
                    
                    amount_calculated += step_amount_out
                    fee_paid_total += step_fee
                    amount_specified_remaining = 0.0
                    
        price_after = self.current_price
        exec_price = (amount_calculated / amount_in) if zero_for_one else (amount_in / amount_calculated)
        impact_pct = abs(price_after - price_before) / price_before * 100.0
        
        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_calculated,
            token_in=token_in,
            token_out=self.token_y if zero_for_one else self.token_x,
            fee_paid=fee_paid_total,
            price_before=price_before,
            price_after=price_after,
            execution_price=exec_price,
            price_impact_pct=impact_pct,
            new_reserve_x=0.0,
            new_reserve_y=0.0,
        )


# =========================================================================
# 3. CURVE STABLESWAP AMM
# =========================================================================

class StableswapAMM:
    """Curve Finance Stableswap Invariant AMM for pegged/correlated assets.
    
    Invariant equation:
        A * n^n * sum(x_i) + D = A * D * n^n + D^(n+1) / (n^n * prod(x_i))
    """
    
    def __init__(
        self,
        reserves: List[float],
        token_symbols: Optional[List[str]] = None,
        A: float = 100.0,  # Amplification coefficient
        fee_rate: float = 0.0004,  # 0.04% typical Curve fee
    ):
        if len(reserves) < 2:
            raise ValueError("Stableswap pool requires at least 2 tokens.")
        self.reserves = np.array(reserves, dtype=float)
        self.n = len(reserves)
        self.token_symbols = token_symbols or [f"TOKEN_{i}" for i in range(self.n)]
        self.A = float(A)
        self.fee_rate = float(fee_rate)
        self.D = self.get_D(self.reserves, self.A)

    @classmethod
    def get_D(cls, xp: np.ndarray, A: float, max_iter: int = 255) -> float:
        """Solves for the invariant D using Newton-Raphson iteration."""
        n = len(xp)
        S = np.sum(xp)
        if S == 0:
            return 0.0
            
        D = S
        Ann = A * (n ** n)
        
        for _ in range(max_iter):
            D_P = D
            for x in xp:
                D_P = D_P * D / (x * n)  # D^(n+1) / (n^n * prod(x))
                
            Dprev = D
            numerator = D * (Ann * S + D_P * n)
            denominator = (Ann - 1.0) * D + (n + 1.0) * D_P
            D = numerator / denominator
            
            if abs(D - Dprev) <= 1e-6:
                break
                
        return D

    @classmethod
    def get_y(cls, i: int, j: int, x: float, xp: np.ndarray, A: float, D: float, max_iter: int = 255) -> float:
        """Calculates reserve of token j when reserve of token i changes to x."""
        n = len(xp)
        Ann = A * (n ** n)
        S_ = 0.0
        c = D
        
        for k in range(n):
            if k == i:
                _x = x
            elif k != j:
                _x = xp[k]
            else:
                continue
            S_ += _x
            c = c * D / (_x * n)
            
        c = c * D / (Ann * n)
        b = S_ + D / Ann
        y = D
        
        for _ in range(max_iter):
            y_prev = y
            y = (y * y + c) / (2.0 * y + b - D)
            if abs(y - y_prev) <= 1e-6:
                break
                
        return y

    def swap(self, i: int, j: int, dx: float) -> float:
        """Executes a swap from token i to token j for input dx, returning net dy after fees."""
        if dx <= 0:
            raise ValueError("dx must be positive.")
        if i == j or i < 0 or j < 0 or i >= self.n or j >= self.n:
            raise ValueError("Invalid token indices.")
            
        xp = self.reserves.copy()
        x_new = xp[i] + dx
        y_new = self.get_y(i, j, x_new, xp, self.A, self.D)
        
        dy = xp[j] - y_new
        fee = dy * self.fee_rate
        dy_net = dy - fee
        
        self.reserves[i] = x_new
        self.reserves[j] = xp[j] - dy_net
        self.D = self.get_D(self.reserves, self.A)
        return dy_net
