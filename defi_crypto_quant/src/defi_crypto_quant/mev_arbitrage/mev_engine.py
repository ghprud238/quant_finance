"""Cross-DEX Flash Loans, Triangular Arbitrage & MEV Searcher Engine (Project 43).

Implements mathematical models for AMM Constant Product Market Makers (CPMM),
optimal closed-form spatial arbitrage trade sizing, Bellman-Ford negative cycle
triangular arbitrage search, and atomic MEV sandwich attack simulation.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq


class PoolType(Enum):
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    SUSHISWAP = "sushiswap"
    CURVE = "curve"


@dataclass
class LiquidityPool:
    """Constant Product Automated Market Maker (CPMM) Liquidity Pool."""
    name: str
    pool_type: PoolType
    token_a: str
    token_b: str
    reserve_a: float
    reserve_b: float
    fee: float = 0.003  # 0.30% standard fee (gamma = 1 - fee = 0.997)

    @property
    def gamma(self) -> float:
        """Fee multiplier gamma = 1 - fee."""
        return 1.0 - self.fee

    @property
    def k(self) -> float:
        """Invariant k = reserve_a * reserve_b."""
        return self.reserve_a * self.reserve_b

    def get_reserve(self, token: str) -> float:
        """Returns reserve for a given token symbol."""
        if token == self.token_a:
            return self.reserve_a
        elif token == self.token_b:
            return self.reserve_b
        raise ValueError(f"Token {token} not in pool {self.name} ({self.token_a}/{self.token_b})")

    def get_other_token(self, token: str) -> str:
        """Returns the counterpart token in the pair."""
        if token == self.token_a:
            return self.token_b
        elif token == self.token_b:
            return self.token_a
        raise ValueError(f"Token {token} not in pool {self.name}")

    def get_spot_price(self, base_token: str, quote_token: str) -> float:
        """Returns the instantaneous marginal spot price of base in units of quote."""
        r_base = self.get_reserve(base_token)
        r_quote = self.get_reserve(quote_token)
        return r_quote / r_base

    def get_amount_out(self, token_in: str, amount_in: float) -> float:
        """Computes exact output amount from constant product formula with fees:
        Delta_y = (R_out * gamma * Delta_x) / (R_in + gamma * Delta_x)
        """
        if amount_in <= 0.0:
            return 0.0
        r_in = self.get_reserve(token_in)
        r_out = self.get_reserve(self.get_other_token(token_in))
        
        amount_in_with_fee = amount_in * self.gamma
        numerator = r_out * amount_in_with_fee
        denominator = r_in + amount_in_with_fee
        return numerator / denominator

    def get_amount_in(self, token_out: str, amount_out: float) -> float:
        """Computes required input amount for desired output amount:
        Delta_x = (R_in * Delta_y) / (gamma * (R_out - Delta_y))
        """
        r_out = self.get_reserve(token_out)
        r_in = self.get_reserve(self.get_other_token(token_out))
        if amount_out >= r_out:
            raise ValueError(f"Insufficient liquidity: requested {amount_out} >= reserve {r_out}")
        
        numerator = r_in * amount_out
        denominator = self.gamma * (r_out - amount_out)
        return numerator / denominator

    def execute_swap(self, token_in: str, amount_in: float) -> float:
        """Mutates pool state and returns output amount."""
        amount_out = self.get_amount_out(token_in, amount_in)
        token_out = self.get_other_token(token_in)
        
        if token_in == self.token_a:
            self.reserve_a += amount_in
            self.reserve_b -= amount_out
        else:
            self.reserve_b += amount_in
            self.reserve_a -= amount_out
        return amount_out

    def clone(self) -> "LiquidityPool":
        """Deep copy for non-mutating sandbox simulation."""
        return LiquidityPool(
            name=self.name,
            pool_type=self.pool_type,
            token_a=self.token_a,
            token_b=self.token_b,
            reserve_a=self.reserve_a,
            reserve_b=self.reserve_b,
            fee=self.fee,
        )


@dataclass
class SpatialArbitrageResult:
    """Structured result for cross-DEX spatial arbitrage."""
    pool_buy: str
    pool_sell: str
    token_borrow: str
    token_target: str
    optimal_input: float
    intermediate_output: float
    final_output: float
    gross_profit: float
    flash_loan_fee: float
    gas_cost: float
    net_profit: float
    return_on_capital_pct: float
    is_profitable: bool
    price_pool_buy: float
    price_pool_sell: float

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "Pool Buy": self.pool_buy,
            "Pool Sell": self.pool_sell,
            "Borrow Token": self.token_borrow,
            "Optimal Input": f"{self.optimal_input:,.4f}",
            "Gross Profit": f"{self.gross_profit:+,.4f}",
            "Flash Loan Fee": f"{self.flash_loan_fee:,.4f}",
            "Gas Cost": f"${self.gas_cost:,.2f}",
            "Net Profit": f"{self.net_profit:+,.4f}",
            "Net RoC (%)": f"{self.return_on_capital_pct:+.2f}%",
            "Profitable": self.is_profitable,
        }


@dataclass
class TriangularArbitragePath:
    """Triangular Arbitrage cycle result."""
    path_tokens: List[str]
    pools: List[str]
    cycle_multiplier: float
    optimal_input: float
    expected_output: float
    net_profit: float
    profit_pct: float
    is_profitable: bool


@dataclass
class SandwichResult:
    """Structured result for an MEV Sandwich attack simulation."""
    victim_tx_hash: str
    victim_input_amount: float
    victim_token_in: str
    victim_token_out: str
    frontrun_amount_in: float
    frontrun_amount_out: float
    victim_received_without_sandwich: float
    victim_received_with_sandwich: float
    victim_slippage_drag_pct: float
    backrun_amount_in: float
    backrun_amount_out: float
    gross_mev_profit: float
    gas_cost_usd: float
    builder_bribe_usd: float
    net_searcher_profit: float
    pool_initial_price: float
    pool_post_frontrun_price: float
    pool_post_victim_price: float
    pool_final_price: float


class CrossDEXArbitrageEngine:
    """Core engine for spatial cross-DEX arbitrage with flash loans."""

    def __init__(
        self,
        pools: Optional[Dict[str, LiquidityPool]] = None,
        default_flash_loan_fee: float = 0.0009,  # 0.09% Aave standard
        eth_price_usd: float = 3000.0,
    ):
        self.pools = pools or {}
        self.default_flash_loan_fee = default_flash_loan_fee
        self.eth_price_usd = eth_price_usd

    def add_pool(self, pool: LiquidityPool) -> None:
        self.pools[pool.name] = pool

    def compute_closed_form_optimal_input(
        self,
        pool1: LiquidityPool,
        pool2: LiquidityPool,
        token_borrow: str,
    ) -> float:
        """Calculates theoretical closed-form optimal input Delta_x* for two CPMM pools:
        Delta_x* = (sqrt(x1 * x2 * y1 * y2 * gamma1 * gamma2) - x1 * y2) / (gamma1 * y2 + gamma1 * gamma2 * y1)
        """
        target_token = pool1.get_other_token(token_borrow)
        x1 = pool1.get_reserve(token_borrow)
        y1 = pool1.get_reserve(target_token)
        gamma1 = pool1.gamma

        # Pool 2: Input is target_token, Output is token_borrow
        y2 = pool2.get_reserve(target_token)
        x2 = pool2.get_reserve(token_borrow)
        gamma2 = pool2.gamma

        numerator = math.sqrt(x1 * x2 * y1 * y2 * gamma1 * gamma2) - (x1 * y2)
        denominator = (gamma1 * y2) + (gamma1 * gamma2 * y1)

        if denominator <= 0.0 or numerator <= 0.0:
            return 0.0
        return max(0.0, numerator / denominator)

    def evaluate_spatial_arbitrage(
        self,
        pool1: LiquidityPool,
        pool2: LiquidityPool,
        token_borrow: str,
        gas_units: int = 250_000,
        gas_price_gwei: float = 30.0,
        flash_loan_fee_pct: Optional[float] = None,
    ) -> SpatialArbitrageResult:
        """Evaluates spatial arbitrage between pool1 (buy intermediate) and pool2 (sell intermediate)."""
        loan_fee_pct = flash_loan_fee_pct if flash_loan_fee_pct is not None else self.default_flash_loan_fee
        target_token = pool1.get_other_token(token_borrow)

        # Gas cost calculation in USD
        gas_cost_eth = (gas_units * gas_price_gwei * 1e-9)
        gas_cost_usd = gas_cost_eth * self.eth_price_usd
        
        # Convert gas cost to borrow token units
        if token_borrow in ["USDC", "USDT", "DAI"]:
            gas_cost_in_token = gas_cost_usd
        elif token_borrow in ["WETH", "ETH"]:
            gas_cost_in_token = gas_cost_eth
        else:
            gas_cost_in_token = gas_cost_usd / pool1.get_spot_price(token_borrow, "USDC") if "USDC" in [pool1.token_a, pool1.token_b] else gas_cost_eth

        # Objective function for profit maximization
        def objective(dx: float) -> float:
            if dx <= 0.0:
                return 0.0
            p1_sim = pool1.clone()
            p2_sim = pool2.clone()
            
            dy = p1_sim.get_amount_out(token_borrow, dx)
            dx_out = p2_sim.get_amount_out(target_token, dy)
            
            flash_fee = dx * loan_fee_pct
            net_pnl = dx_out - dx - flash_fee - gas_cost_in_token
            return -net_pnl  # minimize negative profit

        # Compute initial closed-form estimate as bound
        cf_dx = self.compute_closed_form_optimal_input(pool1, pool2, token_borrow)
        max_bound = min(pool1.get_reserve(token_borrow) * 0.5, pool2.get_reserve(token_borrow) * 0.5)

        if cf_dx > 0.0 and max_bound > 0.0:
            res = minimize_scalar(
                objective,
                bounds=(0.0, max(cf_dx * 2.0, max_bound)),
                method="bounded",
            )
            opt_dx = max(0.0, float(res.x)) if res.success else cf_dx
        else:
            opt_dx = 0.0

        if opt_dx > 0.0:
            p1_sim = pool1.clone()
            p2_sim = pool2.clone()
            inter_dy = p1_sim.get_amount_out(token_borrow, opt_dx)
            final_dx = p2_sim.get_amount_out(target_token, inter_dy)
            gross_pnl = final_dx - opt_dx
            loan_fee = opt_dx * loan_fee_pct
            net_pnl = gross_pnl - loan_fee - gas_cost_in_token
        else:
            inter_dy = 0.0
            final_dx = 0.0
            gross_pnl = 0.0
            loan_fee = 0.0
            net_pnl = 0.0

        roc = (net_pnl / opt_dx * 100.0) if opt_dx > 0.0 else 0.0

        return SpatialArbitrageResult(
            pool_buy=pool1.name,
            pool_sell=pool2.name,
            token_borrow=token_borrow,
            token_target=target_token,
            optimal_input=opt_dx,
            intermediate_output=inter_dy,
            final_output=final_dx,
            gross_profit=gross_pnl,
            flash_loan_fee=loan_fee,
            gas_cost=gas_cost_usd,
            net_profit=net_pnl,
            return_on_capital_pct=roc,
            is_profitable=(net_pnl > 0.0),
            price_pool_buy=pool1.get_spot_price(target_token, token_borrow),
            price_pool_sell=pool2.get_spot_price(target_token, token_borrow),
        )


class TriangularArbitrageSearcher:
    """Finds circular/triangular arbitrage paths using the Bellman-Ford negative cycle algorithm."""

    def __init__(self, pools: List[LiquidityPool]):
        self.pools = pools
        self.tokens = self._extract_unique_tokens()

    def _extract_unique_tokens(self) -> List[str]:
        toks = set()
        for p in self.pools:
            toks.add(p.token_a)
            toks.add(p.token_b)
        return sorted(list(toks))

    def build_exchange_rate_graph(self) -> Dict[str, Dict[str, Tuple[float, LiquidityPool]]]:
        """Builds directed graph of exchange rates with fees accounted for:
        Rate(A -> B) = get_amount_out(A, 1.0)
        """
        graph: Dict[str, Dict[str, Tuple[float, LiquidityPool]]] = {t: {} for t in self.tokens}
        for p in self.pools:
            # Token A -> Token B
            rate_a_to_b = p.get_amount_out(p.token_a, 1.0)
            if p.token_b not in graph[p.token_a] or rate_a_to_b > graph[p.token_a][p.token_b][0]:
                graph[p.token_a][p.token_b] = (rate_a_to_b, p)
            
            # Token B -> Token A
            rate_b_to_a = p.get_amount_out(p.token_b, 1.0)
            if p.token_a not in graph[p.token_b] or rate_b_to_a > graph[p.token_b][p.token_a][0]:
                graph[p.token_b][p.token_a] = (rate_b_to_a, p)
        return graph

    def find_arbitrage_cycles(
        self,
        start_token: str = "WETH",
        initial_amount: float = 10.0,
    ) -> List[TriangularArbitragePath]:
        """Bellman-Ford detection of negative-log product cycles."""
        graph = self.build_exchange_rate_graph()

        # Check all 3-hop cycles explicitly for robust execution
        cycles: List[TriangularArbitragePath] = []
        
        for t1 in self.tokens:
            if t1 != start_token:
                continue
            for t2, (r1, p1) in graph[t1].items():
                for t3, (r2, p2) in graph[t2].items():
                    if t3 == t1:
                        continue
                    if t1 in graph[t3]:
                        r3, p3 = graph[t3][t1]
                        product = r1 * r2 * r3
                        
                        # Simulate actual path execution with reserves impact
                        p1_sim = p1.clone()
                        p2_sim = p2.clone()
                        p3_sim = p3.clone()

                        out1 = p1_sim.get_amount_out(t1, initial_amount)
                        out2 = p2_sim.get_amount_out(t2, out1)
                        final_out = p3_sim.get_amount_out(t3, out2)
                        
                        net_profit = final_out - initial_amount
                        pct_profit = (net_profit / initial_amount) * 100.0

                        if product > 1.0001:
                            cycles.append(TriangularArbitragePath(
                                path_tokens=[t1, t2, t3, t1],
                                pools=[p1.name, p2.name, p3.name],
                                cycle_multiplier=product,
                                optimal_input=initial_amount,
                                expected_output=final_out,
                                net_profit=net_profit,
                                profit_pct=pct_profit,
                                is_profitable=(net_profit > 0.0),
                            ))

        cycles.sort(key=lambda x: x.cycle_multiplier, reverse=True)
        return cycles


class MEVSandwichSimulator:
    """Simulates atomic mempool sandwich attacks (Frontrun + Victim Execution + Backrun)."""

    def __init__(
        self,
        eth_price_usd: float = 3000.0,
        priority_fee_gwei: float = 50.0,
        builder_bribe_pct: float = 0.85,  # 85% bribe to block builder via Flashbots
    ):
        self.eth_price_usd = eth_price_usd
        self.priority_fee_gwei = priority_fee_gwei
        self.builder_bribe_pct = builder_bribe_pct

    def simulate_sandwich(
        self,
        pool: LiquidityPool,
        victim_token_in: str,
        victim_amount_in: float,
        victim_max_slippage_pct: float = 0.01,
        victim_tx_hash: str = "0xvictim_tx_sample",
    ) -> SandwichResult:
        """Simulates exact frontrunning, victim execution, and backrunning sequence."""
        target_token = pool.get_other_token(victim_token_in)
        p_initial = pool.get_spot_price(victim_token_in, target_token)

        # 1. Baseline: what would victim receive without sandwich?
        p_baseline = pool.clone()
        victim_out_baseline = p_baseline.get_amount_out(victim_token_in, victim_amount_in)

        # Worst acceptable output amount victim is willing to receive
        min_victim_out = victim_out_baseline * (1.0 - victim_max_slippage_pct)

        # 2. Optimal Searcher Frontrun Amount:
        # Searcher pushes price so victim receives exactly min_victim_out
        r_in = pool.get_reserve(victim_token_in)
        r_out = pool.get_reserve(target_token)
        gamma = pool.gamma

        def victim_out_given_frontrun(dx_front: float) -> float:
            dy_front = (r_out * gamma * dx_front) / (r_in + gamma * dx_front)
            r_in_after = r_in + dx_front
            r_out_after = r_out - dy_front
            v_out = (r_out_after * gamma * victim_amount_in) / (r_in_after + gamma * victim_amount_in)
            return v_out - min_victim_out

        # Search for maximal frontrunning amount
        try:
            opt_frontrun_in = brentq(victim_out_given_frontrun, 0.0, r_in * 0.45)
        except Exception:
            opt_frontrun_in = 0.0

        # Execute sandwich sequence
        p_active = pool.clone()

        # Step 1: Frontrun
        dy_frontrun = p_active.execute_swap(victim_token_in, opt_frontrun_in)
        p_post_frontrun = p_active.get_spot_price(victim_token_in, target_token)

        # Step 2: Victim Swap
        victim_out_actual = p_active.execute_swap(victim_token_in, victim_amount_in)
        p_post_victim = p_active.get_spot_price(victim_token_in, target_token)

        # Step 3: Backrun
        dx_backrun = p_active.execute_swap(target_token, dy_frontrun)
        p_final = p_active.get_spot_price(victim_token_in, target_token)

        # Accounting
        gross_profit = dx_backrun - opt_frontrun_in
        gas_cost_usd = (300_000 * self.priority_fee_gwei * 1e-9) * self.eth_price_usd
        
        # Convert profit to USD if necessary
        if victim_token_in in ["USDC", "USDT", "DAI"]:
            gross_profit_usd = gross_profit
        elif victim_token_in in ["WETH", "ETH"]:
            gross_profit_usd = gross_profit * self.eth_price_usd
        else:
            gross_profit_usd = gross_profit * p_initial

        bribe_usd = max(0.0, (gross_profit_usd - gas_cost_usd) * self.builder_bribe_pct)
        net_profit_usd = gross_profit_usd - gas_cost_usd - bribe_usd

        slippage_drag_pct = ((victim_out_baseline - victim_out_actual) / victim_out_baseline) * 100.0

        return SandwichResult(
            victim_tx_hash=victim_tx_hash,
            victim_input_amount=victim_amount_in,
            victim_token_in=victim_token_in,
            victim_token_out=target_token,
            frontrun_amount_in=opt_frontrun_in,
            frontrun_amount_out=dy_frontrun,
            victim_received_without_sandwich=victim_out_baseline,
            victim_received_with_sandwich=victim_out_actual,
            victim_slippage_drag_pct=slippage_drag_pct,
            backrun_amount_in=dy_frontrun,
            backrun_amount_out=dx_backrun,
            gross_mev_profit=gross_profit,
            gas_cost_usd=gas_cost_usd,
            builder_bribe_usd=bribe_usd,
            net_searcher_profit=net_profit_usd,
            pool_initial_price=p_initial,
            pool_post_frontrun_price=p_post_frontrun,
            pool_post_victim_price=p_post_victim,
            pool_final_price=p_final,
        )
