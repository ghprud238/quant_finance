"""Unit tests for Project 41: Constant Function Market Makers & Concentrated Liquidity AMMs."""

import unittest
import numpy as np

from defi_crypto_quant.uniswap_amm import (
    ConstantProductAMM,
    ConcentratedLiquidityAMM,
    StableswapAMM,
    SwapResult,
)


class TestConstantProductAMM(unittest.TestCase):
    """Validates Uniswap v2 constant product (x * y = k) AMM math."""

    def setUp(self):
        # 1000 ETH and 3,000,000 USDC -> P = $3,000 / ETH
        self.amm = ConstantProductAMM(
            reserve_x=1000.0,
            reserve_y=3_000_000.0,
            token_x_symbol="ETH",
            token_y_symbol="USDC",
            fee_rate=0.0030,
        )

    def test_spot_price(self):
        self.assertAlmostEqual(self.amm.spot_price_y_per_x, 3000.0, places=4)
        self.assertAlmostEqual(self.amm.spot_price_x_per_y, 1.0 / 3000.0, places=6)

    def test_exact_output_formula(self):
        # Selling 10 ETH into pool
        # Delta_y = (y * gamma * dx) / (x + gamma * dx)
        # gamma = 0.997
        # Delta_y = (3,000,000 * 9.97) / (1000 + 9.97) = 29,910,000 / 1009.97 = 29,614.74 USDC
        amount_out, fee_paid = self.amm.get_amount_out(10.0, "ETH")
        expected_dy = (3_000_000.0 * 9.97) / (1000.0 + 9.97)
        self.assertAlmostEqual(amount_out, expected_dy, places=2)
        self.assertAlmostEqual(fee_paid, 0.03, places=4)

    def test_swap_execution_and_invariant_growth(self):
        k_before = self.amm.k
        res = self.amm.swap(amount_in=10.0, token_in="ETH")
        
        self.assertEqual(res.token_in, "ETH")
        self.assertEqual(res.token_out, "USDC")
        self.assertGreater(res.amount_out, 29000.0)
        self.assertGreater(self.amm.spot_price_y_per_x, 0.0)
        
        # Invariant k grows due to fees retained in pool
        self.assertGreaterEqual(self.amm.k, k_before)

    def test_add_and_remove_liquidity(self):
        shares_minted, ax, ay = self.amm.add_liquidity(100.0, 300_000.0)
        self.assertGreater(shares_minted, 0.0)
        self.assertAlmostEqual(self.amm.spot_price_y_per_x, 3000.0, places=2)
        
        rx, ry = self.amm.remove_liquidity(shares_minted)
        self.assertAlmostEqual(rx, 100.0, places=2)
        self.assertAlmostEqual(ry, 300_000.0, places=2)


class TestConcentratedLiquidityAMM(unittest.TestCase):
    """Validates Uniswap v3 concentrated liquidity, tick math, and virtual reserves."""

    def setUp(self):
        self.amm = ConcentratedLiquidityAMM(
            current_price=3000.0,
            fee_tier=0.0030,
            token_x_symbol="ETH",
            token_y_symbol="USDC",
        )

    def test_tick_conversions(self):
        price = 3000.0
        tick = self.amm.price_to_tick(price)
        recovered_price = self.amm.tick_to_price(tick)
        self.assertAlmostEqual(recovered_price, price, delta=price * 0.0001)

    def test_capital_efficiency(self):
        # Range [2500, 3500] vs full range [0, inf]
        p_a = 2500.0
        p_b = 3500.0
        eff = self.amm.capital_efficiency_multiplier(p_a, p_b)
        # 1 / (1 - sqrt(2500/3500)) = 1 / (1 - 0.84515) = 1 / 0.15485 = 6.45x
        self.assertGreater(eff, 5.0)
        self.assertLess(eff, 10.0)

    def test_mint_position_and_amounts(self):
        p_a = 2700.0
        p_b = 3300.0
        pos = self.amm.mint_position(
            owner="Alice",
            price_lower=p_a,
            price_upper=p_b,
            amount_x=10.0,
            amount_y=30_000.0,
        )
        self.assertGreater(pos.liquidity, 0.0)
        self.assertGreater(self.amm.liquidity, 0.0)
        self.assertGreater(pos.amount_x, 0.0)
        self.assertGreater(pos.amount_y, 0.0)

    def test_v3_swap_within_and_across_ticks(self):
        self.amm.mint_position("Alice", 2500.0, 3500.0, 50.0, 150_000.0)
        p_before = self.amm.current_price
        
        # Sell 2 ETH into pool -> price should drop
        res = self.amm.swap(amount_in=2.0, token_in="ETH")
        self.assertEqual(res.token_out, "USDC")
        self.assertGreater(res.amount_out, 5000.0)
        self.assertLess(self.amm.current_price, p_before)


class TestStableswapAMM(unittest.TestCase):
    """Validates Curve Stableswap invariant."""

    def setUp(self):
        # 3pool: 1,000,000 DAI, 1,000,000 USDC, 1,000,000 USDT
        self.curve = StableswapAMM(
            reserves=[1_000_000.0, 1_000_000.0, 1_000_000.0],
            token_symbols=["DAI", "USDC", "USDT"],
            A=200.0,
            fee_rate=0.0004,
        )

    def test_invariant_D(self):
        self.assertAlmostEqual(self.curve.D, 3_000_000.0, delta=1.0)

    def test_stableswap_low_slippage_swap(self):
        # Swap 10,000 DAI for USDC
        dy_net = self.curve.swap(0, 1, 10_000.0)
        # Should be extremely close to 10,000 minus 4 bps fee (9996.0)
        self.assertGreater(dy_net, 9990.0)
        self.assertLess(dy_net, 10000.0)


if __name__ == "__main__":
    unittest.main()
