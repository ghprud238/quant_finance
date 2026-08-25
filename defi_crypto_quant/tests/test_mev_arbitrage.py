"""Unit tests for Project 43: Cross-DEX Flash Loans, Triangular Arbitrage & MEV Searcher."""

import unittest
import math
import numpy as np

from defi_crypto_quant.mev_arbitrage import (
    PoolType,
    LiquidityPool,
    CrossDEXArbitrageEngine,
    TriangularArbitrageSearcher,
    MEVSandwichSimulator,
)


class TestMEVArbitrage(unittest.TestCase):
    """Validates AMM pricing, optimal spatial arbitrage sizing, Bellman-Ford cycles, and sandwich attacks."""

    def setUp(self):
        self.pool_uni = LiquidityPool(
            name="Uniswap_ETH_USDC",
            pool_type=PoolType.UNISWAP_V2,
            token_a="WETH",
            token_b="USDC",
            reserve_a=2500.0,
            reserve_b=7_500_000.0,  # $3000 / WETH
            fee=0.003,
        )
        self.pool_sushi = LiquidityPool(
            name="Sushiswap_ETH_USDC",
            pool_type=PoolType.SUSHISWAP,
            token_a="WETH",
            token_b="USDC",
            reserve_a=1800.0,
            reserve_b=5_580_000.0,  # $3100 / WETH (Mispriced higher)
            fee=0.003,
        )

    def test_constant_product_output(self):
        """Verifies AMM output formula Delta_y = (R_out * gamma * dx) / (R_in + gamma * dx)."""
        dx = 10.0  # 10 WETH
        expected_out = (7_500_000.0 * 0.997 * 10.0) / (2500.0 + 0.997 * 10.0)
        actual_out = self.pool_uni.get_amount_out("WETH", dx)
        self.assertAlmostEqual(actual_out, expected_out, places=4)

    def test_amount_in_inversion(self):
        """Verifies get_amount_in is the exact inverse of get_amount_out."""
        desired_out = 50_000.0  # 50,000 USDC
        required_in = self.pool_uni.get_amount_in("USDC", desired_out)
        recovered_out = self.pool_uni.get_amount_out("WETH", required_in)
        self.assertAlmostEqual(recovered_out, desired_out, places=4)

    def test_pool_spot_price_and_swap_execution(self):
        """Verifies spot price calculation and state mutation upon swap execution."""
        p = self.pool_uni.clone()
        initial_price = p.get_spot_price("WETH", "USDC")
        self.assertAlmostEqual(initial_price, 3000.0, places=4)

        # Execute 50 WETH market sell for USDC
        out_usdc = p.execute_swap("WETH", 50.0)
        self.assertGreater(out_usdc, 0.0)
        self.assertEqual(p.reserve_a, 2550.0)
        self.assertEqual(p.reserve_b, 7_500_000.0 - out_usdc)
        
        # Price of WETH should have dropped after large sell
        new_price = p.get_spot_price("WETH", "USDC")
        self.assertLess(new_price, initial_price)

    def test_closed_form_spatial_arbitrage_optimality(self):
        """Verifies closed-form optimal trade size matches exact first-order optimality."""
        engine = CrossDEXArbitrageEngine()
        opt_dx = engine.compute_closed_form_optimal_input(
            pool1=self.pool_uni,
            pool2=self.pool_sushi,
            token_borrow="USDC",
        )
        self.assertGreater(opt_dx, 0.0)

        # Perturb slightly +/- to verify local maximum of profit function
        def calc_profit(dx):
            p1 = self.pool_uni.clone()
            p2 = self.pool_sushi.clone()
            dy = p1.get_amount_out("USDC", dx)
            return p2.get_amount_out("WETH", dy) - dx

        base_profit = calc_profit(opt_dx)
        profit_plus = calc_profit(opt_dx * 1.05)
        profit_minus = calc_profit(opt_dx * 0.95)

        self.assertGreater(base_profit, profit_plus)
        self.assertGreater(base_profit, profit_minus)

    def test_spatial_arbitrage_execution_and_profit(self):
        """Tests full spatial arbitrage evaluation with gas and flash loan fees."""
        engine = CrossDEXArbitrageEngine(default_flash_loan_fee=0.0009, eth_price_usd=3000.0)
        res = engine.evaluate_spatial_arbitrage(
            pool1=self.pool_uni,
            pool2=self.pool_sushi,
            token_borrow="USDC",
            flash_loan_fee_pct=0.0005,  # 5 bps Balancer flash loan
        )
        self.assertTrue(res.is_profitable)
        self.assertGreater(res.net_profit, 0.0)
        self.assertGreater(res.optimal_input, 1000.0)
        self.assertGreater(res.return_on_capital_pct, 0.0)

    def test_unprofitable_spatial_arbitrage(self):
        """Verifies that identical price pools return non-profitable zero arbitrage."""
        engine = CrossDEXArbitrageEngine()
        p_identical = self.pool_uni.clone()
        p_identical.name = "Uniswap_Copy"
        res = engine.evaluate_spatial_arbitrage(
            pool1=self.pool_uni,
            pool2=p_identical,
            token_borrow="USDC",
        )
        self.assertFalse(res.is_profitable)
        self.assertEqual(res.optimal_input, 0.0)

    def test_triangular_arbitrage_cycle_detection(self):
        """Tests Bellman-Ford negative-log cycle detection on a triangular pool network."""
        p_eth_usdc = LiquidityPool("U_ETH_USDC", PoolType.UNISWAP_V2, "WETH", "USDC", 1000.0, 3_000_000.0, 0.003)
        p_btc_usdc = LiquidityPool("U_BTC_USDC", PoolType.UNISWAP_V2, "WBTC", "USDC", 50.0, 3_000_000.0, 0.003)
        # Mispriced WBTC/WETH pool: 1 WBTC = 22 WETH instead of 20 WETH
        p_btc_eth = LiquidityPool("U_BTC_ETH", PoolType.UNISWAP_V2, "WBTC", "WETH", 50.0, 1100.0, 0.003)

        searcher = TriangularArbitrageSearcher(pools=[p_eth_usdc, p_btc_usdc, p_btc_eth])
        cycles = searcher.find_arbitrage_cycles(start_token="WETH", initial_amount=5.0)
        
        self.assertGreater(len(cycles), 0)
        best_cycle = cycles[0]
        self.assertTrue(best_cycle.is_profitable)
        self.assertGreater(best_cycle.cycle_multiplier, 1.0)
        self.assertEqual(best_cycle.path_tokens[0], "WETH")
        self.assertEqual(best_cycle.path_tokens[-1], "WETH")

    def test_mev_sandwich_attack_mechanics(self):
        """Tests frontrunning + backrunning sandwich attack on a large victim market swap."""
        simulator = MEVSandwichSimulator(eth_price_usd=3000.0, priority_fee_gwei=40.0, builder_bribe_pct=0.80)
        
        pool = self.pool_uni.clone()
        res = simulator.simulate_sandwich(
            pool=pool,
            victim_token_in="USDC",
            victim_amount_in=300_000.0,  # $300k victim swap
            victim_max_slippage_pct=0.015,  # 1.5% slippage tolerance
        )

        self.assertGreater(res.frontrun_amount_in, 0.0)
        self.assertGreater(res.gross_mev_profit, 0.0)
        self.assertGreater(res.net_searcher_profit, 0.0)
        self.assertLess(res.victim_received_with_sandwich, res.victim_received_without_sandwich)
        self.assertGreater(res.victim_slippage_drag_pct, 0.0)
        self.assertAlmostEqual(res.victim_slippage_drag_pct, 1.5, places=1)


if __name__ == "__main__":
    unittest.main()
