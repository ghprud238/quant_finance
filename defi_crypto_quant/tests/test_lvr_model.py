"""Unit tests for Project 42: Impermanent Loss & Loss-Versus-Rebalancing (LVR) Models."""

import unittest
import numpy as np
import pandas as pd

from defi_crypto_quant.lvr_impermanent_loss import (
    ImpermanentLossCalculator,
    LossVersusRebalancingEngine,
    LVRSimulationResult,
)


class TestImpermanentLossCalculator(unittest.TestCase):
    """Validates standard and concentrated Impermanent Loss mathematical bounds."""

    def test_v2_il_at_par(self):
        # k = 1.0 -> IL = 0.0
        il_par = ImpermanentLossCalculator.calculate_v2_impermanent_loss(1.0)
        self.assertAlmostEqual(il_par, 0.0, places=6)

    def test_v2_il_price_doubling(self):
        # k = 2.0 -> IL = 2 * sqrt(2) / 3 - 1 = 2 * 1.41421356 / 3 - 1 = -0.05719 (-5.72%)
        il_2x = ImpermanentLossCalculator.calculate_v2_impermanent_loss(2.0)
        expected_il = (2.0 * np.sqrt(2.0)) / 3.0 - 1.0
        self.assertAlmostEqual(il_2x, expected_il, places=5)
        self.assertAlmostEqual(il_2x, -0.05719, places=4)

    def test_v2_il_price_halving(self):
        # k = 0.5 -> IL = 2 * sqrt(0.5) / 1.5 - 1 = -0.05719 (-5.72%)
        il_half = ImpermanentLossCalculator.calculate_v2_impermanent_loss(0.5)
        expected_il = (2.0 * np.sqrt(0.5)) / 1.5 - 1.0
        self.assertAlmostEqual(il_half, expected_il, places=5)

    def test_v3_concentrated_il_amplification(self):
        # Range [2500, 3500] around P0 = 3000
        p0 = 3000.0
        pa = 2500.0
        pb = 3500.0
        
        # When price moves to 3300 (10% up), concentrated IL is larger in magnitude than v2 IL
        il_v3 = ImpermanentLossCalculator.calculate_v3_impermanent_loss(3300.0, p0, pa, pb)
        il_v2 = ImpermanentLossCalculator.calculate_v2_impermanent_loss(3300.0 / p0)
        
        self.assertLess(il_v3, il_v2)  # More negative loss
        self.assertLess(il_v3, 0.0)


class TestLossVersusRebalancingEngine(unittest.TestCase):
    """Validates LVR continuous integral, simulation mechanics, and LP profitability."""

    def setUp(self):
        self.engine_v2 = LossVersusRebalancingEngine(pool_type="v2", fee_rate=0.0030)
        self.engine_v3 = LossVersusRebalancingEngine(pool_type="v3", fee_rate=0.0030)
        
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        self.prices = pd.Series(3000.0 + np.cumsum(np.random.normal(0, 15, 100)), index=dates)
        self.volumes = pd.Series(np.random.lognormal(12, 0.5, 100), index=dates)

    def test_continuous_lvr_formula(self):
        prices = np.array([3000.0, 3050.0, 3100.0])
        liquidity = np.array([1000.0, 1000.0, 1000.0])
        vol_ann = 0.60
        dt = 1.0 / (365 * 24)
        
        cum_lvr = LossVersusRebalancingEngine.continuous_lvr_integral(prices, liquidity, vol_ann, dt)
        self.assertEqual(len(cum_lvr), 3)
        self.assertTrue(np.all(np.diff(cum_lvr) >= 0))  # Monotonically increasing

    def test_lp_simulation_v2(self):
        res = self.engine_v2.simulate_lp_performance(
            price_series=self.prices,
            volume_series=self.volumes,
            initial_capital_usd=100_000.0,
            volatility_ann=0.65,
        )
        self.assertIsInstance(res, LVRSimulationResult)
        self.assertGreater(res.total_lvr_usd, 0.0)
        self.assertGreater(res.total_fee_revenue_usd, 0.0)
        self.assertGreater(res.breakeven_volatility_ann, 0.0)
        self.assertFalse(res.summary_table.empty)

    def test_lp_simulation_v3(self):
        res = self.engine_v3.simulate_lp_performance(
            price_series=self.prices,
            volume_series=self.volumes,
            initial_capital_usd=100_000.0,
            price_lower=2500.0,
            price_upper=3500.0,
            volatility_ann=0.65,
        )
        self.assertIsInstance(res, LVRSimulationResult)
        self.assertGreater(res.total_lvr_usd, 0.0)
        self.assertGreater(res.total_fee_revenue_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
