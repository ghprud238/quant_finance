"""Unit tests for Module 30: Full Production Systematic Trading System."""

import unittest
import numpy as np
import pandas as pd

from interview_quant.data.loader import generate_market_data
from interview_quant.systematic_system.trading_system import (
    ProductionTradingSystem,
    ProductionSystemResult,
    AlmgrenChrissSchedule,
    StressGatingResult,
)


class TestProductionTradingSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_market_data(
            tickers=["SPY", "QQQ", "AAPL", "MSFT", "TLT"],
            start_date="2018-01-01",
            end_date="2024-12-31",
            seed=42,
        )
        cls.system = ProductionTradingSystem(
            target_annual_vol=0.10,
            vol_lookback_days=60,
            max_asset_weight=0.25,
            max_gross_leverage=2.0,
            drawdown_warning_level=0.08,
            drawdown_circuit_breaker=0.15,
        )

    def test_alpha_signals_generation(self):
        signals = self.system.compute_alpha_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(signals.shape[1], 5)
        # Signals bounded in [-1.0, 1.0]
        self.assertTrue((signals >= -1.0 - 1e-6).all().all())
        self.assertTrue((signals <= 1.0 + 1e-6).all().all())
        self.assertFalse(signals.isna().any().any())

    def test_risk_management_and_circuit_breaker(self):
        close = self.data.xs("Close", level="Field", axis=1)
        signals = self.system.compute_alpha_signals(self.data)

        weights, circuit_active, leverage = self.system.size_positions_with_risk_control(close, signals)

        # Asset bounds
        self.assertTrue((weights <= self.system.max_asset_weight + 1e-6).all().all())
        self.assertTrue((weights >= -self.system.max_asset_weight - 1e-6).all().all())

        # Leverage bounds
        self.assertTrue((leverage <= self.system.max_gross_leverage + 1e-6).all())
        self.assertIsInstance(circuit_active, pd.Series)

    def test_almgren_chriss_optimal_execution(self):
        schedule = self.system.compute_almgren_chriss_schedule(
            total_shares=100_000.0,
            time_horizon_hours=6.5,
            n_intervals=13,
            daily_vol=0.015,
        )
        self.assertIsInstance(schedule, AlmgrenChrissSchedule)
        self.assertEqual(len(schedule.optimal_trade_list), 13)
        self.assertEqual(len(schedule.inventory_trajectory), 14)

        # Total shares executed must equal 100,000
        self.assertAlmostEqual(np.sum(schedule.optimal_trade_list), 100_000.0, places=3)
        self.assertAlmostEqual(schedule.inventory_trajectory[0], 100_000.0, places=3)
        self.assertAlmostEqual(schedule.inventory_trajectory[-1], 0.0, places=3)
        self.assertGreater(schedule.expected_cost_bps, 0.0)

    def test_stress_gating_pre_trade_check(self):
        w_safe = pd.Series([0.05, 0.05, 0.05, 0.05, -0.05])
        gate_safe = self.system.evaluate_stress_gating(w_safe, max_allowed_loss_pct=0.15)
        self.assertIsInstance(gate_safe, StressGatingResult)
        self.assertTrue(gate_safe.passed)

        w_risky = pd.Series([0.50, 0.50, 0.50, 0.50, 0.50])
        gate_risky = self.system.evaluate_stress_gating(w_risky, max_allowed_loss_pct=0.10)
        self.assertFalse(gate_risky.passed)

    def test_end_to_end_system_run_oos(self):
        result = self.system.run_systematic_system(self.data, oos_start_date="2020-01-01")
        self.assertIsInstance(result, ProductionSystemResult)

        # Check Out-of-Sample metrics
        self.assertIn("CAGR", result.metrics)
        self.assertIn("Sharpe Ratio", result.metrics)
        self.assertIn("Max Drawdown", result.metrics)
        self.assertIn("Win Rate", result.metrics)

        self.assertGreater(result.cumulative_equity.iloc[-1], result.cumulative_equity.iloc[0])
        self.assertGreater(result.metrics["Sharpe Ratio"], 0.0)
        self.assertLessEqual(result.metrics["Max Drawdown"], 0.0)
        self.assertGreater(result.metrics["Win Rate"], 0.40)


if __name__ == "__main__":
    unittest.main()
