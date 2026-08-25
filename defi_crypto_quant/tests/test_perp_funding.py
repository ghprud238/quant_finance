"""Unit tests for Project 44: Crypto Perpetual Futures & Basis Trading."""

import unittest
import numpy as np
import pandas as pd

from defi_crypto_quant.perp_funding import (
    PerpetualFundingEngine,
    CashAndCarryBasisTrader,
)
from defi_crypto_quant.data.loader import generate_synthetic_funding_rates


class TestPerpetualFundingAndBasis(unittest.TestCase):
    """Validates funding rate clamping, statistical analytics, and cash-and-carry delta neutrality."""

    def setUp(self):
        self.engine = PerpetualFundingEngine()
        self.df_market = generate_synthetic_funding_rates(n_periods=300, base_rate_annual=0.15, seed=42)

    def test_funding_rate_formula_and_clamping(self):
        """Verifies 8-hour funding rate calculation and clamping within [-0.75%, +0.75%]."""
        # Normal positive premium
        r_normal = self.engine.compute_funding_rate(
            index_price=3000.0,
            impact_bid=3003.0,
            impact_ask=3004.0,
            interest_rate_8h=0.0001,
        )
        self.assertGreater(r_normal, 0.0)
        self.assertLess(r_normal, 0.0075)

        # Extreme bull premium exceeding upper clamp
        r_extreme = self.engine.compute_funding_rate(
            index_price=3000.0,
            impact_bid=3100.0,
            impact_ask=3105.0,
        )
        self.assertAlmostEqual(r_extreme, 0.0075)

        # Extreme discount exceeding lower clamp
        r_discount = self.engine.compute_funding_rate(
            index_price=3000.0,
            impact_bid=2800.0,
            impact_ask=2805.0,
        )
        self.assertAlmostEqual(r_discount, -0.0075)

    def test_funding_rate_series_statistics(self):
        """Verifies statistical metrics computation on historical funding rates."""
        stats = self.engine.analyze_funding_series(self.df_market["Funding_Rate_8h"])
        self.assertEqual(stats.n_periods, 300)
        self.assertGreater(stats.annualized_yield_apy, 0.0)
        self.assertGreater(stats.positive_funding_pct, 50.0)
        self.assertLessEqual(stats.max_rate_8h, 0.0075)
        self.assertGreaterEqual(stats.min_rate_8h, -0.0075)

        summary_df = stats.summary_dataframe()
        self.assertFalse(summary_df.empty)
        self.assertIn("Metric", summary_df.columns)
        self.assertIn("Value", summary_df.columns)

    def test_cash_and_carry_delta_neutrality(self):
        """Verifies that cash-and-carry strategy remains delta-neutral across large spot market moves."""
        # Create a market with a severe spot price crash (-50%)
        dates = pd.date_range("2025-01-01", periods=100, freq="8h")
        spot_crash = np.linspace(3000.0, 1500.0, 100)  # -50% crash
        perp_crash = spot_crash * 1.001
        constant_positive_funding = np.full(100, 0.0002)  # constant positive funding

        df_crash = pd.DataFrame({
            "Timestamp": dates,
            "Spot_Price": spot_crash,
            "Perp_Price": perp_crash,
            "Funding_Rate_8h": constant_positive_funding,
        })

        trader = CashAndCarryBasisTrader(
            initial_capital_usd=1_000_000.0,
            spot_allocation_pct=0.50,
            staking_yield_apy=0.0,  # disable staking to isolate pure basis pnl
        )
        res = trader.backtest(df_crash)

        # Despite a 50% spot crash, delta-neutral portfolio should have a positive return due to funding collected
        self.assertGreater(res.final_equity_usd, 1_000_000.0)
        self.assertGreater(res.total_return_pct, 0.0)
        self.assertGreater(res.total_funding_collected_usd, 0.0)
        self.assertGreater(res.sharpe_ratio, 0.0)

    def test_staking_yield_enhancement(self):
        """Verifies that spot staking yield enhances the overall cash-and-carry return."""
        trader_no_stake = CashAndCarryBasisTrader(
            initial_capital_usd=1_000_000.0,
            spot_allocation_pct=0.50,
            staking_yield_apy=0.0,
        )
        res_no_stake = trader_no_stake.backtest(self.df_market)

        trader_with_stake = CashAndCarryBasisTrader(
            initial_capital_usd=1_000_000.0,
            spot_allocation_pct=0.50,
            staking_yield_apy=0.04,  # +4% stETH staking yield
        )
        res_with_stake = trader_with_stake.backtest(self.df_market)

        self.assertGreater(res_with_stake.total_return_pct, res_no_stake.total_return_pct)
        self.assertGreater(res_with_stake.final_equity_usd, res_no_stake.final_equity_usd)

    def test_margin_call_triggering(self):
        """Verifies that high leverage and extreme price rallies trigger margin calls."""
        dates = pd.date_range("2025-01-01", periods=50, freq="8h")
        # Extreme vertical price pump (+150%)
        spot_pump = np.linspace(1000.0, 2500.0, 50)
        perp_pump = spot_pump * 1.002
        zero_funding = np.zeros(50)

        df_pump = pd.DataFrame({
            "Timestamp": dates,
            "Spot_Price": spot_pump,
            "Perp_Price": perp_pump,
            "Funding_Rate_8h": zero_funding,
        })

        # Aggressive 90% spot allocation (leaving only 10% margin cash buffer)
        aggressive_trader = CashAndCarryBasisTrader(
            initial_capital_usd=100_000.0,
            spot_allocation_pct=0.90,
            maintenance_margin_rate=0.05,
        )
        res = aggressive_trader.backtest(df_pump)
        self.assertGreater(res.margin_calls_count, 0)
        self.assertGreater(res.peak_margin_utilization_pct, 50.0)


if __name__ == "__main__":
    unittest.main()
