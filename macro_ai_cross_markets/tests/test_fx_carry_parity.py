"""Unit tests for Module 49: Cross-Economy FX Carry Trade, Interest Rate Parity & Volatility Surface."""

import unittest
import numpy as np
import pandas as pd

from macro_ai_cross_markets.fx_carry_parity.carry_engine import (
    FXCarryParityEngine,
    ParityResult,
    FamaRegressionResult,
    MalzVolSurfaceResult,
    FXCarryStrategyResult,
)


class TestFXCarryParityEngine(unittest.TestCase):
    """Validates CIP/UIP parity calculations, Fama regression, Malz vol surface, and FX carry backtest."""

    def setUp(self):
        self.engine = FXCarryParityEngine(default_transaction_cost_bps=2.0)

    def test_cip_and_uip_parity_math(self):
        spot = 1.1000  # EUR/USD
        r_usd = 0.05   # Domestic (USD) 5%
        r_eur = 0.03   # Foreign (EUR) 3%
        tau = 1.0

        # CIP Forward: F = 1.1000 * (1 + 0.05) / (1 + 0.03) = 1.1000 * 1.05 / 1.03 = 1.121359
        res = self.engine.calculate_interest_rate_parity(
            spot_rate=spot,
            r_domestic=r_usd,
            r_foreign=r_eur,
            tenor_years=tau,
        )

        expected_fwd = 1.1000 * (1.05 / 1.03)
        self.assertAlmostEqual(res.cip_theoretical_forward, expected_fwd, places=5)
        self.assertAlmostEqual(res.cip_basis_bps, 0.0, places=4)
        self.assertAlmostEqual(res.carry_yield_spread_pct, -2.0, places=4)
        self.assertFalse(res.is_cip_arbitrage_profitable)

        # Injected market dislocation
        res_disloc = self.engine.calculate_interest_rate_parity(
            spot_rate=spot,
            r_domestic=r_usd,
            r_foreign=r_eur,
            tenor_years=tau,
            forward_market=1.1350,  # +136 bps basis
        )
        self.assertTrue(res_disloc.is_cip_arbitrage_profitable)
        self.assertGreater(res_disloc.cip_basis_bps, 50.0)

    def test_fama_forward_bias_regression(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        np.random.seed(42)

        # Synthetic spot & forward demonstrating Fama forward rate puzzle (beta < 0)
        spot_levels = 1.10 * np.cumprod(1.0 + np.random.normal(0, 0.008, 100))
        spots = pd.Series(spot_levels, index=dates)

        # Forward prices with interest rate differential
        rate_diff = 0.03 / 52.0
        fwds = spots * (1.0 + rate_diff + np.random.normal(0, 0.002, 100))

        reg_res = self.engine.fama_forward_rate_bias_regression(spots, fwds, horizon_steps=1)
        self.assertIsInstance(reg_res, FamaRegressionResult)
        self.assertIsInstance(reg_res.beta, float)
        self.assertIsInstance(reg_res.r_squared, float)
        self.assertTrue(reg_res.is_forward_bias_present)
        self.assertFalse(reg_res.summary_dataframe.empty)

    def test_malz_vol_surface_interpolation(self):
        atm_vol = 0.10             # 10.0% ATM vol
        risk_reversal_25 = 0.015   # +1.5% RR (Call vol > Put vol)
        butterfly_25 = 0.005       # +0.5% BF (Fat tails)

        res = self.engine.fit_malz_vol_surface(
            atm_vol=atm_vol,
            risk_reversal_25=risk_reversal_25,
            butterfly_25=butterfly_25,
        )

        self.assertIsInstance(res, MalzVolSurfaceResult)
        # Check ATM 50D volatility matches exactly
        self.assertAlmostEqual(res.vol_by_delta[0.50], 0.10, places=5)

        # Call 25D should equal ATM + BF + 0.5 * RR = 0.10 + 0.005 + 0.0075 = 0.1125
        # In Malz: sigma(0.25) -> Delta=0.25 (Put) vs Delta=0.75 (Call)
        # Check that 25D call > 25D put when RR > 0
        self.assertGreater(res.call_25d_vol, res.put_25d_vol)
        self.assertFalse(res.summary_dataframe.empty)

    def test_fx_carry_strategy_backtest(self):
        dates = pd.date_range("2021-01-01", periods=250, freq="B")
        np.random.seed(42)

        # Funding currencies (Low yield: USD 2%, JPY 0.1%, EUR 1.5%, CHF 0.5%)
        # Target currencies (High yield: BRL 12%, MXN 10%, ZAR 8%, INR 7%)
        fx_spots = pd.DataFrame({
            "USD": np.ones(250),
            "EUR": 1.15 * np.cumprod(1.0 + np.random.normal(0, 0.004, 250)),
            "JPY": 110.0 * np.cumprod(1.0 + np.random.normal(0, 0.005, 250)),
            "CHF": 0.92 * np.cumprod(1.0 + np.random.normal(0, 0.004, 250)),
            "BRL": 5.20 * np.cumprod(1.0 + np.random.normal(0.0002, 0.007, 250)),
            "MXN": 20.0 * np.cumprod(1.0 + np.random.normal(0.0001, 0.006, 250)),
            "ZAR": 15.5 * np.cumprod(1.0 + np.random.normal(0.0001, 0.008, 250)),
            "INR": 75.0 * np.cumprod(1.0 + np.random.normal(0.0001, 0.003, 250)),
        }, index=dates)

        fx_rates = pd.DataFrame({
            "USD": np.full(250, 0.02),
            "EUR": np.full(250, 0.015),
            "JPY": np.full(250, 0.001),
            "CHF": np.full(250, 0.005),
            "BRL": np.full(250, 0.12),
            "MXN": np.full(250, 0.10),
            "ZAR": np.full(250, 0.08),
            "INR": np.full(250, 0.07),
        }, index=dates)

        res = self.engine.backtest_fx_carry_strategy(
            fx_spot_df=fx_spots,
            interest_rates_df=fx_rates,
            funding_currencies=["USD", "EUR", "JPY", "CHF"],
            target_currencies=["BRL", "MXN", "ZAR", "INR"],
            vol_filter=True,
            target_annual_vol=0.10,
        )

        self.assertIsInstance(res, FXCarryStrategyResult)
        self.assertEqual(len(res.cumulative_equity), 250)
        self.assertIsInstance(res.sharpe_ratio, float)
        self.assertIsInstance(res.cagr, float)
        self.assertIsInstance(res.skewness, float)
        self.assertFalse(res.metrics_table.empty)


if __name__ == "__main__":
    unittest.main()
