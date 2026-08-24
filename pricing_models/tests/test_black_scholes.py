import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
"""Unit tests for Black-Scholes Option Pricing Engine (Module 16)."""

import unittest
import numpy as np
import pandas as pd

from pricing_models.black_scholes.engine import BlackScholesModel, OptionChainPricer, OptionPriceResult


class TestBlackScholesModel(unittest.TestCase):
    """Tests pricing accuracy, Put-Call parity, and boundary conditions."""

    def setUp(self):
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.20
        self.q = 0.0

    def test_standard_textbook_pricing(self):
        """Tests benchmark Black-Scholes price against known closed-form values."""
        call = BlackScholesModel.call_price(self.S, self.K, self.T, self.r, self.sigma, self.q)
        put = BlackScholesModel.put_price(self.S, self.K, self.T, self.r, self.sigma, self.q)

        # Known benchmark values: C ≈ 10.45058, P ≈ 5.57353
        self.assertAlmostEqual(call, 10.45058, places=4)
        self.assertAlmostEqual(put, 5.57353, places=4)

    def test_dividend_yield_pricing(self):
        """Tests Merton (1973) continuous dividend yield extension."""
        q = 0.02
        call = BlackScholesModel.call_price(self.S, self.K, self.T, self.r, self.sigma, q)
        put = BlackScholesModel.put_price(self.S, self.K, self.T, self.r, self.sigma, q)

        # Dividend reduces call price and increases put price
        call_no_div = BlackScholesModel.call_price(self.S, self.K, self.T, self.r, self.sigma, 0.0)
        put_no_div = BlackScholesModel.put_price(self.S, self.K, self.T, self.r, self.sigma, 0.0)

        self.assertLess(call, call_no_div)
        self.assertGreater(put, put_no_div)

    def test_put_call_parity(self):
        """Tests Put-Call Parity across multiple moneyness, rates, and dividend yields."""
        spots = [80.0, 95.0, 100.0, 105.0, 120.0]
        strikes = [90.0, 100.0, 110.0]
        expiries = [0.1, 0.5, 1.0, 2.0]
        rates = [0.0, 0.03, 0.08]
        vols = [0.10, 0.25, 0.50]
        dividends = [0.0, 0.02, 0.05]

        for s in spots:
            for k in strikes:
                for t in expiries:
                    for r in rates:
                        for sig in vols:
                            for q in dividends:
                                res = BlackScholesModel.verify_put_call_parity(s, k, t, r, sig, q)
                                self.assertTrue(res["is_parity_valid"], f"Parity failed for S={s}, K={k}, T={t}")
                                self.assertLess(res["abs_error"], 1e-7)

    def test_intrinsic_and_time_value(self):
        """Tests intrinsic and time value breakdown."""
        # In-the-money call: S=110, K=100
        call = BlackScholesModel.call_price(110.0, 100.0, 1.0, 0.05, 0.20, 0.0)
        iv_call = BlackScholesModel.intrinsic_value(110.0, 100.0, "call")
        tv_call = BlackScholesModel.time_value(call, 110.0, 100.0, "call")

        self.assertEqual(iv_call, 10.0)
        self.assertGreater(tv_call, 0.0)
        self.assertAlmostEqual(call, iv_call + tv_call, places=6)

        # Out-of-the-money call: S=90, K=100
        call_otm = BlackScholesModel.call_price(90.0, 100.0, 1.0, 0.05, 0.20, 0.0)
        iv_otm = BlackScholesModel.intrinsic_value(90.0, 100.0, "call")
        tv_otm = BlackScholesModel.time_value(call_otm, 90.0, 100.0, "call")

        self.assertEqual(iv_otm, 0.0)
        self.assertEqual(call_otm, tv_otm)

    def test_boundary_conditions(self):
        """Tests edge cases: T=0, sigma=0, deep ITM/OTM."""
        # Expiry T = 0
        call_exp = BlackScholesModel.call_price(105.0, 100.0, 0.0, 0.05, 0.20)
        put_exp = BlackScholesModel.put_price(95.0, 100.0, 0.0, 0.05, 0.20)
        self.assertEqual(call_exp, 5.0)
        self.assertEqual(put_exp, 5.0)

        # Deep ITM / OTM
        call_deep_itm = BlackScholesModel.call_price(500.0, 100.0, 1.0, 0.05, 0.20)
        self.assertAlmostEqual(call_deep_itm, 500.0 - 100.0 * np.exp(-0.05), delta=0.5)

        call_deep_otm = BlackScholesModel.call_price(10.0, 100.0, 1.0, 0.05, 0.20)
        self.assertAlmostEqual(call_deep_otm, 0.0, places=4)

    def test_vectorization(self):
        """Tests vectorized array pricing across strikes."""
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        calls = BlackScholesModel.call_price(self.S, strikes, self.T, self.r, self.sigma, self.q)
        puts = BlackScholesModel.put_price(self.S, strikes, self.T, self.r, self.sigma, self.q)

        self.assertEqual(len(calls), 5)
        self.assertEqual(len(puts), 5)
        # Monotonicity: Call prices decrease with strike, Put prices increase
        self.assertTrue(np.all(np.diff(calls) < 0))
        self.assertTrue(np.all(np.diff(puts) > 0))

    def test_option_price_result(self):
        """Tests full calculate() container."""
        res = BlackScholesModel.calculate(self.S, self.K, self.T, self.r, self.sigma, self.q, "call")
        self.assertIsInstance(res, OptionPriceResult)
        self.assertAlmostEqual(res.price, 10.45058, places=4)
        self.assertAlmostEqual(res.call_price, 10.45058, places=4)
        self.assertAlmostEqual(res.put_price, 5.57353, places=4)

    def test_option_chain_pricer(self):
        """Tests OptionChainPricer table generation."""
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        chain = OptionChainPricer.generate_chain(self.S, strikes, self.T, self.r, self.sigma, self.q)

        self.assertIsInstance(chain, pd.DataFrame)
        self.assertEqual(len(chain), 5)
        expected_cols = ["Call_Delta", "Call_Gamma", "Call_Theta", "Call_Vega", "Call_Price", "Strike", "Moneyness", "Put_Price", "Put_Delta", "Put_Gamma", "Put_Theta", "Put_Vega", "IV"]
        for col in expected_cols:
            self.assertIn(col, chain.columns)


if __name__ == "__main__":
    unittest.main()
