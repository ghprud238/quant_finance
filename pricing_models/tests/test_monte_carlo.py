import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
"""Unit tests for Monte Carlo Option Pricing Engine & Exotics."""

import unittest
import numpy as np
from pricing_models.data import BlackScholesAnalytical
from pricing_models.monte_carlo import MonteCarloOptionPricer, ExoticOptionPricer


class TestMonteCarloOptionPricer(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.20
        self.q = 0.01

    def test_european_call_confidence_interval(self):
        bs_call = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, 'call')
        pricer = MonteCarloOptionPricer(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        res = pricer.price('call', n_simulations=100000, antithetic=True, random_state=42)

        # 95% Confidence interval must cover true analytical BS price
        ci_low, ci_high = res.confidence_interval_95
        self.assertTrue(ci_low <= bs_call <= ci_high, f"{ci_low} <= {bs_call} <= {ci_high}")
        self.assertAlmostEqual(res.price, bs_call, delta=0.15)

    def test_antithetic_variance_reduction(self):
        pricer = MonteCarloOptionPricer(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        res_raw = pricer.price('call', n_simulations=50000, antithetic=False, random_state=42)
        res_anti = pricer.price('call', n_simulations=50000, antithetic=True, random_state=42)

        # Antithetic variates should yield lower or equal standard error
        self.assertLessEqual(res_anti.standard_error, res_raw.standard_error * 1.05)

    def test_control_variates_variance_reduction(self):
        pricer = MonteCarloOptionPricer(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        res_raw = pricer.price('call', n_simulations=50000, antithetic=False, control_variate=False, random_state=42)
        res_cv = pricer.price('call', n_simulations=50000, antithetic=False, control_variate=True, random_state=42)

        # Control variates should significantly reduce standard error
        self.assertLess(res_cv.standard_error, res_raw.standard_error)
        self.assertGreater(res_cv.variance_reduction_ratio, 1.20)

    def test_monte_carlo_greeks(self):
        pricer = MonteCarloOptionPricer(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        mc_greeks = pricer.greeks('call', n_simulations=100000, random_state=42)
        bs_greeks = BlackScholesAnalytical.greeks(self.S0, self.K, self.T, self.r, self.sigma, self.q, 'call')

        self.assertAlmostEqual(mc_greeks.delta, bs_greeks.delta, delta=0.03)
        self.assertAlmostEqual(mc_greeks.vega, bs_greeks.vega, delta=2.5)

    def test_asian_option_properties(self):
        exotic_pricer = ExoticOptionPricer(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        res_asian = exotic_pricer.price_asian('call', 'arithmetic', n_simulations=50000, n_steps=100)
        bs_vanilla = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, 'call')

        # Asian average price call is cheaper than vanilla European call due to volatility dampening
        self.assertLess(res_asian.price, bs_vanilla)
        self.assertGreater(res_asian.price, 0.0)

    def test_barrier_option_properties(self):
        exotic_pricer = ExoticOptionPricer(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        res_uo = exotic_pricer.price_barrier('call', 'up_and_out', barrier_level=130.0, n_simulations=50000, n_steps=100)
        bs_vanilla = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, 'call')

        # Up-and-out call must be strictly cheaper than vanilla call
        self.assertLess(res_uo.price, bs_vanilla)
        self.assertGreater(res_uo.price, 0.0)
        self.assertGreater(res_uo.hit_probability, 0.0)

    def test_lookback_option_properties(self):
        exotic_pricer = ExoticOptionPricer(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        res_float_call = exotic_pricer.price_lookback('call', 'floating', n_simulations=50000, n_steps=100)
        bs_vanilla = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, 'call')

        # Floating strike lookback call (S_T - min S) is strictly more valuable than vanilla call
        self.assertGreater(res_float_call.price, bs_vanilla)

    def test_american_lsm_put(self):
        exotic_pricer = ExoticOptionPricer(self.S0, 110.0, self.T, 0.08, self.sigma, q=0.0)
        res_lsm = exotic_pricer.price_american_lsm('put', n_simulations=50000, n_steps=50, polynomial_degree=3)

        # American put via LSM must exceed European Black-Scholes price
        self.assertGreater(res_lsm.price, res_lsm.european_price)
        self.assertGreater(res_lsm.early_exercise_premium, 0.20)
        self.assertGreater(res_lsm.exercise_frequency_pct, 10.0)


if __name__ == '__main__':
    unittest.main()
