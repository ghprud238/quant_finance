import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
"""Unit tests for Binomial Tree Lattice Option Pricer."""

import unittest
import numpy as np
from pricing_models.data import BlackScholesAnalytical
from pricing_models.binomial_tree import BinomialTreePricer


class TestBinomialTreePricer(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.20
        self.q = 0.0

    def test_crr_european_call_convergence(self):
        bs_call = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, 'call')
        pricer = BinomialTreePricer(self.S0, self.K, self.T, self.r, self.sigma, self.q, n_steps=200)
        res = pricer.price('call', 'european', model='crr')

        self.assertAlmostEqual(res.price, bs_call, delta=0.08)
        self.assertEqual(res.early_exercise_premium, 0.0)

    def test_crr_european_put_convergence(self):
        bs_put = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, 'put')
        pricer = BinomialTreePricer(self.S0, self.K, self.T, self.r, self.sigma, self.q, n_steps=200)
        res = pricer.price('put', 'european', model='crr')

        self.assertAlmostEqual(res.price, bs_put, delta=0.08)

    def test_american_put_early_exercise_premium(self):
        # American put with positive interest rate must have positive early exercise premium
        pricer = BinomialTreePricer(self.S0, 110.0, self.T, 0.08, self.sigma, self.q, n_steps=150)
        res = pricer.price('put', 'american', model='crr')

        self.assertGreater(res.american_price, res.european_price)
        self.assertGreater(res.early_exercise_premium, 0.20)

    def test_american_call_no_dividend_equals_european(self):
        # American call with q=0 should equal European call (Merton 1973 theorem)
        pricer = BinomialTreePricer(self.S0, self.K, self.T, self.r, self.sigma, q=0.0, n_steps=150)
        res = pricer.price('call', 'american', model='crr')

        self.assertAlmostEqual(res.american_price, res.european_price, delta=1e-5)
        self.assertAlmostEqual(res.early_exercise_premium, 0.0, delta=1e-5)

    def test_american_call_with_dividend_early_exercise(self):
        # American call with high dividend yield (q > r) should have early exercise premium
        pricer = BinomialTreePricer(self.S0, 90.0, self.T, 0.02, 0.25, q=0.10, n_steps=150)
        res = pricer.price('call', 'american', model='crr')

        self.assertGreater(res.american_price, res.european_price)
        self.assertGreater(res.early_exercise_premium, 0.10)

    def test_jarrow_rudd_and_leisen_reimer_convergence(self):
        bs_call = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, 'call')
        pricer = BinomialTreePricer(self.S0, self.K, self.T, self.r, self.sigma, self.q, n_steps=151)

        res_jr = pricer.price('call', 'european', model='jr')
        res_lr = pricer.price('call', 'european', model='lr')

        self.assertAlmostEqual(res_jr.price, bs_call, delta=0.10)
        self.assertAlmostEqual(res_lr.price, bs_call, delta=0.02)  # LR has faster convergence

    def test_lattice_greeks(self):
        pricer = BinomialTreePricer(self.S0, self.K, self.T, self.r, self.sigma, self.q, n_steps=100)
        res_call = pricer.price('call', 'european', model='crr')
        res_put = pricer.price('put', 'european', model='crr')

        # Call Delta in (0, 1), Put Delta in (-1, 0)
        self.assertGreater(res_call.greeks.delta, 0.40)
        self.assertLess(res_call.greeks.delta, 0.80)
        self.assertGreater(res_put.greeks.delta, -0.60)
        self.assertLess(res_put.greeks.delta, -0.20)

        # Gamma must be positive
        self.assertGreater(res_call.greeks.gamma, 0.0)
        self.assertGreater(res_put.greeks.gamma, 0.0)

    def test_build_tree_inspection(self):
        pricer = BinomialTreePricer(self.S0, self.K, self.T, self.r, self.sigma, self.q, n_steps=3)
        tree = pricer.build_tree(n_steps=3, option_type='call', exercise_style='american', model='crr')

        self.assertEqual(len(tree), 4)  # steps 0, 1, 2, 3
        self.assertEqual(len(tree[0]), 1)
        self.assertEqual(len(tree[1]), 2)
        self.assertEqual(len(tree[2]), 3)
        self.assertEqual(len(tree[3]), 4)

        # Root node stock price must equal S0
        self.assertAlmostEqual(tree[0][0].stock_price, self.S0)
        self.assertGreater(tree[0][0].option_value, 0.0)


if __name__ == '__main__':
    unittest.main()
