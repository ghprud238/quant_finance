import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
"""Unit tests for Analytical and Numerical Option Greeks (Module 18)."""

import unittest
import numpy as np

from pricing_models.black_scholes.engine import BlackScholesModel
from pricing_models.greeks.analytical import AnalyticalGreeks, GreeksResult
from pricing_models.greeks.numerical import NumericalGreeks


class TestOptionGreeks(unittest.TestCase):
    """Tests analytical Greek formulas and numerical finite-difference convergence."""

    def setUp(self):
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.20
        self.q = 0.0

    def test_delta_bounds_and_relationship(self):
        """Tests Delta bounds: Call in [0, 1], Put in [-1, 0], Call - Put = exp(-q*T)."""
        call_delta = AnalyticalGreeks.delta(self.S, self.K, self.T, self.r, self.sigma, self.q, "call")
        put_delta = AnalyticalGreeks.delta(self.S, self.K, self.T, self.r, self.sigma, self.q, "put")

        self.assertGreater(call_delta, 0.0)
        self.assertLess(call_delta, 1.0)
        self.assertLess(put_delta, 0.0)
        self.assertGreater(put_delta, -1.0)
        self.assertAlmostEqual(call_delta - put_delta, np.exp(-self.q * self.T), places=6)

    def test_gamma_properties(self):
        """Tests Gamma is strictly positive and identical for Calls and Puts."""
        gamma = AnalyticalGreeks.gamma(self.S, self.K, self.T, self.r, self.sigma, self.q)
        self.assertGreater(gamma, 0.0)

        # Gamma peaks near ATM (S=100) vs deep OTM (S=70)
        gamma_otm = AnalyticalGreeks.gamma(70.0, self.K, self.T, self.r, self.sigma, self.q)
        self.assertGreater(gamma, gamma_otm)

    def test_vega_properties(self):
        """Tests Vega is non-negative and identical for Calls and Puts."""
        vega = AnalyticalGreeks.vega(self.S, self.K, self.T, self.r, self.sigma, self.q)
        vega_pct = AnalyticalGreeks.vega_percentage(self.S, self.K, self.T, self.r, self.sigma, self.q)

        self.assertGreater(vega, 0.0)
        self.assertAlmostEqual(vega_pct, vega / 100.0, places=6)

    def test_theta_properties(self):
        """Tests Theta is negative (time decay) and daily conversion."""
        call_theta = AnalyticalGreeks.theta(self.S, self.K, self.T, self.r, self.sigma, self.q, "call")
        call_theta_d = AnalyticalGreeks.theta_daily(self.S, self.K, self.T, self.r, self.sigma, self.q, "call")

        self.assertLess(call_theta, 0.0)
        self.assertAlmostEqual(call_theta_d, call_theta / 365.0, places=6)

    def test_rho_properties(self):
        """Tests Rho is positive for Calls and negative for Puts."""
        call_rho = AnalyticalGreeks.rho(self.S, self.K, self.T, self.r, self.sigma, self.q, "call")
        put_rho = AnalyticalGreeks.rho(self.S, self.K, self.T, self.r, self.sigma, self.q, "put")

        self.assertGreater(call_rho, 0.0)
        self.assertLess(put_rho, 0.0)

    def test_analytical_vs_numerical_convergence(self):
        """Compares analytical Greeks against numerical finite differences."""
        spots = [85.0, 100.0, 115.0]
        options = ["call", "put"]

        for s in spots:
            for opt in options:
                # Delta
                ana_delta = AnalyticalGreeks.delta(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                num_delta = NumericalGreeks.delta(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                self.assertAlmostEqual(ana_delta, num_delta, places=4, msg=f"Delta mismatch at S={s}, opt={opt}")

                # Gamma
                ana_gamma = AnalyticalGreeks.gamma(s, self.K, self.T, self.r, self.sigma, self.q)
                num_gamma = NumericalGreeks.gamma(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                self.assertAlmostEqual(ana_gamma, num_gamma, places=4, msg=f"Gamma mismatch at S={s}, opt={opt}")

                # Vega
                ana_vega = AnalyticalGreeks.vega(s, self.K, self.T, self.r, self.sigma, self.q)
                num_vega = NumericalGreeks.vega(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                self.assertAlmostEqual(ana_vega, num_vega, places=4, msg=f"Vega mismatch at S={s}, opt={opt}")

                # Theta
                ana_theta = AnalyticalGreeks.theta(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                num_theta = NumericalGreeks.theta(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                self.assertAlmostEqual(ana_theta, num_theta, places=3, msg=f"Theta mismatch at S={s}, opt={opt}")

                # Rho
                ana_rho = AnalyticalGreeks.rho(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                num_rho = NumericalGreeks.rho(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                self.assertAlmostEqual(ana_rho, num_rho, places=4, msg=f"Rho mismatch at S={s}, opt={opt}")

                # Vanna
                ana_vanna = AnalyticalGreeks.vanna(s, self.K, self.T, self.r, self.sigma, self.q)
                num_vanna = NumericalGreeks.vanna(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                self.assertAlmostEqual(ana_vanna, num_vanna, places=3, msg=f"Vanna mismatch at S={s}, opt={opt}")

                # Volga
                ana_volga = AnalyticalGreeks.volga(s, self.K, self.T, self.r, self.sigma, self.q)
                num_volga = NumericalGreeks.volga(s, self.K, self.T, self.r, self.sigma, self.q, opt)
                self.assertAlmostEqual(ana_volga, num_volga, places=3, msg=f"Volga mismatch at S={s}, opt={opt}")

    def test_higher_order_greeks(self):
        """Tests higher order Greeks (Charm, Speed, Zomma, Color)."""
        charm = AnalyticalGreeks.charm(self.S, self.K, self.T, self.r, self.sigma, self.q, "call")
        speed = AnalyticalGreeks.speed(self.S, self.K, self.T, self.r, self.sigma, self.q)
        zomma = AnalyticalGreeks.zomma(self.S, self.K, self.T, self.r, self.sigma, self.q)
        color = AnalyticalGreeks.color(self.S, self.K, self.T, self.r, self.sigma, self.q)

        self.assertIsInstance(charm, float)
        self.assertIsInstance(speed, float)
        self.assertIsInstance(zomma, float)
        self.assertIsInstance(color, float)

    def test_greeks_result_dataclass(self):
        """Tests full calculate_all() container and serialization."""
        res = AnalyticalGreeks.calculate_all(self.S, self.K, self.T, self.r, self.sigma, self.q, "call")
        self.assertIsInstance(res, GreeksResult)
        self.assertAlmostEqual(res.price, 10.45058, places=4)
        d = res.to_dict()
        self.assertIn("delta", d)
        self.assertIn("gamma", d)
        self.assertIn("theta", d)
        self.assertIn("vega", d)
        self.assertIn("rho", d)


if __name__ == "__main__":
    unittest.main()
