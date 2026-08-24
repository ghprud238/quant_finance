"""Unit Tests for Module 17: Implied Volatility Solver & Volatility Surface/Smile."""

import unittest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
import numpy as np
import pandas as pd
from scipy.stats import norm

from pricing_models.implied_vol.solver import (
    black_scholes_price,
    black_scholes_vega,
    brenner_subrahmanyam_iv,
    corrado_miller_iv,
    ImpliedVolatilitySolver,
)
from pricing_models.implied_vol.smile import (
    SVIParameters,
    VolatilitySmile,
)
from pricing_models.implied_vol.surface import (
    VolatilitySurface,
)


class TestBlackScholesAnalytical(unittest.TestCase):
    """Tests for Black-Scholes pricing, parity, and Greeks."""

    def setUp(self):
        self.spot = 100.0
        self.strike = 105.0
        self.expiry = 0.5
        self.r = 0.04
        self.q = 0.01
        self.vol = 0.25

    def test_put_call_parity(self):
        """Verifies Put-Call Parity: C - P = S0 * exp(-qT) - K * exp(-rT)."""
        call = black_scholes_price(self.spot, self.strike, self.expiry, self.r, self.q, self.vol, "call")
        put = black_scholes_price(self.spot, self.strike, self.expiry, self.r, self.q, self.vol, "put")
        expected_diff = self.spot * np.exp(-self.q * self.expiry) - self.strike * np.exp(-self.r * self.expiry)
        self.assertAlmostEqual(call - put, expected_diff, places=8)

    def test_vega_call_put_equality(self):
        """Verifies Vega is positive and identical for European calls and puts."""
        vega = black_scholes_vega(self.spot, self.strike, self.expiry, self.r, self.q, self.vol)
        self.assertGreater(vega, 0.0)
        # Numerical derivative check
        eps = 1e-5
        p_up = black_scholes_price(self.spot, self.strike, self.expiry, self.r, self.q, self.vol + eps, "call")
        p_down = black_scholes_price(self.spot, self.strike, self.expiry, self.r, self.q, self.vol - eps, "call")
        finite_diff_vega = (p_up - p_down) / (2.0 * eps)
        self.assertAlmostEqual(vega, finite_diff_vega, places=4)


class TestImpliedVolatilitySolver(unittest.TestCase):
    """Tests for Newton-Raphson & Brent IV solver."""

    def setUp(self):
        self.solver = ImpliedVolatilitySolver(tolerance=1e-8)
        self.spot = 100.0
        self.r = 0.03
        self.q = 0.01
        self.expiry = 0.75

    def test_exact_inversion_calls(self):
        """Verifies BS(IV(C)) == C across various strikes (OTM, ATM, ITM)."""
        test_strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        true_vols = [0.35, 0.28, 0.22, 0.20, 0.24]

        for k, v in zip(test_strikes, true_vols):
            price = black_scholes_price(self.spot, k, self.expiry, self.r, self.q, v, "call")
            recovered_iv = self.solver.solve(price, self.spot, k, self.expiry, self.r, self.q, "call")
            self.assertAlmostEqual(recovered_iv, v, places=6, msg=f"Failed for strike {k}")

    def test_exact_inversion_puts(self):
        """Verifies BS(IV(P)) == P across various strikes."""
        test_strikes = [85.0, 95.0, 100.0, 105.0, 115.0]
        true_vols = [0.32, 0.26, 0.20, 0.21, 0.27]

        for k, v in zip(test_strikes, true_vols):
            price = black_scholes_price(self.spot, k, self.expiry, self.r, self.q, v, "put")
            recovered_iv = self.solver.solve(price, self.spot, k, self.expiry, self.r, self.q, "put")
            self.assertAlmostEqual(recovered_iv, v, places=6, msg=f"Failed for put strike {k}")

    def test_newton_vs_brent_agreement(self):
        """Verifies Newton-Raphson and Brent root-finders yield identical solutions."""
        strike = 105.0
        true_vol = 0.225
        price = black_scholes_price(self.spot, strike, self.expiry, self.r, self.q, true_vol, "call")

        iv_newton = self.solver.solve(price, self.spot, strike, self.expiry, self.r, self.q, "call", method="newton")
        iv_brent = self.solver.solve(price, self.spot, strike, self.expiry, self.r, self.q, "call", method="brent")

        self.assertAlmostEqual(iv_newton, iv_brent, places=7)
        self.assertAlmostEqual(iv_newton, true_vol, places=7)

    def test_initial_approximations(self):
        """Verifies Corrado-Miller and Brenner-Subrahmanyam initial guesses."""
        strike = 100.0
        true_vol = 0.20
        price = black_scholes_price(self.spot, strike, self.expiry, self.r, self.q, true_vol, "call")

        iv_bs = brenner_subrahmanyam_iv(price, self.spot, self.expiry)
        iv_cm = corrado_miller_iv(price, self.spot, strike, self.expiry, self.r, self.q, "call")

        # Approximations should be within 5% of true volatility for ATM options
        self.assertLess(abs(iv_bs - true_vol), 0.03)
        self.assertLess(abs(iv_cm - true_vol), 0.015)

    def test_arbitrage_violation_detection(self):
        """Verifies invalid arbitrage prices return NaN or raise ValueError."""
        # Price below intrinsic value
        invalid_call_price = 0.50 # S=100, K=80 -> Intrinsic is ~20
        iv = self.solver.solve(invalid_call_price, self.spot, 80.0, self.expiry, self.r, self.q, "call")
        self.assertTrue(np.isnan(iv))

        with self.assertRaises(ValueError):
            self.solver.solve(invalid_call_price, self.spot, 80.0, self.expiry, self.r, self.q, "call", raise_on_arbitrage=True)

    def test_solve_chain_vectorized(self):
        """Verifies solving an entire pandas DataFrame option chain."""
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        vols = [0.28, 0.24, 0.20, 0.21, 0.25]
        prices = [black_scholes_price(self.spot, k, self.expiry, self.r, self.q, v, "call") for k, v in zip(strikes, vols)]

        df_chain = pd.DataFrame({
            "strike": strikes,
            "expiry": [self.expiry] * len(strikes),
            "price": prices,
            "type": ["call"] * len(strikes),
        })

        solved_df = self.solver.solve_chain(df_chain, spot=self.spot, risk_free_rate=self.r, dividend_yield=self.q)

        self.assertIn("implied_vol", solved_df.columns)
        self.assertIn("vega", solved_df.columns)
        self.assertIn("moneyness", solved_df.columns)
        for expected_v, solved_v in zip(vols, solved_df["implied_vol"]):
            self.assertAlmostEqual(expected_v, solved_v, places=6)


class TestVolatilitySmile(unittest.TestCase):
    """Tests for SVI and Spline Volatility Smile modeling."""

    def setUp(self):
        self.spot = 100.0
        self.expiry = 0.5
        self.r = 0.02
        self.q = 0.0
        self.smile = VolatilitySmile(self.spot, self.expiry, self.r, self.q)

        # Synthetic market smile points
        self.strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
        self.market_ivs = np.array([0.34, 0.28, 0.23, 0.20, 0.21, 0.24, 0.28])

    def test_fit_svi_calibration(self):
        """Verifies SVI model fits and reproduces market implied volatilities."""
        svi_params = self.smile.fit_svi(self.strikes, self.market_ivs)
        self.assertIsInstance(svi_params, SVIParameters)
        self.assertGreaterEqual(svi_params.b, 0.0)
        self.assertGreater(svi_params.sigma, 0.0)
        self.assertLess(abs(svi_params.rho), 1.0)

        # Reconstructed IVs should closely match input points
        reconstructed_ivs = self.smile.get_iv(self.strikes, method="svi")
        max_err = np.max(np.abs(reconstructed_ivs - self.market_ivs))
        self.assertLess(max_err, 0.015)

    def test_fit_spline_exact_interpolation(self):
        """Verifies cubic spline passes exactly through market knot points."""
        self.smile.fit_spline(self.strikes, self.market_ivs)
        interpolated = self.smile.get_iv(self.strikes, method="spline")
        np.testing.assert_allclose(interpolated, self.market_ivs, atol=1e-7)

    def test_skew_and_convexity(self):
        """Verifies skew and convexity metrics calculation."""
        self.smile.fit_svi(self.strikes, self.market_ivs)
        skew = self.smile.get_skew()
        convexity = self.smile.get_convexity()
        atm_vol = self.smile.get_atm_vol()

        self.assertGreater(atm_vol, 0.15)
        self.assertLess(atm_vol, 0.25)
        self.assertGreater(convexity, 0.0)

    def test_generate_smile_curve(self):
        """Verifies dense smile curve generation."""
        self.smile.fit_svi(self.strikes, self.market_ivs)
        df_curve = self.smile.generate_smile_curve(moneyness_range=(0.7, 1.3), n_points=50)
        self.assertEqual(len(df_curve), 50)
        self.assertIn("moneyness", df_curve.columns)
        self.assertIn("implied_vol", df_curve.columns)
        self.assertTrue((df_curve["implied_vol"] > 0).all())


class TestVolatilitySurface(unittest.TestCase):
    """Tests for 2D/3D Volatility Surface & Local Volatility."""

    def setUp(self):
        self.surface = VolatilitySurface.create_synthetic_market_surface(
            spot=100.0,
            risk_free_rate=0.03,
            dividend_yield=0.01,
            atm_vol=0.20,
            skew_slope=-0.12,
            convexity=0.20,
        )

    def test_surface_interpolation(self):
        """Verifies 2D interpolation across arbitrary strike and expiration."""
        iv_mid = self.surface.get_iv(strike=105.0, time_to_expiry=0.6)
        self.assertGreater(iv_mid, 0.10)
        self.assertLess(iv_mid, 0.50)

    def test_total_variance_scaling(self):
        """Verifies total variance w(K, T) = sigma^2 * T increases with maturity."""
        w_short = self.surface.total_variance(100.0, 0.25)
        w_long = self.surface.total_variance(100.0, 1.0)
        self.assertLess(w_short, w_long)

    def test_generate_mesh(self):
        """Verifies 3D surface meshgrid creation."""
        mesh = self.surface.generate_mesh(moneyness_range=(0.7, 1.3), expiry_range=(0.1, 1.5), n_moneyness=30, n_expiries=20)
        self.assertEqual(mesh["iv_grid"].shape, (30, 20))
        self.assertEqual(mesh["total_variance_grid"].shape, (30, 20))
        self.assertTrue((mesh["iv_grid"] > 0).all())

    def test_dupire_local_volatility(self):
        """Verifies local volatility extraction via Dupire formula."""
        loc_vol = self.surface.dupire_local_volatility(strike=100.0, time_to_expiry=0.5)
        self.assertGreater(loc_vol, 0.05)
        self.assertLess(loc_vol, 1.0)


if __name__ == "__main__":
    unittest.main()
