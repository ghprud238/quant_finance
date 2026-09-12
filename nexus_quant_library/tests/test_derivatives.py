"""
Unit tests for Derivatives, Option Pricing, Greeks, Heston FFT/COS & Breeden-Litzenberger RND.
"""

import unittest
import numpy as np
import pandas as pd
from nexus_quant.derivatives.black_scholes import (
    BlackScholesEngine,
    ImpliedVolatilitySolver,
    SVIModel,
)
from nexus_quant.derivatives.heston_fft import (
    CarrMadanFFTPricer,
    FangOosterleeCOSPricer,
    HestonCalibrator,
    HestonCharacteristicFunction,
    HestonParameters,
)
from nexus_quant.derivatives.breeden_litzenberger import (
    BreedenLitzenbergerDensityEstimator,
)


class TestDerivativesSuite(unittest.TestCase):
    """Test suite for derivatives pricing and state-price extraction."""

    def setUp(self):
        self.spot = 100.0
        self.strike = 100.0
        self.time_to_expiry = 1.0
        self.risk_free_rate = 0.05
        self.volatility = 0.20
        self.dividend_yield = 0.01

    def test_black_scholes_pricing_and_parity(self):
        """Test BSM European call & put pricing and Put-Call Parity exactness."""
        res_call = BlackScholesEngine.evaluate(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, self.dividend_yield, "call"
        )
        res_put = BlackScholesEngine.evaluate(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, self.dividend_yield, "put"
        )

        self.assertGreater(res_call.price, 0.0)
        self.assertGreater(res_put.price, 0.0)
        self.assertLess(res_call.put_call_parity_diff, 1e-10)
        self.assertAlmostEqual(res_call.price - res_put.price,
                               self.spot * np.exp(-0.01) - self.strike * np.exp(-0.05), places=5)

    def test_analytical_greeks_bounds(self):
        """Test 1st and 2nd order analytical Greek sensitivities and signs."""
        greeks_call = BlackScholesEngine.calculate_greeks(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, self.dividend_yield, "call"
        )
        greeks_put = BlackScholesEngine.calculate_greeks(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, self.dividend_yield, "put"
        )

        # Call Delta in (0, 1), Put Delta in (-1, 0)
        self.assertGreater(greeks_call.delta, 0.0)
        self.assertLess(greeks_call.delta, 1.0)
        self.assertLess(greeks_put.delta, 0.0)
        self.assertGreater(greeks_put.delta, -1.0)

        # Gamma and Vega strictly positive
        self.assertGreater(greeks_call.gamma, 0.0)
        self.assertGreater(greeks_call.vega, 0.0)
        self.assertAlmostEqual(greeks_call.gamma, greeks_put.gamma, places=8)
        self.assertAlmostEqual(greeks_call.vega, greeks_put.vega, places=8)

        # Negative time decay (Theta)
        self.assertLess(greeks_call.theta, 0.0)

    def test_implied_volatility_inversion(self):
        """Test root-finder recovery of implied volatility from market price."""
        target_iv = 0.285
        market_price = BlackScholesEngine.price(
            self.spot, 105.0, 0.5, self.risk_free_rate, target_iv, self.dividend_yield, "call"
        )
        recovered_iv = ImpliedVolatilitySolver.solve(
            market_price, self.spot, 105.0, 0.5, self.risk_free_rate, self.dividend_yield, "call"
        )
        self.assertAlmostEqual(target_iv, recovered_iv, places=5)

    def test_svi_smile_calibration(self):
        """Test Gatheral Raw SVI parameter calibration on smile slice."""
        strikes = np.array([80, 90, 100, 110, 120])
        market_ivs = np.array([0.26, 0.23, 0.20, 0.19, 0.21])
        svi_params = SVIModel.calibrate_slice(strikes, market_ivs, self.spot, 1.0, 0.05, 0.0)

        self.assertGreater(svi_params.b, 0.0)
        self.assertGreater(svi_params.sigma, 0.0)
        # Evaluated IV should match market closely
        fwd = self.spot * np.exp(0.05)
        k_atm = np.log(100.0 / fwd)
        iv_model = svi_params.implied_volatility(k_atm, 1.0)
        self.assertAlmostEqual(iv_model, 0.20, delta=0.03)

    def test_heston_characteristic_function_and_fft_cos(self):
        """Test Heston stable characteristic function and FFT vs COS pricing agreement."""
        params = HestonParameters(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.60, r=0.05, q=0.01
        )
        self.assertTrue(params.is_feller_satisfied)

        # Characteristic function at u=0 must equal 1.0 (or total probability)
        phi_0 = HestonCharacteristicFunction.evaluate(0.0, 1.0, self.spot, params)
        self.assertAlmostEqual(np.abs(phi_0), 1.0, places=6)

        # Compare COS vs FFT pricing across strikes
        strikes = np.array([90.0, 100.0, 110.0])
        fft_prices = CarrMadanFFTPricer.price_call_strip(self.spot, strikes, 1.0, params)
        cos_prices = np.array([
            FangOosterleeCOSPricer.price(self.spot, k, 1.0, params, "call")
            for k in strikes
        ])

        # Agreement within 0.05 cents
        for p_fft, p_cos in zip(fft_prices, cos_prices):
            self.assertAlmostEqual(p_fft, p_cos, delta=0.05)

    def test_breeden_litzenberger_density_extraction(self):
        """Test Breeden-Litzenberger risk-neutral density extraction and moment properties."""
        estimator = BreedenLitzenbergerDensityEstimator(
            spot=self.spot, time_to_expiry=1.0, risk_free_rate=0.05, dividend_yield=0.01,
            n_dense_strikes=1000
        )
        strikes = np.linspace(60, 150, 19)
        log_m = np.log(strikes / self.spot)
        # Asymmetric downward equity skew
        skewed_ivs = 0.20 - 0.15 * log_m + 0.10 * (log_m**2)

        res = estimator.extract_from_iv(strikes, skewed_ivs, smoothing_method="spline")

        # 1. Non-negativity
        self.assertTrue(np.all(res.density >= -1e-8))
        # 2. Total mass integrates to ~1.0
        dk = res.strikes[1] - res.strikes[0]
        total_mass = np.sum(res.density) * dk
        self.assertAlmostEqual(total_mass, 1.0, delta=0.02)
        # 3. Martingale mean close to forward price
        self.assertAlmostEqual(res.moments.mean, estimator.forward_price, delta=1.50)
        # 4. Skewness is negative (reflecting implied crash risk)
        self.assertLess(res.moments.skewness, 0.0)
        # 5. Tail probabilities positive
        self.assertGreater(res.crash_prob_10pct, 0.0)


if __name__ == "__main__":
    unittest.main()
