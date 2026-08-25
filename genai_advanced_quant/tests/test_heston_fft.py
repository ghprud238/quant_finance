"""Unit tests for Project 32: Heston Stochastic Volatility & Carr-Madan FFT Calibration."""

import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm

from genai_advanced_quant.data.loader import generate_market_option_surface
from genai_advanced_quant.heston_fft.model import (
    HestonParameters,
    HestonOptionPricer,
    heston_characteristic_function,
    carr_madan_fft_price,
    fang_oosterlee_cos_price,
)


class TestHestonFFT(unittest.TestCase):
    """Validates characteristic function, Carr-Madan FFT, Fang-Oosterlee COS, and calibration."""
    
    def setUp(self):
        self.params = HestonParameters(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.30,
            rho=-0.70,
            r=0.05,
            q=0.01
        )
        self.pricer = HestonOptionPricer(self.params)
        
    def test_characteristic_function_at_zero(self):
        # phi(0) = E[exp(0)] = 1.0
        phi_zero = heston_characteristic_function(0.0, S0=100.0, T=1.0, params=self.params)
        self.assertAlmostEqual(abs(phi_zero), 1.0, places=5)
        
    def test_feller_condition(self):
        # 2 * kappa * theta / xi^2 = 2 * 2.0 * 0.04 / 0.09 = 0.16 / 0.09 = 1.777 > 1.0
        self.assertTrue(self.params.is_feller_satisfied)
        self.assertGreater(self.params.feller_ratio, 1.0)
        
    def test_black_scholes_limit_convergence(self):
        # When xi -> 0 and v0 = theta = sigma^2, Heston equals Black-Scholes
        sigma = 0.20
        bs_params = HestonParameters(
            v0=sigma**2,
            kappa=1.0,
            theta=sigma**2,
            xi=1e-5,
            rho=0.0,
            r=0.05,
            q=0.01
        )
        S0, K, T = 100.0, 100.0, 1.0
        d1 = (np.log(S0 / K) + (0.05 - 0.01 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_call = S0 * np.exp(-0.01 * T) * norm.cdf(d1) - K * np.exp(-0.05 * T) * norm.cdf(d2)
        
        cos_call = fang_oosterlee_cos_price(S0, K, T, bs_params, N=128)
        self.assertAlmostEqual(cos_call, bs_call, places=3)
        
    def test_cos_vs_fft_agreement(self):
        S0, T = 100.0, 0.5
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        
        fft_calls = carr_madan_fft_price(S0, strikes, T, self.params, option_type="call")
        cos_calls = [fang_oosterlee_cos_price(S0, K, T, self.params, N=128, option_type="call") for K in strikes]
        
        for c_fft, c_cos in zip(fft_calls, cos_calls):
            self.assertAlmostEqual(c_fft, c_cos, places=2)
            
    def test_put_call_parity(self):
        S0, K, T = 100.0, 105.0, 0.75
        c_price = self.pricer.price_call(S0, K, T, method="cos")
        p_price = self.pricer.price_put(S0, K, T, method="cos")
        
        # Parity: C - P = S0 * exp(-q*T) - K * exp(-r*T)
        parity_diff = (c_price - p_price) - (S0 * np.exp(-self.params.q * T) - K * np.exp(-self.params.r * T))
        self.assertAlmostEqual(parity_diff, 0.0, places=3)
        
    def test_surface_generation(self):
        iv_surface = self.pricer.implied_volatility_surface(S0=100.0)
        self.assertIsInstance(iv_surface, pd.DataFrame)
        self.assertGreater(iv_surface.shape[0], 3)
        self.assertGreater(iv_surface.shape[1], 5)
        # Check that all IV values are positive and reasonable
        self.assertTrue((iv_surface.values > 0.05).all())
        self.assertTrue((iv_surface.values < 1.0).all())
        
    def test_heston_calibration(self):
        market_surface = generate_market_option_surface(spot=100.0, r=0.05, q=0.01, seed=42)
        calib_res = self.pricer.calibrate(market_surface, spot=100.0, r=0.05, q=0.01)
        
        self.assertLess(calib_res.rmse, 0.50)
        self.assertGreater(calib_res.r_squared, 0.95)
        self.assertTrue(calib_res.feller_satisfied)
        self.assertFalse(calib_res.pricing_comparison_df.empty)


if __name__ == '__main__':
    unittest.main()
