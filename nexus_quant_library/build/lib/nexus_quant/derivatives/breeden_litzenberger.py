"""
Breeden-Litzenberger (1978) Risk-Neutral Probability Density Extraction with Arbitrage-Free Smoothing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from .black_scholes import BlackScholesEngine, ImpliedVolatilitySolver, SVIModel


@dataclass
class RiskNeutralMoments:
    """Implied risk-neutral statistical moments."""
    mean: float
    variance: float
    volatility: float
    skewness: float
    kurtosis: float
    excess_kurtosis: float
    forward_price: float
    martingale_error_pct: float


@dataclass
class DensityExtractionResult:
    """Output container for Breeden-Litzenberger risk-neutral density estimation."""
    strikes: np.ndarray
    density: np.ndarray
    cumulative_density: np.ndarray
    smoothed_implied_vol: np.ndarray
    smoothed_call_prices: np.ndarray
    moments: RiskNeutralMoments
    kl_divergence_vs_bs: float
    crash_prob_10pct: float
    crash_prob_20pct: float
    rally_prob_10pct: float
    rally_prob_20pct: float


class BreedenLitzenbergerDensityEstimator:
    """
    Extracts the continuous risk-neutral probability density function q(K) from option prices:
    q(K) = exp(r * T) * (d^2 C / dK^2).
    """

    def __init__(
        self,
        spot: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.04,
        dividend_yield: float = 0.0,
        n_dense_strikes: int = 1000,
        strike_span_pct: float = 0.70,
    ):
        self.spot = spot
        self.time_to_expiry = time_to_expiry
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.n_dense_strikes = n_dense_strikes
        self.strike_span_pct = strike_span_pct
        self.forward_price = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)

    def extract_from_iv(
        self,
        market_strikes: np.ndarray,
        market_implied_vols: np.ndarray,
        smoothing_method: str = "spline",
    ) -> DensityExtractionResult:
        """
        Extract risk-neutral density from market implied volatilities.
        Applies smoothing in volatility space to avoid negative probabilities and noise explosion.
        """
        s0 = self.spot
        t = self.time_to_expiry
        r = self.risk_free_rate
        q = self.dividend_yield

        # Create dense strike grid spanning [S0*(1-span), S0*(1+span)]
        k_min = max(1.0, s0 * (1.0 - self.strike_span_pct))
        k_max = s0 * (1.0 + self.strike_span_pct)
        dense_strikes = np.linspace(k_min, k_max, self.n_dense_strikes)
        dk = dense_strikes[1] - dense_strikes[0]

        # Smooth implied volatility curve
        if smoothing_method.lower() == "svi":
            svi_params = SVIModel.calibrate_slice(
                market_strikes, market_implied_vols, s0, t, r, q
            )
            k_dense = np.log(dense_strikes / self.forward_price)
            smoothed_iv = svi_params.implied_volatility(k_dense, t)
        else:
            # Natural cubic spline with flat extrapolation
            cs = CubicSpline(market_strikes, market_implied_vols, bc_type="natural")
            smoothed_iv = cs(dense_strikes)
            # Boundary clamp
            smoothed_iv = np.clip(smoothed_iv, 0.05, 2.50)

        # Generate smooth call price curve via Black-Scholes
        smoothed_calls = np.zeros(self.n_dense_strikes)
        for i, (k_val, iv_val) in enumerate(zip(dense_strikes, smoothed_iv)):
            smoothed_calls[i] = BlackScholesEngine.price(
                s0, k_val, t, r, iv_val, q, option_type="call"
            )

        # Compute 2nd derivative: q(K) = exp(r*T) * (C_{i-1} - 2*C_i + C_{i+1}) / (dK^2)
        d2c_dk2 = np.gradient(np.gradient(smoothed_calls, dense_strikes), dense_strikes)
        raw_density = np.exp(r * t) * d2c_dk2

        # Enforce non-negativity and normalize integral to unity
        density = np.maximum(raw_density, 0.0)
        total_mass = np.trapz(density, dense_strikes)
        if total_mass > 1e-8:
            density /= total_mass
        else:
            # fallback to lognormal if numerical breakdown
            sigma_atm = float(np.interp(s0, market_strikes, market_implied_vols))
            fwd = self.forward_price
            d1 = (np.log(fwd / dense_strikes) + 0.5 * (sigma_atm**2) * t) / (sigma_atm * np.sqrt(t))
            d2 = d1 - sigma_atm * np.sqrt(t)
            density = norm.pdf(d2) / (dense_strikes * sigma_atm * np.sqrt(t))
            density /= np.trapz(density, dense_strikes)

        cdf = np.cumsum(density) * dk
        cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)

        # Compute implied risk-neutral moments
        mean_rn = float(np.trapz(dense_strikes * density, dense_strikes))
        var_rn = float(np.trapz(((dense_strikes - mean_rn)**2) * density, dense_strikes))
        vol_rn = float(np.sqrt(max(var_rn, 1e-8)))
        skew_rn = float(np.trapz(((dense_strikes - mean_rn)**3) * density, dense_strikes) / (vol_rn**3))
        kurt_rn = float(np.trapz(((dense_strikes - mean_rn)**4) * density, dense_strikes) / (vol_rn**4))

        martingale_err = (mean_rn - self.forward_price) / self.forward_price

        moments = RiskNeutralMoments(
            mean=mean_rn,
            variance=var_rn,
            volatility=vol_rn,
            skewness=skew_rn,
            kurtosis=kurt_rn,
            excess_kurtosis=kurt_rn - 3.0,
            forward_price=self.forward_price,
            martingale_error_pct=float(martingale_err * 100.0),
        )

        # Benchmark Black-Scholes lognormal density with ATM volatility
        sigma_atm = float(np.interp(s0, market_strikes, market_implied_vols))
        d2_bs = (np.log(self.forward_price / dense_strikes) - 0.5 * (sigma_atm**2) * t) / (sigma_atm * np.sqrt(t))
        bs_density = norm.pdf(d2_bs) / (dense_strikes * sigma_atm * np.sqrt(t))
        bs_density /= np.trapz(bs_density, dense_strikes)

        # Kullback-Leibler (KL) Divergence: D_KL(q || p_bs) = integral q(x) * ln(q(x) / p(x)) dx
        eps = 1e-12
        kl_integrand = density * np.log((density + eps) / (bs_density + eps))
        kl_div = float(max(0.0, np.trapz(kl_integrand, dense_strikes)))

        # Crash & Rally Tail Probabilities
        k_10pct_down = s0 * 0.90
        k_20pct_down = s0 * 0.80
        k_10pct_up = s0 * 1.10
        k_20pct_up = s0 * 1.20

        mask_10_down = dense_strikes <= k_10pct_down
        mask_20_down = dense_strikes <= k_20pct_down
        mask_10_up = dense_strikes >= k_10pct_up
        mask_20_up = dense_strikes >= k_20pct_up

        p_crash_10 = float(np.trapz(density[mask_10_down], dense_strikes[mask_10_down])) if np.any(mask_10_down) else 0.0
        p_crash_20 = float(np.trapz(density[mask_20_down], dense_strikes[mask_20_down])) if np.any(mask_20_down) else 0.0
        p_rally_10 = float(np.trapz(density[mask_10_up], dense_strikes[mask_10_up])) if np.any(mask_10_up) else 0.0
        p_rally_20 = float(np.trapz(density[mask_20_up], dense_strikes[mask_20_up])) if np.any(mask_20_up) else 0.0

        return DensityExtractionResult(
            strikes=dense_strikes,
            density=density,
            cumulative_density=cdf,
            smoothed_implied_vol=smoothed_iv,
            smoothed_call_prices=smoothed_calls,
            moments=moments,
            kl_divergence_vs_bs=kl_div,
            crash_prob_10pct=p_crash_10,
            crash_prob_20pct=p_crash_20,
            rally_prob_10pct=p_rally_10,
            rally_prob_20pct=p_rally_20,
        )

    def extract_from_option_prices(
        self,
        strikes: np.ndarray,
        call_prices: np.ndarray,
    ) -> DensityExtractionResult:
        """Helper to extract density by first inverting market call prices to implied volatilities."""
        s0 = self.spot
        t = self.time_to_expiry
        r = self.risk_free_rate
        q = self.dividend_yield

        implied_vols = []
        for k_val, c_val in zip(strikes, call_prices):
            iv = ImpliedVolatilitySolver.solve(c_val, s0, k_val, t, r, q, option_type="call")
            implied_vols.append(iv)
        return self.extract_from_iv(strikes, np.array(implied_vols))
