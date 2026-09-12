"""
Black-Scholes-Merton European Option Pricing, Greeks Suite & SVI Implied Volatility Surface.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm


@dataclass
class OptionGreeks:
    """Option Greek sensitivities."""
    delta: float
    gamma: float
    theta: float
    theta_daily: float
    vega: float
    vega_pct: float
    rho: float
    rho_pct: float
    vanna: float
    volga: float
    charm: float


@dataclass
class OptionPriceResult:
    """Output container for Black-Scholes evaluation."""
    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float
    option_type: str
    price: float
    intrinsic_value: float
    time_value: float
    d1: float
    d2: float
    greeks: OptionGreeks
    put_call_parity_diff: float


class BlackScholesEngine:
    """
    Closed-form Black-Scholes-Merton pricing and analytical Greeks engine.
    Supports continuous dividend yield (Merton 1973 expansion).
    """

    @staticmethod
    def calculate_d1_d2(
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        risk_free_rate: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        dividend_yield: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute standard d1 and d2 parameters."""
        s = np.maximum(spot, 1e-8)
        k = np.maximum(strike, 1e-8)
        t = np.maximum(time_to_expiry, 1e-8)
        v = np.maximum(volatility, 1e-8)

        num = np.log(s / k) + (risk_free_rate - dividend_yield + 0.5 * v**2) * t
        den = v * np.sqrt(t)
        d1 = num / den
        d2 = d1 - den
        return d1, d2

    @classmethod
    def price(
        cls,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        option_type: str = "call",
    ) -> float:
        """Compute closed-form Black-Scholes price."""
        opt_type = option_type.lower()
        if time_to_expiry <= 1e-8:
            if opt_type == "call":
                return max(0.0, spot - strike)
            else:
                return max(0.0, strike - spot)

        d1, d2 = cls.calculate_d1_d2(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
        )
        df_r = np.exp(-risk_free_rate * time_to_expiry)
        df_q = np.exp(-dividend_yield * time_to_expiry)

        if opt_type == "call":
            return float(spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2))
        elif opt_type == "put":
            return float(strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1))
        else:
            raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'.")

    @classmethod
    def calculate_greeks(
        cls,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        option_type: str = "call",
    ) -> OptionGreeks:
        """Compute full 1st and 2nd order analytical Greek sensitivities."""
        opt_type = option_type.lower()
        t = max(time_to_expiry, 1e-8)
        s = max(spot, 1e-8)
        v = max(volatility, 1e-8)
        k = max(strike, 1e-8)
        r = risk_free_rate
        q = dividend_yield

        d1, d2 = cls.calculate_d1_d2(s, k, t, r, v, q)
        df_r = np.exp(-r * t)
        df_q = np.exp(-q * t)
        phi_d1 = norm.pdf(d1)
        sqrt_t = np.sqrt(t)

        # 1st-Order Greeks
        if opt_type == "call":
            delta = float(df_q * norm.cdf(d1))
            rho = float(k * t * df_r * norm.cdf(d2))
            theta = float(
                - (s * df_q * phi_d1 * v) / (2.0 * sqrt_t)
                - r * k * df_r * norm.cdf(d2)
                + q * s * df_q * norm.cdf(d1)
            )
            charm = float(
                q * df_q * norm.cdf(d1)
                - df_q * phi_d1 * ((2.0 * (r - q) * t - d2 * v * sqrt_t) / (2.0 * t * v * sqrt_t))
            )
        else:
            delta = float(-df_q * norm.cdf(-d1))
            rho = float(-k * t * df_r * norm.cdf(-d2))
            theta = float(
                - (s * df_q * phi_d1 * v) / (2.0 * sqrt_t)
                + r * k * df_r * norm.cdf(-d2)
                - q * s * df_q * norm.cdf(-d1)
            )
            charm = float(
                -q * df_q * norm.cdf(-d1)
                - df_q * phi_d1 * ((2.0 * (r - q) * t - d2 * v * sqrt_t) / (2.0 * t * v * sqrt_t))
            )

        # 2nd-Order Greeks (symmetric between calls and puts under continuous dividends)
        gamma = float((df_q * phi_d1) / (s * v * sqrt_t))
        vega = float(s * df_q * sqrt_t * phi_d1)
        vanna = float(-df_q * phi_d1 * (d2 / v))
        volga = float(vega * (d1 * d2 / v))

        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            theta_daily=theta / 365.0,
            vega=vega,
            vega_pct=vega / 100.0,
            rho=rho,
            rho_pct=rho / 100.0,
            vanna=vanna,
            volga=volga,
            charm=charm,
        )

    @classmethod
    def evaluate(
        cls,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        option_type: str = "call",
    ) -> OptionPriceResult:
        """Full evaluation returning price, greeks, and parity verification."""
        opt_type = option_type.lower()
        p = cls.price(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, opt_type)
        greeks = cls.calculate_greeks(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, opt_type)
        d1, d2 = cls.calculate_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)

        c_val = p if opt_type == "call" else cls.price(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "call")
        p_val = p if opt_type == "put" else cls.price(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "put")
        fwd_disc = spot * np.exp(-dividend_yield * time_to_expiry) - strike * np.exp(-risk_free_rate * time_to_expiry)
        parity_diff = abs((c_val - p_val) - fwd_disc)

        intrinsic = max(0.0, (spot - strike) if opt_type == "call" else (strike - spot))
        time_val = max(0.0, p - intrinsic)

        return OptionPriceResult(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=opt_type,
            price=p,
            intrinsic_value=intrinsic,
            time_value=time_val,
            d1=float(d1),
            d2=float(d2),
            greeks=greeks,
            put_call_parity_diff=float(parity_diff),
        )


class ImpliedVolatilitySolver:
    """
    Robust root-finder for inverting Black-Scholes formula to recover Implied Volatility.
    Uses Newton-Raphson with Vega as gradient and Brent's method fallback.
    """

    @classmethod
    def solve(
        cls,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        option_type: str = "call",
        tolerance: float = 1e-7,
        max_iterations: int = 100,
    ) -> float:
        """Invert market price to implied volatility."""
        opt_type = option_type.lower()
        df_r = np.exp(-risk_free_rate * time_to_expiry)
        df_q = np.exp(-dividend_yield * time_to_expiry)

        # No-arbitrage boundaries
        intrinsic = max(0.0, (spot * df_q - strike * df_r) if opt_type == "call" else (strike * df_r - spot * df_q))
        max_price = spot * df_q if opt_type == "call" else strike * df_r

        if market_price <= intrinsic + 1e-7:
            return 1e-4
        if market_price >= max_price - 1e-7:
            return 5.0

        # Initial guess via Brenner-Subrahmanyam (1988) / Corrado-Miller
        sigma = np.sqrt(2.0 * np.pi / max(time_to_expiry, 1e-4)) * (market_price / spot)
        sigma = np.clip(sigma, 0.05, 1.50)

        # Newton-Raphson iteration
        for _ in range(max_iterations):
            p = BlackScholesEngine.price(spot, strike, time_to_expiry, risk_free_rate, sigma, dividend_yield, opt_type)
            diff = p - market_price
            if abs(diff) < tolerance:
                return float(sigma)

            greeks = BlackScholesEngine.calculate_greeks(spot, strike, time_to_expiry, risk_free_rate, sigma, dividend_yield, opt_type)
            vega = greeks.vega
            if vega < 1e-6:
                break  # switch to Brent
            sigma -= diff / vega
            if sigma <= 0.001 or sigma > 5.0:
                break

        # Brent's bracketed fallback
        def obj(v: float) -> float:
            return BlackScholesEngine.price(spot, strike, time_to_expiry, risk_free_rate, v, dividend_yield, opt_type) - market_price

        try:
            return float(brentq(obj, 1e-4, 8.0, xtol=tolerance))
        except Exception:
            return float(np.clip(sigma, 0.01, 5.0))


@dataclass
class SVIParameters:
    """Gatheral Raw SVI Total Variance Parameters: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, log_moneyness: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate SVI total variance w(k)."""
        k = log_moneyness
        disc = np.sqrt((k - self.m)**2 + self.sigma**2)
        return self.a + self.b * (self.rho * (k - self.m) + disc)

    def implied_volatility(self, log_moneyness: Union[float, np.ndarray], time_to_expiry: float) -> Union[float, np.ndarray]:
        """Convert SVI total variance w(k) to annualized implied volatility sigma(k)."""
        w = np.maximum(self.total_variance(log_moneyness), 1e-8)
        return np.sqrt(w / max(time_to_expiry, 1e-6))


class SVIModel:
    """Gatheral (2004) Raw SVI Parameterization and Surface Calibrator."""

    @classmethod
    def calibrate_slice(
        cls,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        spot: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
    ) -> SVIParameters:
        """Calibrate Raw SVI parameters to market implied volatility smile slice."""
        forward = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)
        k = np.log(strikes / forward)
        w_market = (implied_vols**2) * time_to_expiry

        # Initial heuristics
        a0 = np.min(w_market) * 0.5
        b0 = 0.10
        rho0 = -0.30
        m0 = 0.0
        sigma0 = 0.10
        init_guess = [a0, b0, rho0, m0, sigma0]

        bounds = [
            (1e-5, 5.0),      # a
            (1e-5, 2.0),      # b
            (-0.999, 0.999),  # rho
            (-2.0, 2.0),      # m
            (1e-4, 1.0),      # sigma
        ]

        def loss(p: List[float]) -> float:
            a, b, rho, m, sig = p
            # Butterfly / non-negativity constraint
            if a + b * sig * np.sqrt(1.0 - rho**2) < 0:
                return 1e6
            model_w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sig**2))
            return float(np.sum((model_w - w_market)**2))

        res = minimize(loss, init_guess, method="L-BFGS-B", bounds=bounds)
        a, b, rho, m, sig = res.x
        return SVIParameters(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sig))
