"""Analytical Black-Scholes-Merton baseline and option market chain generator."""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class BlackScholesGreeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class BlackScholesAnalytical:
    """Exact Black-Scholes-Merton analytical pricer with continuous dividend yield."""

    @staticmethod
    def d1_d2(S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Tuple[float, float]:
        if T <= 0.0 or sigma <= 0.0:
            return 0.0, 0.0
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(d1), float(d2)

    @classmethod
    def price(
        cls,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> float:
        """Computes analytical Black-Scholes price."""
        if T <= 0.0:
            if option_type.lower() == "call":
                return float(max(S0 - K, 0.0))
            else:
                return float(max(K - S0, 0.0))

        d1, d2 = cls.d1_d2(S0, K, T, r, sigma, q)
        if option_type.lower() == "call":
            return float(S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type.lower() == "put":
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1))
        else:
            raise ValueError(f"Unknown option_type: {option_type}. Must be 'call' or 'put'.")

    @classmethod
    def greeks(
        cls,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> BlackScholesGreeks:
        """Computes exact analytical Greeks (Delta, Gamma, Vega, Theta, Rho)."""
        if T <= 0.0 or sigma <= 0.0:
            p = cls.price(S0, K, T, r, sigma, q, option_type)
            return BlackScholesGreeks(price=p, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

        d1, d2 = cls.d1_d2(S0, K, T, r, sigma, q)
        sqrt_T = np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)

        opt = option_type.lower()
        if opt == "call":
            p = float(S0 * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2))
            delta = float(disc_q * norm.cdf(d1))
            theta = float(
                -(S0 * disc_q * pdf_d1 * sigma) / (2 * sqrt_T)
                - r * K * disc_r * norm.cdf(d2)
                + q * S0 * disc_q * norm.cdf(d1)
            )
            rho = float(K * T * disc_r * norm.cdf(d2))
        elif opt == "put":
            p = float(K * disc_r * norm.cdf(-d2) - S0 * disc_q * norm.cdf(-d1))
            delta = float(-disc_q * norm.cdf(-d1))
            theta = float(
                -(S0 * disc_q * pdf_d1 * sigma) / (2 * sqrt_T)
                + r * K * disc_r * norm.cdf(-d2)
                - q * S0 * disc_q * norm.cdf(-d1)
            )
            rho = float(-K * T * disc_r * norm.cdf(-d2))
        else:
            raise ValueError(f"Unknown option_type: {option_type}")

        gamma = float((disc_q * pdf_d1) / (S0 * sigma * sqrt_T))
        vega = float(S0 * disc_q * sqrt_T * pdf_d1)

        return BlackScholesGreeks(
            price=p,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
        )


@dataclass
class OptionMarketChain:
    """Option chain container."""
    underlying_price: float
    risk_free_rate: float
    dividend_yield: float
    quotes_df: pd.DataFrame


def generate_sample_option_chain(
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.01,
    base_sigma: float = 0.20,
    maturities: Optional[List[float]] = None,
    strikes: Optional[List[float]] = None,
) -> OptionMarketChain:
    """Generates realistic option chain with volatility skew/smile."""
    if maturities is None:
        maturities = [1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0, 1.0]
    if strikes is None:
        strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]

    rows = []
    for T in maturities:
        days = int(round(T * 365))
        for K in strikes:
            moneyness = np.log(K / S0)
            iv = base_sigma - 0.08 * moneyness + 0.14 * (moneyness ** 2)
            iv = float(np.clip(iv, 0.05, 0.80))

            call_px = BlackScholesAnalytical.price(S0, K, T, r, iv, q, "call")
            put_px = BlackScholesAnalytical.price(S0, K, T, r, iv, q, "put")

            call_greeks = BlackScholesAnalytical.greeks(S0, K, T, r, iv, q, "call")
            put_greeks = BlackScholesAnalytical.greeks(S0, K, T, r, iv, q, "put")

            rows.append({
                "Maturity_Years": T,
                "Days_to_Expiry": days,
                "Strike": K,
                "Moneyness_K_over_S": K / S0,
                "Implied_Vol": iv,
                "Call_Price": call_px,
                "Call_Delta": call_greeks.delta,
                "Call_Gamma": call_greeks.gamma,
                "Call_Theta": call_greeks.theta / 365.0,
                "Call_Vega": call_greeks.vega / 100.0,
                "Put_Price": put_px,
                "Put_Delta": put_greeks.delta,
                "Put_Gamma": put_greeks.gamma,
                "Put_Theta": put_greeks.theta / 365.0,
                "Put_Vega": put_greeks.vega / 100.0,
            })

    df = pd.DataFrame(rows)
    return OptionMarketChain(
        underlying_price=S0,
        risk_free_rate=r,
        dividend_yield=q,
        quotes_df=df,
    )
