"""Analytical Option Greeks Calculator under Black-Scholes-Merton Model."""

from dataclasses import dataclass
from typing import Union, Dict, Any, Optional
import numpy as np
from scipy.stats import norm

from ..black_scholes.engine import BlackScholesModel


@dataclass
class GreeksResult:
    """Container for complete suite of 1st, 2nd, and higher-order Greeks."""
    spot: float
    strike: float
    expiry: float
    rate: float
    volatility: float
    dividend: float
    option_type: str
    price: float
    
    # 1st Order Greeks
    delta: float
    theta: float          # Annualized theta
    theta_daily: float    # 1-day theta (theta / 365)
    vega: float           # Annualized vega (per unit vol)
    vega_pct: float       # Vega per 1% vol change (vega / 100)
    rho: float            # Annualized rho (per unit rate)
    rho_pct: float        # Rho per 1% rate change (rho / 100)
    
    # 2nd Order Greeks
    gamma: float
    vanna: float
    volga: float          # Vomma
    charm: float          # Delta decay (-dDelta/dT)
    
    # Higher Order Greeks
    speed: float
    zomma: float
    color: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spot": self.spot,
            "strike": self.strike,
            "expiry": self.expiry,
            "rate": self.rate,
            "volatility": self.volatility,
            "dividend": self.dividend,
            "option_type": self.option_type,
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "theta_daily": self.theta_daily,
            "vega": self.vega,
            "vega_pct": self.vega_pct,
            "rho": self.rho,
            "rho_pct": self.rho_pct,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "speed": self.speed,
            "zomma": self.zomma,
            "color": self.color,
        }


class AnalyticalGreeks:
    """Analytical Greeks for European options under Black-Scholes-Merton (1973)."""

    @classmethod
    def delta(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
        option_type: str = "call",
    ) -> Union[float, np.ndarray]:
        """Delta (dV/dS): sensitivity of option price to underlying spot price.
        Call: exp(-q*T) * N(d1)
        Put:  -exp(-q*T) * N(-d1)
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        disc_q = np.exp(-q_arr * T_arr)

        opt_type = option_type.lower()
        if opt_type in ("call", "c"):
            val = np.where(T_arr <= 0, np.where(S_arr > K_arr, 1.0, 0.0), disc_q * norm.cdf(d1_val))
        elif opt_type in ("put", "p"):
            val = np.where(T_arr <= 0, np.where(S_arr < K_arr, -1.0, 0.0), -disc_q * norm.cdf(-d1_val))
        else:
            raise ValueError(f"Invalid option_type: {option_type}.")
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def gamma(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Gamma (d^2V / dS^2): rate of change of Delta with respect to spot price.
        Gamma = exp(-q*T) * phi(d1) / (S * sigma * sqrt(T)) (identical for Calls and Puts).
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        denom = S_arr * sigma_arr * np.sqrt(np.maximum(T_arr, 1e-12))
        disc_q = np.exp(-q_arr * T_arr)

        gamma_val = np.where(T_arr <= 1e-8, 0.0, disc_q * norm.pdf(d1_val) / np.maximum(denom, 1e-12))
        return float(gamma_val) if np.ndim(gamma_val) == 0 else gamma_val

    @classmethod
    def vega(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Vega (dV/dsigma): sensitivity of option price to volatility.
        Vega = S * exp(-q*T) * sqrt(T) * phi(d1) (identical for Calls and Puts).
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        disc_q = np.exp(-q_arr * T_arr)
        vega_val = S_arr * disc_q * np.sqrt(np.maximum(T_arr, 0.0)) * norm.pdf(d1_val)
        return float(vega_val) if np.ndim(vega_val) == 0 else vega_val

    @classmethod
    def vega_percentage(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Vega per 1 percentage point (0.01) change in implied volatility: Vega / 100."""
        return cls.vega(S, K, T, r, sigma, q) / 100.0

    @classmethod
    def theta(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
        option_type: str = "call",
    ) -> Union[float, np.ndarray]:
        """Theta (-dV/dT = dV/dt): annualized time decay of option price.
        Call: - (S * exp(-q*T) * phi(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r*T) * N(d2) + q * S * exp(-q*T) * N(d1)
        Put:  - (S * exp(-q*T) * phi(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r*T) * N(-d2) - q * S * exp(-q*T) * N(-d1)
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        d2_val = BlackScholesModel.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)

        disc_q = np.exp(-q_arr * T_arr)
        disc_r = np.exp(-r_arr * T_arr)
        sqrt_T = np.sqrt(np.maximum(T_arr, 1e-12))

        term1 = - (S_arr * disc_q * norm.pdf(d1_val) * sigma_arr) / (2.0 * sqrt_T)

        opt_type = option_type.lower()
        if opt_type in ("call", "c"):
            term2 = - r_arr * K_arr * disc_r * norm.cdf(d2_val)
            term3 = q_arr * S_arr * disc_q * norm.cdf(d1_val)
            val = term1 + term2 + term3
        elif opt_type in ("put", "p"):
            term2 = r_arr * K_arr * disc_r * norm.cdf(-d2_val)
            term3 = - q_arr * S_arr * disc_q * norm.cdf(-d1_val)
            val = term1 + term2 + term3
        else:
            raise ValueError(f"Invalid option_type: {option_type}.")

        # Handle expired options
        val = np.where(T_arr <= 0.0, 0.0, val)
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def theta_daily(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
        option_type: str = "call",
        days_per_year: float = 365.0,
    ) -> Union[float, np.ndarray]:
        """Daily theta (1 calendar day decay): Theta / 365."""
        return cls.theta(S, K, T, r, sigma, q, option_type) / float(days_per_year)

    @classmethod
    def rho(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
        option_type: str = "call",
    ) -> Union[float, np.ndarray]:
        """Rho (dV/dr): sensitivity of option price to risk-free interest rate.
        Call: K * T * exp(-r*T) * N(d2)
        Put:  -K * T * exp(-r*T) * N(-d2)
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d2_val = BlackScholesModel.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        disc_r = np.exp(-r_arr * T_arr)

        opt_type = option_type.lower()
        if opt_type in ("call", "c"):
            val = K_arr * T_arr * disc_r * norm.cdf(d2_val)
        elif opt_type in ("put", "p"):
            val = -K_arr * T_arr * disc_r * norm.cdf(-d2_val)
        else:
            raise ValueError(f"Invalid option_type: {option_type}.")
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def rho_percentage(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
        option_type: str = "call",
    ) -> Union[float, np.ndarray]:
        """Rho per 1 percentage point (0.01) change in risk-free rate: Rho / 100."""
        return cls.rho(S, K, T, r, sigma, q, option_type) / 100.0

    @classmethod
    def vanna(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Vanna (d^2V / dS dsigma = dDelta/dsigma): cross sensitivity of Delta to Volatility.
        Vanna = -exp(-q*T) * phi(d1) * (d2 / sigma)
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        d2_val = BlackScholesModel.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        disc_q = np.exp(-q_arr * T_arr)

        val = -disc_q * norm.pdf(d1_val) * (d2_val / np.maximum(sigma_arr, 1e-12))
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def volga(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Volga / Vomma (d^2V / dsigma^2 = dVega/dsigma): sensitivity of Vega to Volatility.
        Volga = Vega * d1 * d2 / sigma
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        d2_val = BlackScholesModel.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        vega_val = cls.vega(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)

        val = vega_val * d1_val * d2_val / np.maximum(sigma_arr, 1e-12)
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def charm(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
        option_type: str = "call",
    ) -> Union[float, np.ndarray]:
        """Charm / Delta Decay (-dDelta/dT = dDelta/dt): rate of change of Delta over time.
        Call: q * exp(-q*T) * N(d1) - exp(-q*T) * phi(d1) * [2(r-q)T - d2*sigma*sqrt(T)] / (2*T*sigma*sqrt(T))
        Put:  -q * exp(-q*T) * N(-d1) - exp(-q*T) * phi(d1) * [2(r-q)T - d2*sigma*sqrt(T)] / (2*T*sigma*sqrt(T))
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        d2_val = BlackScholesModel.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)

        disc_q = np.exp(-q_arr * T_arr)
        sqrt_T = np.sqrt(np.maximum(T_arr, 1e-12))
        denom = 2.0 * np.maximum(T_arr, 1e-12) * sigma_arr * sqrt_T
        common_term = disc_q * norm.pdf(d1_val) * (2.0 * (r_arr - q_arr) * T_arr - d2_val * sigma_arr * sqrt_T) / np.maximum(denom, 1e-12)

        opt_type = option_type.lower()
        if opt_type in ("call", "c"):
            val = q_arr * disc_q * norm.cdf(d1_val) - common_term
        elif opt_type in ("put", "p"):
            val = -q_arr * disc_q * norm.cdf(-d1_val) - common_term
        else:
            raise ValueError(f"Invalid option_type: {option_type}.")
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def speed(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Speed (d^3V / dS^3 = dGamma/dS): 3rd order derivative with respect to Spot.
        Speed = -Gamma / S * (d1 / (sigma * sqrt(T)) + 1)
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        gamma_val = cls.gamma(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        sqrt_T = np.sqrt(np.maximum(T_arr, 1e-12))
        denom = sigma_arr * sqrt_T

        val = - (gamma_val / np.maximum(S_arr, 1e-12)) * (d1_val / np.maximum(denom, 1e-12) + 1.0)
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def zomma(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Zomma (dGamma/dsigma): rate of change of Gamma with respect to volatility.
        Zomma = Gamma * (d1 * d2 - 1) / sigma
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        d2_val = BlackScholesModel.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        gamma_val = cls.gamma(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)

        val = gamma_val * (d1_val * d2_val - 1.0) / np.maximum(sigma_arr, 1e-12)
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def color(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Color / Gamma Decay (-dGamma/dT = dGamma/dt): rate of change of Gamma over time.
        Color = -Gamma * [q + (1 - d1*d2)/(2*T) + d1*(r-q)/(sigma*sqrt(T))]
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        d1_val = BlackScholesModel.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        d2_val = BlackScholesModel.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        gamma_val = cls.gamma(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        sqrt_T = np.sqrt(np.maximum(T_arr, 1e-12))

        bracket = q_arr + (1.0 - d1_val * d2_val) / (2.0 * np.maximum(T_arr, 1e-12)) + d1_val * (r_arr - q_arr) / np.maximum(sigma_arr * sqrt_T, 1e-12)
        val = -gamma_val * bracket
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def calculate_all(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> GreeksResult:
        """Calculates all Greeks returning a structured GreeksResult."""
        opt_type = option_type.lower()
        price_val = float(BlackScholesModel.price(S, K, T, r, sigma, q, opt_type))
        d = float(cls.delta(S, K, T, r, sigma, q, opt_type))
        g = float(cls.gamma(S, K, T, r, sigma, q))
        th = float(cls.theta(S, K, T, r, sigma, q, opt_type))
        th_d = float(cls.theta_daily(S, K, T, r, sigma, q, opt_type))
        v = float(cls.vega(S, K, T, r, sigma, q))
        v_pct = float(cls.vega_percentage(S, K, T, r, sigma, q))
        rh = float(cls.rho(S, K, T, r, sigma, q, opt_type))
        rh_pct = float(cls.rho_percentage(S, K, T, r, sigma, q, opt_type))
        
        va = float(cls.vanna(S, K, T, r, sigma, q))
        vo = float(cls.volga(S, K, T, r, sigma, q))
        ch = float(cls.charm(S, K, T, r, sigma, q, opt_type))
        sp = float(cls.speed(S, K, T, r, sigma, q))
        zm = float(cls.zomma(S, K, T, r, sigma, q))
        co = float(cls.color(S, K, T, r, sigma, q))

        return GreeksResult(
            spot=S,
            strike=K,
            expiry=T,
            rate=r,
            volatility=sigma,
            dividend=q,
            option_type=opt_type,
            price=price_val,
            delta=d,
            theta=th,
            theta_daily=th_d,
            vega=v,
            vega_pct=v_pct,
            rho=rh,
            rho_pct=rh_pct,
            gamma=g,
            vanna=va,
            volga=vo,
            charm=ch,
            speed=sp,
            zomma=zm,
            color=co,
        )
