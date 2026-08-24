"""Numerical Finite-Difference Option Greeks Calculator."""

from typing import Callable, Dict, Any, Optional
import numpy as np

from ..black_scholes.engine import BlackScholesModel


class NumericalGreeks:
    """Calculates option Greeks using numerical finite-difference schemes.

    Useful for model-agnostic Greeks verification and black-box valuation engines.
    """

    @staticmethod
    def _default_pricer(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call"
    ) -> float:
        return float(BlackScholesModel.price(S, K, T, r, sigma, q, option_type))

    @classmethod
    def delta(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        dS: float = 1e-4,
        pricer: Optional[Callable] = None,
    ) -> float:
        """Central finite difference for Delta: (V(S+dS) - V(S-dS)) / (2*dS)."""
        p_func = pricer or cls._default_pricer
        h = S * dS if S > 0 else dS
        v_up = p_func(S + h, K, T, r, sigma, q, option_type)
        v_dn = p_func(S - h, K, T, r, sigma, q, option_type)
        return (v_up - v_dn) / (2.0 * h)

    @classmethod
    def gamma(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        dS: float = 1e-4,
        pricer: Optional[Callable] = None,
    ) -> float:
        """Central finite difference for Gamma: (V(S+dS) - 2*V(S) + V(S-dS)) / (dS^2)."""
        p_func = pricer or cls._default_pricer
        h = S * dS if S > 0 else dS
        v_up = p_func(S + h, K, T, r, sigma, q, option_type)
        v_mid = p_func(S, K, T, r, sigma, q, option_type)
        v_dn = p_func(S - h, K, T, r, sigma, q, option_type)
        return (v_up - 2.0 * v_mid + v_dn) / (h ** 2)

    @classmethod
    def vega(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        dSigma: float = 1e-4,
        pricer: Optional[Callable] = None,
    ) -> float:
        """Central finite difference for Vega: (V(sigma+dSigma) - V(sigma-dSigma)) / (2*dSigma)."""
        p_func = pricer or cls._default_pricer
        h = dSigma
        v_up = p_func(S, K, T, r, sigma + h, q, option_type)
        v_dn = p_func(S, K, T, r, sigma - h, q, option_type)
        return (v_up - v_dn) / (2.0 * h)

    @classmethod
    def theta(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        dT: float = 1e-5,
        pricer: Optional[Callable] = None,
    ) -> float:
        """Finite difference for Theta: - (V(T+dT) - V(T-dT)) / (2*dT)."""
        p_func = pricer or cls._default_pricer
        h = min(dT, T * 0.1) if T > 0 else dT
        v_up = p_func(S, K, T + h, r, sigma, q, option_type)
        v_dn = p_func(S, K, max(T - h, 1e-8), r, sigma, q, option_type)
        return - (v_up - v_dn) / (2.0 * h)

    @classmethod
    def rho(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        dr: float = 1e-4,
        pricer: Optional[Callable] = None,
    ) -> float:
        """Central finite difference for Rho: (V(r+dr) - V(r-dr)) / (2*dr)."""
        p_func = pricer or cls._default_pricer
        h = dr
        v_up = p_func(S, K, T, r + h, sigma, q, option_type)
        v_dn = p_func(S, K, T, r - h, sigma, q, option_type)
        return (v_up - v_dn) / (2.0 * h)

    @classmethod
    def vanna(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        dS: float = 1e-4,
        dSigma: float = 1e-4,
        pricer: Optional[Callable] = None,
    ) -> float:
        """Numerical cross derivative for Vanna (d^2V / dS dsigma)."""
        p_func = pricer or cls._default_pricer
        h_s = S * dS if S > 0 else dS
        h_sig = dSigma
        
        v_pp = p_func(S + h_s, K, T, r, sigma + h_sig, q, option_type)
        v_pm = p_func(S + h_s, K, T, r, sigma - h_sig, q, option_type)
        v_mp = p_func(S - h_s, K, T, r, sigma + h_sig, q, option_type)
        v_mm = p_func(S - h_s, K, T, r, sigma - h_sig, q, option_type)
        return (v_pp - v_pm - v_mp + v_mm) / (4.0 * h_s * h_sig)

    @classmethod
    def volga(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        dSigma: float = 1e-4,
        pricer: Optional[Callable] = None,
    ) -> float:
        """Numerical 2nd derivative for Volga / Vomma (d^2V / dsigma^2)."""
        p_func = pricer or cls._default_pricer
        h = dSigma
        v_up = p_func(S, K, T, r, sigma + h, q, option_type)
        v_mid = p_func(S, K, T, r, sigma, q, option_type)
        v_dn = p_func(S, K, T, r, sigma - h, q, option_type)
        return (v_up - 2.0 * v_mid + v_dn) / (h ** 2)

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
    ) -> Dict[str, float]:
        """Calculates all numerical Greeks in a single dictionary."""
        return {
            "delta": cls.delta(S, K, T, r, sigma, q, option_type),
            "gamma": cls.gamma(S, K, T, r, sigma, q, option_type),
            "vega": cls.vega(S, K, T, r, sigma, q, option_type),
            "theta": cls.theta(S, K, T, r, sigma, q, option_type),
            "rho": cls.rho(S, K, T, r, sigma, q, option_type),
            "vanna": cls.vanna(S, K, T, r, sigma, q, option_type),
            "volga": cls.volga(S, K, T, r, sigma, q, option_type),
        }
