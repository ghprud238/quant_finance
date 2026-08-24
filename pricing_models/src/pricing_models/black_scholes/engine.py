"""Black-Scholes Option Pricing Engine (Merton 1973 Continuous Dividend Extension)."""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class OptionPriceResult:
    """Container for option pricing results and breakdown."""
    price: float
    option_type: str
    spot: float
    strike: float
    expiry: float
    rate: float
    volatility: float
    dividend: float
    d1: float
    d2: float
    intrinsic_value: float
    time_value: float
    call_price: float
    put_price: float
    put_call_parity_diff: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": self.price,
            "option_type": self.option_type,
            "spot": self.spot,
            "strike": self.strike,
            "expiry": self.expiry,
            "rate": self.rate,
            "volatility": self.volatility,
            "dividend": self.dividend,
            "d1": self.d1,
            "d2": self.d2,
            "intrinsic_value": self.intrinsic_value,
            "time_value": self.time_value,
            "call_price": self.call_price,
            "put_price": self.put_price,
            "put_call_parity_diff": self.put_call_parity_diff,
        }


class BlackScholesModel:
    """Black-Scholes-Merton (1973) European option pricing model.

    Supports continuous dividend yield q, scalar and vectorized inputs,
    edge case handling (T <= 0, sigma <= 0), and Put-Call Parity validation.
    """

    @staticmethod
    def d1(
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Calculates d1 parameter in Black-Scholes formula."""
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        # Avoid division by zero for T <= 0 or sigma <= 0
        denom = sigma_arr * np.sqrt(np.maximum(T_arr, 1e-12))
        numer = np.log(np.maximum(S_arr, 1e-12) / np.maximum(K_arr, 1e-12)) + (r_arr - q_arr + 0.5 * sigma_arr**2) * T_arr
        d1_val = np.where(denom > 1e-10, numer / denom, np.where(S_arr >= K_arr, 100.0, -100.0))
        return float(d1_val) if np.ndim(d1_val) == 0 else d1_val

    @staticmethod
    def d2(
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Calculates d2 parameter in Black-Scholes formula (d2 = d1 - sigma * sqrt(T))."""
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma, q)
        T_arr = np.asarray(T, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        d2_val = d1_val - sigma_arr * np.sqrt(np.maximum(T_arr, 0.0))
        return float(d2_val) if np.ndim(d2_val) == 0 else d2_val

    @classmethod
    def call_price(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Calculates European Call price under Black-Scholes-Merton model:
        C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        # Handle expiration T <= 0
        intrinsic = np.maximum(S_arr - K_arr, 0.0)

        # Handle zero volatility
        forward_val = np.maximum(S_arr * np.exp(-q_arr * T_arr) - K_arr * np.exp(-r_arr * T_arr), 0.0)

        d1_val = cls.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        d2_val = cls.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)

        bs_call = S_arr * np.exp(-q_arr * T_arr) * norm.cdf(d1_val) - K_arr * np.exp(-r_arr * T_arr) * norm.cdf(d2_val)
        
        # Apply boundary conditions
        call_val = np.where(T_arr <= 0.0, intrinsic, np.where(sigma_arr <= 1e-8, forward_val, bs_call))
        call_val = np.maximum(call_val, 0.0)
        return float(call_val) if np.ndim(call_val) == 0 else call_val

    @classmethod
    def put_price(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Calculates European Put price under Black-Scholes-Merton model:
        P = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)
        """
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        # Handle expiration T <= 0
        intrinsic = np.maximum(K_arr - S_arr, 0.0)

        # Handle zero volatility
        forward_val = np.maximum(K_arr * np.exp(-r_arr * T_arr) - S_arr * np.exp(-q_arr * T_arr), 0.0)

        d1_val = cls.d1(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
        d2_val = cls.d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)

        bs_put = K_arr * np.exp(-r_arr * T_arr) * norm.cdf(-d2_val) - S_arr * np.exp(-q_arr * T_arr) * norm.cdf(-d1_val)

        # Apply boundary conditions
        put_val = np.where(T_arr <= 0.0, intrinsic, np.where(sigma_arr <= 1e-8, forward_val, bs_put))
        put_val = np.maximum(put_val, 0.0)
        return float(put_val) if np.ndim(put_val) == 0 else put_val

    @classmethod
    def price(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        q: Union[float, np.ndarray] = 0.0,
        option_type: str = "call",
    ) -> Union[float, np.ndarray]:
        """Prices an option (call or put) given parameters."""
        opt_type = option_type.lower()
        if opt_type in ("call", "c"):
            return cls.call_price(S, K, T, r, sigma, q)
        elif opt_type in ("put", "p"):
            return cls.put_price(S, K, T, r, sigma, q)
        else:
            raise ValueError(f"Invalid option_type: {option_type}. Expected 'call' or 'put'.")

    @classmethod
    def intrinsic_value(
        cls,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        option_type: str = "call",
    ) -> Union[float, np.ndarray]:
        """Computes intrinsic value: max(S-K, 0) for call, max(K-S, 0) for put."""
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        opt_type = option_type.lower()
        if opt_type in ("call", "c"):
            val = np.maximum(S_arr - K_arr, 0.0)
        elif opt_type in ("put", "p"):
            val = np.maximum(K_arr - S_arr, 0.0)
        else:
            raise ValueError(f"Invalid option_type: {option_type}.")
        return float(val) if np.ndim(val) == 0 else val

    @classmethod
    def time_value(
        cls,
        price: Union[float, np.ndarray],
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        option_type: str = "call",
    ) -> Union[float, np.ndarray]:
        """Computes time value: Option Price - Intrinsic Value."""
        intrinsic = cls.intrinsic_value(S, K, option_type)
        tv = np.asarray(price, dtype=float) - np.asarray(intrinsic, dtype=float)
        return float(tv) if np.ndim(tv) == 0 else tv

    @classmethod
    def verify_put_call_parity(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        call_price: Optional[float] = None,
        put_price: Optional[float] = None,
    ) -> Dict[str, float]:
        """Verifies Put-Call Parity: C - P = S * exp(-q*T) - K * exp(-r*T)."""
        c = cls.call_price(S, K, T, r, sigma, q) if call_price is None else call_price
        p = cls.put_price(S, K, T, r, sigma, q) if put_price is None else put_price
        lhs = c - p
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        diff = lhs - rhs
        return {
            "call_price": c,
            "put_price": p,
            "lhs_c_minus_p": lhs,
            "rhs_synthetic_forward": rhs,
            "parity_difference": diff,
            "abs_error": abs(diff),
            "is_parity_valid": bool(abs(diff) < 1e-7),
        }

    @classmethod
    def calculate(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> OptionPriceResult:
        """Evaluates full pricing breakdown returning a structured OptionPriceResult."""
        c_price = float(cls.call_price(S, K, T, r, sigma, q))
        p_price = float(cls.put_price(S, K, T, r, sigma, q))
        d1_val = float(cls.d1(S, K, T, r, sigma, q))
        d2_val = float(cls.d2(S, K, T, r, sigma, q))
        
        opt_type = option_type.lower()
        selected_price = c_price if opt_type in ("call", "c") else p_price
        intrinsic = float(cls.intrinsic_value(S, K, opt_type))
        tv = selected_price - intrinsic
        
        parity_res = cls.verify_put_call_parity(S, K, T, r, sigma, q, c_price, p_price)

        return OptionPriceResult(
            price=selected_price,
            option_type=opt_type,
            spot=S,
            strike=K,
            expiry=T,
            rate=r,
            volatility=sigma,
            dividend=q,
            d1=d1_val,
            d2=d2_val,
            intrinsic_value=intrinsic,
            time_value=tv,
            call_price=c_price,
            put_price=p_price,
            put_call_parity_diff=parity_res["parity_difference"],
        )


class OptionChainPricer:
    """Generates comprehensive European option chains across a range of strikes."""

    @staticmethod
    def generate_chain(
        S: float,
        strikes: Union[List[float], np.ndarray],
        T: float,
        r: float,
        sigma: Union[float, List[float], np.ndarray],
        q: float = 0.0,
    ) -> pd.DataFrame:
        """Generates formatted option chain table across multiple strikes."""
        from ..greeks.analytical import AnalyticalGreeks

        strikes_arr = np.asarray(strikes, dtype=float)
        n_strikes = len(strikes_arr)

        if isinstance(sigma, (int, float)):
            sigma_arr = np.full(n_strikes, float(sigma))
        else:
            sigma_arr = np.asarray(sigma, dtype=float)
            if len(sigma_arr) != n_strikes:
                raise ValueError("Length of sigma array must match length of strikes array.")

        # Calls
        call_prices = BlackScholesModel.call_price(S, strikes_arr, T, r, sigma_arr, q)
        call_deltas = AnalyticalGreeks.delta(S, strikes_arr, T, r, sigma_arr, q, "call")
        call_gammas = AnalyticalGreeks.gamma(S, strikes_arr, T, r, sigma_arr, q)
        call_thetas = AnalyticalGreeks.theta_daily(S, strikes_arr, T, r, sigma_arr, q, "call")
        call_vegas = AnalyticalGreeks.vega_percentage(S, strikes_arr, T, r, sigma_arr, q)

        # Puts
        put_prices = BlackScholesModel.put_price(S, strikes_arr, T, r, sigma_arr, q)
        put_deltas = AnalyticalGreeks.delta(S, strikes_arr, T, r, sigma_arr, q, "put")
        put_gammas = AnalyticalGreeks.gamma(S, strikes_arr, T, r, sigma_arr, q)
        put_thetas = AnalyticalGreeks.theta_daily(S, strikes_arr, T, r, sigma_arr, q, "put")
        put_vegas = AnalyticalGreeks.vega_percentage(S, strikes_arr, T, r, sigma_arr, q)

        # Moneyness
        moneyness = strikes_arr / S

        df = pd.DataFrame({
            "Call_Delta": call_deltas,
            "Call_Gamma": call_gammas,
            "Call_Theta": call_thetas,
            "Call_Vega": call_vegas,
            "Call_Price": call_prices,
            "Strike": strikes_arr,
            "Moneyness": moneyness,
            "Put_Price": put_prices,
            "Put_Delta": put_deltas,
            "Put_Gamma": put_gammas,
            "Put_Theta": put_thetas,
            "Put_Vega": put_vegas,
            "IV": sigma_arr,
        })
        return df
