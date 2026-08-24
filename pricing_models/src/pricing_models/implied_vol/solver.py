"""Implied Volatility Solver.

Inverts the Black-Scholes-Merton option pricing formula using:
- Newton-Raphson method with analytical Vega
- Brent's method / Bisection fallback for guaranteed convergence
- Brenner-Subrahmanyam & Corrado-Miller analytic initial approximations
- Arbitrage boundary enforcement and vectorized option chain processing
"""

from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_price(
    spot: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray] = 0.0,
    dividend_yield: Union[float, np.ndarray] = 0.0,
    volatility: Union[float, np.ndarray] = 0.20,
    option_type: str = "call",
) -> Union[float, np.ndarray]:
    """Calculates analytical Black-Scholes-Merton European option price.

    Args:
        spot: Underlying asset price S_0
        strike: Strike price K
        time_to_expiry: Time to maturity in years T
        risk_free_rate: Continuously compounded risk-free rate r
        dividend_yield: Continuous dividend yield q
        volatility: Annualized volatility sigma
        option_type: 'call' (default) or 'put'

    Returns:
        European option theoretical price
    """
    s = np.asarray(spot, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(time_to_expiry, dtype=float)
    r = np.asarray(risk_free_rate, dtype=float)
    q = np.asarray(dividend_yield, dtype=float)
    v = np.asarray(volatility, dtype=float)

    # Edge cases: zero or negative time to expiry
    if np.any(t <= 0):
        intrinsic_call = np.maximum(s * np.exp(-q * t) - k * np.exp(-r * t), 0.0)
        intrinsic_put = np.maximum(k * np.exp(-r * t) - s * np.exp(-q * t), 0.0)
        if option_type.lower() in ("call", "c"):
            return float(intrinsic_call) if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(time_to_expiry) else intrinsic_call
        else:
            return float(intrinsic_put) if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(time_to_expiry) else intrinsic_put

    # Edge cases: zero or negative volatility
    if np.any(v <= 0):
        intrinsic_call = np.maximum(s * np.exp(-q * t) - k * np.exp(-r * t), 0.0)
        intrinsic_put = np.maximum(k * np.exp(-r * t) - s * np.exp(-q * t), 0.0)
        if option_type.lower() in ("call", "c"):
            return float(intrinsic_call) if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(time_to_expiry) else intrinsic_call
        else:
            return float(intrinsic_put) if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(time_to_expiry) else intrinsic_put

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v ** 2) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t

    df_r = np.exp(-r * t)
    df_q = np.exp(-q * t)

    is_call = option_type.lower() in ("call", "c")
    if is_call:
        price = s * df_q * norm.cdf(d1) - k * df_r * norm.cdf(d2)
    else:
        price = k * df_r * norm.cdf(-d2) - s * df_q * norm.cdf(-d1)

    if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(time_to_expiry) and np.isscalar(volatility):
        return float(price)
    return price


def black_scholes_vega(
    spot: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray] = 0.0,
    dividend_yield: Union[float, np.ndarray] = 0.0,
    volatility: Union[float, np.ndarray] = 0.20,
) -> Union[float, np.ndarray]:
    """Calculates Black-Scholes Vega (dPrice / dSigma).

    Vega is identical for European Calls and Puts.
    """
    s = np.asarray(spot, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(time_to_expiry, dtype=float)
    r = np.asarray(risk_free_rate, dtype=float)
    q = np.asarray(dividend_yield, dtype=float)
    v = np.asarray(volatility, dtype=float)

    if np.any(t <= 0) or np.any(v <= 0):
        return 0.0 if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(time_to_expiry) else np.zeros_like(s)

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v ** 2) * t) / (v * sqrt_t)
    df_q = np.exp(-q * t)

    vega = s * df_q * sqrt_t * norm.pdf(d1)
    if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(time_to_expiry) and np.isscalar(volatility):
        return float(vega)
    return vega


def brenner_subrahmanyam_iv(
    market_price: float,
    spot: float,
    time_to_expiry: float,
) -> float:
    """Brenner-Subrahmanyam (1988) analytical ATM initial approximation for IV.

    sigma_0 approx sqrt(2 * pi / T) * (C_ATM / S_0)
    """
    if time_to_expiry <= 0 or spot <= 0 or market_price <= 0:
        return 0.20
    approx = np.sqrt(2.0 * np.pi / time_to_expiry) * (market_price / spot)
    return float(np.clip(approx, 0.01, 3.0))


def corrado_miller_iv(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> float:
    """Corrado-Miller (1996) analytical closed-form initial approximation for non-ATM IV.

    Extends Brenner-Subrahmanyam to out-of-the-money and in-the-money European options.
    """
    if time_to_expiry <= 0 or spot <= 0 or strike <= 0 or market_price <= 0:
        return 0.20

    df_r = np.exp(-risk_free_rate * time_to_expiry)
    df_q = np.exp(-dividend_yield * time_to_expiry)
    s_adj = spot * df_q
    k_adj = strike * df_r

    is_call = option_type.lower() in ("call", "c")
    # If put, convert to synthetic call price via Put-Call parity
    if not is_call:
        call_price = market_price + s_adj - k_adj
    else:
        call_price = market_price

    diff = (s_adj - k_adj) / 2.0
    denom = (s_adj + k_adj) * np.sqrt(time_to_expiry)
    if denom <= 0:
        return 0.20

    term1 = call_price - diff
    inner = term1 ** 2 - (s_adj - k_adj) ** 2 / np.pi
    if inner < 0:
        return float(brenner_subrahmanyam_iv(market_price, spot, time_to_expiry))

    sigma = (np.sqrt(2.0 * np.pi) / denom) * (term1 + np.sqrt(inner))
    return float(np.clip(sigma, 0.01, 5.0))


class ImpliedVolatilitySolver:
    """High-performance, robust European option implied volatility solver.

    Supports:
    - Newton-Raphson root-finding with analytical Vega
    - Brent's method with guaranteed superlinear convergence
    - Brenner-Subrahmanyam and Corrado-Miller initial guesses
    - Arbitrage bounds validation
    - Vectorized / DataFrame option chain solving
    """

    def __init__(
        self,
        default_method: str = "auto",
        tolerance: float = 1e-8,
        max_iterations: int = 100,
        vol_lower_bound: float = 1e-4,
        vol_upper_bound: float = 10.0,
    ):
        """Initializes the IV Solver.

        Args:
            default_method: 'auto', 'newton', or 'brent'
            tolerance: Convergence tolerance for absolute price error |BS(sigma) - Price|
            max_iterations: Maximum iterations for Newton-Raphson
            vol_lower_bound: Lower volatility bracket bound (0.01%)
            vol_upper_bound: Upper volatility bracket bound (1000%)
        """
        self.default_method = default_method.lower()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.vol_lower_bound = vol_lower_bound
        self.vol_upper_bound = vol_upper_bound

    def check_arbitrage_bounds(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        option_type: str = "call",
    ) -> Tuple[bool, float, float]:
        """Validates whether the market price falls within no-arbitrage theoretical bounds.

        Returns:
            (is_valid, lower_bound, upper_bound)
        """
        df_r = np.exp(-risk_free_rate * time_to_expiry)
        df_q = np.exp(-dividend_yield * time_to_expiry)
        is_call = option_type.lower() in ("call", "c")

        if is_call:
            lower_bound = max(spot * df_q - strike * df_r, 0.0)
            upper_bound = spot * df_q
        else:
            lower_bound = max(strike * df_r - spot * df_q, 0.0)
            upper_bound = strike * df_r

        # Price must be strictly between lower_bound - eps and upper_bound + eps
        eps = 1e-9
        is_valid = (market_price >= lower_bound - eps) and (market_price <= upper_bound + eps)
        return is_valid, lower_bound, upper_bound

    def solve(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        option_type: str = "call",
        method: Optional[str] = None,
        initial_guess: Optional[float] = None,
        raise_on_arbitrage: bool = False,
    ) -> float:
        """Solves for implied volatility sigma such that BS(sigma) = market_price.

        Args:
            market_price: Observed market price of the option
            spot: Spot price of the underlying
            strike: Strike price
            time_to_expiry: Time to maturity in years (T)
            risk_free_rate: Risk-free rate (annualized)
            dividend_yield: Continuous dividend yield
            option_type: 'call' or 'put'
            method: 'auto', 'newton', or 'brent' (defaults to solver setting)
            initial_guess: Optional starting sigma
            raise_on_arbitrage: If True, raises ValueError on arbitrage violation, else returns np.nan

        Returns:
            Implied volatility sigma (e.g. 0.20 = 20%)
        """
        if time_to_expiry <= 0:
            return 0.0
        if market_price <= 0:
            return 0.0

        is_valid, lower_b, upper_b = self.check_arbitrage_bounds(
            market_price=market_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )

        if not is_valid:
            if raise_on_arbitrage:
                raise ValueError(
                    f"Market price {market_price:.4f} violates theoretical arbitrage bounds "
                    f"[{lower_b:.4f}, {upper_b:.4f}] for {option_type} K={strike}, T={time_to_expiry:.2f}"
                )
            return np.nan

        # If market price is essentially at intrinsic value, IV is 0
        if abs(market_price - lower_b) < 1e-8:
            return 0.0

        method = (method or self.default_method).lower()

        # Compute starting initial guess if not provided
        if initial_guess is None:
            initial_guess = corrado_miller_iv(
                market_price=market_price,
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                option_type=option_type,
            )

        if method in ("newton", "auto"):
            sigma = self._newton_raphson(
                market_price=market_price,
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                option_type=option_type,
                initial_guess=initial_guess,
            )
            if not np.isnan(sigma) and sigma > 0:
                return float(sigma)
            if method == "newton":
                return np.nan

        # Fallback to Brent's method
        return float(self._brent(
            market_price=market_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        ))

    def _newton_raphson(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float,
        option_type: str,
        initial_guess: float,
    ) -> float:
        """Newton-Raphson iteration: sigma_{n+1} = sigma_n - (BS(sigma_n) - C) / Vega(sigma_n)."""
        sigma = float(np.clip(initial_guess, self.vol_lower_bound, self.vol_upper_bound))

        for _ in range(self.max_iterations):
            price = black_scholes_price(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                volatility=sigma,
                option_type=option_type,
            )
            diff = price - market_price

            if abs(diff) < self.tolerance:
                return sigma

            vega = black_scholes_vega(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                volatility=sigma,
            )

            # If Vega is too small, Newton-Raphson gradient vanishes -> abort to Brent
            if vega < 1e-10:
                return np.nan

            step = diff / vega
            sigma_next = sigma - step

            # Damped update if stepping out of bounds
            if sigma_next <= self.vol_lower_bound or sigma_next >= self.vol_upper_bound:
                sigma_next = 0.5 * (sigma + (self.vol_lower_bound if sigma_next <= self.vol_lower_bound else self.vol_upper_bound))

            if abs(sigma_next - sigma) < self.tolerance * 0.1:
                return sigma_next

            sigma = sigma_next

        return np.nan

    def _brent(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float,
        option_type: str,
    ) -> float:
        """Brent's root-finding method on f(sigma) = BS(sigma) - market_price = 0."""
        def objective(v: float) -> float:
            return black_scholes_price(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                volatility=v,
                option_type=option_type,
            ) - market_price

        f_low = objective(self.vol_lower_bound)
        f_high = objective(self.vol_upper_bound)

        if f_low * f_high > 0:
            # Check if increasing upper bound resolves bracket
            f_high_ext = objective(20.0)
            if f_low * f_high_ext <= 0:
                return brentq(objective, self.vol_lower_bound, 20.0, xtol=self.tolerance, maxiter=200)
            return np.nan

        try:
            return brentq(objective, self.vol_lower_bound, self.vol_upper_bound, xtol=self.tolerance, maxiter=200)
        except Exception:
            return np.nan

    def solve_chain(
        self,
        df_options: pd.DataFrame,
        spot: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        price_col: str = "price",
        strike_col: str = "strike",
        expiry_col: str = "expiry",
        type_col: str = "type",
        method: Optional[str] = None,
    ) -> pd.DataFrame:
        """Vectorized / batch solving of implied volatility for an entire option chain.

        Args:
            df_options: DataFrame containing option chain records
            spot: Current underlying asset price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            price_col: Column name for market price
            strike_col: Column name for strike K
            expiry_col: Column name for time to expiry in years T
            type_col: Column name for option type ('call'/'put')
            method: 'auto', 'newton', or 'brent'

        Returns:
            DataFrame augmented with 'implied_vol', 'moneyness' (K/S), 'log_moneyness' (ln(K/F)), and 'vega'.
        """
        df = df_options.copy()
        ivs = []
        vegas = []
        moneyness = []
        log_moneyness = []

        for _, row in df.iterrows():
            mkt_p = float(row[price_col])
            k = float(row[strike_col])
            t = float(row[expiry_col])
            op_type = str(row[type_col]) if type_col in row else "call"

            iv = self.solve(
                market_price=mkt_p,
                spot=spot,
                strike=k,
                time_to_expiry=t,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                option_type=op_type,
                method=method,
            )
            ivs.append(iv)

            if not np.isnan(iv) and iv > 0:
                vg = black_scholes_vega(
                    spot=spot,
                    strike=k,
                    time_to_expiry=t,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                    volatility=iv,
                )
            else:
                vg = np.nan
            vegas.append(vg)

            forward = spot * np.exp((risk_free_rate - dividend_yield) * t)
            moneyness.append(k / spot)
            log_moneyness.append(np.log(k / forward))

        df["implied_vol"] = ivs
        df["vega"] = vegas
        df["moneyness"] = moneyness
        df["log_moneyness"] = log_moneyness
        return df
