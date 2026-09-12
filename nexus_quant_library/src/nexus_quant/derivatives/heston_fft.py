"""
Heston (1993) Stochastic Volatility Model, Carr-Madan FFT & Fang-Oosterlee COS Method.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.optimize import minimize


@dataclass
class HestonParameters:
    """Parameters for the Heston (1993) Stochastic Volatility Model."""
    v0: float      # Initial variance
    kappa: float   # Mean-reversion speed
    theta: float   # Long-term variance
    xi: float      # Volatility of variance (vol of vol)
    rho: float     # Correlation between asset and variance Brownian motions
    r: float = 0.0 # Risk-free rate
    q: float = 0.0 # Dividend yield

    @property
    def feller_ratio(self) -> float:
        """Feller condition ratio: 2*kappa*theta / xi^2 > 1 guarantees strictly positive variance."""
        return (2.0 * self.kappa * self.theta) / max(self.xi**2, 1e-8)

    @property
    def is_feller_satisfied(self) -> bool:
        """Check if Feller condition holds."""
        return self.feller_ratio > 1.0


class HestonCharacteristicFunction:
    """
    Evaluates the stable Heston characteristic function phi(u; T).
    Uses the formulation of Albrecher et al. (2007) to prevent branch cut discontinuities.
    """

    @staticmethod
    def evaluate(
        u: Union[float, np.ndarray, complex],
        T: float,
        spot: float,
        params: HestonParameters,
    ) -> Union[complex, np.ndarray]:
        """Compute the risk-neutral characteristic function phi(u; T)."""
        v0 = params.v0
        kappa = params.kappa
        theta = params.theta
        xi = params.xi
        rho = params.rho
        r = params.r
        q = params.q

        s0 = spot
        i = 1j

        # Albrecher et al. (2007) stable formulation
        d = np.sqrt((kappa - i * rho * xi * u)**2 + xi**2 * (u**2 + i * u))
        g = (kappa - i * rho * xi * u - d) / (kappa - i * rho * xi * u + d)

        exp_neg_dt = np.exp(-d * T)
        one_minus_g_exp = 1.0 - g * exp_neg_dt
        one_minus_g = 1.0 - g

        c_term = (r - q) * i * u * T + (kappa * theta / (xi**2)) * (
            (kappa - i * rho * xi * u - d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g)
        )
        d_term = ((kappa - i * rho * xi * u - d) / (xi**2)) * ((1.0 - exp_neg_dt) / one_minus_g_exp)

        return np.exp(c_term + d_term * v0 + i * u * np.log(s0))


class CarrMadanFFTPricer:
    """
    Carr-Madan (1999) Fast Fourier Transform (FFT) European Option Pricing.
    Evaluates call prices across an entire grid of log-strikes in O(N log N) time.
    """

    @classmethod
    def price_call_strip(
        cls,
        spot: float,
        strikes: np.ndarray,
        time_to_expiry: float,
        params: HestonParameters,
        alpha: float = 1.5,
        n_fft: int = 4096,
        eta: float = 0.25,
    ) -> np.ndarray:
        """Compute call prices across input strikes using Carr-Madan FFT."""
        t = time_to_expiry
        r = params.r

        # FFT grid discretization
        n = n_fft
        lambda_grid = (2.0 * np.pi) / (n * eta)
        b = (n * lambda_grid) / 2.0

        v_j = np.arange(n) * eta
        k_u = -b + np.arange(n) * lambda_grid  # log-strike grid

        # Simpson's rule integration weights
        weights = np.ones(n)
        weights[0] = 1.0 / 3.0
        weights[1:-1:2] = 4.0 / 3.0
        weights[2:-1:2] = 2.0 / 3.0
        weights[-1] = 1.0 / 3.0

        # Modified characteristic function
        u_shifted = v_j - (alpha + 1.0) * 1j
        phi_vals = HestonCharacteristicFunction.evaluate(u_shifted, t, spot, params)
        psi_vals = (np.exp(-r * t) * phi_vals) / (alpha**2 + alpha - v_j**2 + 1j * (2.0 * alpha + 1.0) * v_j)

        # FFT input vector
        fft_input = np.exp(1j * b * v_j) * psi_vals * eta * weights
        fft_output = np.real(fft(fft_input))

        # Damped call prices on log-strike grid
        call_grid = (np.exp(-alpha * k_u) / np.pi) * fft_output
        strike_grid = np.exp(k_u)

        # Interpolate prices onto requested user strikes
        prices = np.interp(strikes, strike_grid, call_grid)
        return np.maximum(prices, 0.0)


class FangOosterleeCOSPricer:
    """
    Fang-Oosterlee (2008) Fourier-Cosine (COS) Series Expansion for Option Pricing.
    Sub-millisecond pricing of European options and analytical Greeks.
    """

    @classmethod
    def price(
        cls,
        spot: float,
        strike: float,
        time_to_expiry: float,
        params: HestonParameters,
        option_type: str = "call",
        n_cos: int = 128,
        l_bounds: float = 10.0,
    ) -> float:
        """Price European Call/Put using the COS method."""
        s0 = spot
        k = strike
        t = time_to_expiry
        r = params.r
        q = params.q
        opt_type = option_type.lower()

        # Cumulant approximations for truncated integration interval [a, b]
        c1 = (r - q) * t + (1.0 - np.exp(-params.kappa * t)) * (params.theta - params.v0) / (2.0 * params.kappa) - 0.5 * params.theta * t
        c2 = (params.v0 / (8.0 * params.kappa**3)) * (
            params.xi * t * params.kappa * np.exp(-params.kappa * t) * (params.kappa - params.xi * params.rho)
            + params.xi**2 * (1.0 - np.exp(-2.0 * params.kappa * t))
        ) + (params.theta * t / (8.0 * params.kappa**2)) * (
            2.0 * params.xi * params.kappa * params.rho - params.xi**2
        )
        c2 = max(abs(c2), 0.01)

        x = np.log(s0 / k)
        a = c1 - l_bounds * np.sqrt(c2)
        b = c1 + l_bounds * np.sqrt(c2)

        k_idx = np.arange(n_cos)
        u_k = k_idx * np.pi / (b - a)

        # Characteristic function evaluation
        phi_k = HestonCharacteristicFunction.evaluate(u_k, t, s0, params) * np.exp(-1j * u_k * np.log(s0))

        # Chi and Psi coefficients for standard European payoffs
        def chi_k_fn(c_val: float, d_val: float) -> np.ndarray:
            term1 = np.cos(u_k * (d_val - a)) * np.exp(d_val) - np.cos(u_k * (c_val - a)) * np.exp(c_val)
            term2 = u_k * np.sin(u_k * (d_val - a)) * np.exp(d_val) - u_k * np.sin(u_k * (c_val - a)) * np.exp(c_val)
            return (term1 + term2) / (1.0 + u_k**2)

        def psi_k_fn(c_val: float, d_val: float) -> np.ndarray:
            res = np.zeros(n_cos)
            res[0] = d_val - c_val
            res[1:] = (np.sin(u_k[1:] * (d_val - a)) - np.sin(u_k[1:] * (c_val - a))) / u_k[1:]
            return res

        if opt_type == "call":
            h_k = (2.0 / (b - a)) * (chi_k_fn(0.0, b) - psi_k_fn(0.0, b))
        else:
            h_k = (2.0 / (b - a)) * (-chi_k_fn(a, 0.0) + psi_k_fn(a, 0.0))

        terms = np.real(phi_k * np.exp(1j * u_k * (x - a))) * h_k
        terms[0] *= 0.5
        price_val = k * np.exp(-r * t) * np.sum(terms)
        return float(max(0.0, price_val))


class HestonCalibrator:
    """Calibrates Heston model parameters to market option price / volatility surfaces."""

    @classmethod
    def calibrate(
        cls,
        market_quotes: pd.DataFrame,
        spot: float,
        risk_free_rate: float = 0.04,
        dividend_yield: float = 0.0,
        initial_guess: Optional[List[float]] = None,
    ) -> Tuple[HestonParameters, float]:
        """
        Calibrate (v0, kappa, theta, xi, rho) from option chain DataFrame.
        DataFrame must contain columns: ['strike', 'time_to_expiry', 'market_price'].
        """
        # Default starting parameters
        if initial_guess is None:
            # v0, kappa, theta, xi, rho
            init_params = [0.04, 2.0, 0.04, 0.30, -0.60]
        else:
            init_params = initial_guess

        bounds = [
            (0.001, 1.50),   # v0
            (0.10, 10.0),    # kappa
            (0.001, 1.50),   # theta
            (0.01, 2.0),     # xi
            (-0.99, 0.99),   # rho
        ]

        strikes = market_quotes["strike"].values
        expiries = market_quotes["time_to_expiry"].values
        mkt_prices = market_quotes["market_price"].values

        def objective(p: List[float]) -> float:
            v0, kappa, theta, xi, rho = p
            h_params = HestonParameters(
                v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, r=risk_free_rate, q=dividend_yield
            )
            model_prices = []
            for k, t in zip(strikes, expiries):
                price_val = FangOosterleeCOSPricer.price(spot, k, t, h_params, option_type="call", n_cos=64)
                model_prices.append(price_val)
            model_prices = np.array(model_prices)
            # Feller regularization penalty
            feller_pen = 100.0 * max(0.0, (xi**2) - (2.0 * kappa * theta))
            rmse = np.sqrt(np.mean((model_prices - mkt_prices)**2)) + feller_pen
            return float(rmse)

        res = minimize(objective, init_params, method="L-BFGS-B", bounds=bounds)
        v0, kappa, theta, xi, rho = res.x
        opt_params = HestonParameters(
            v0=float(v0),
            kappa=float(kappa),
            theta=float(theta),
            xi=float(xi),
            rho=float(rho),
            r=risk_free_rate,
            q=dividend_yield,
        )
        return opt_params, float(res.fun)
