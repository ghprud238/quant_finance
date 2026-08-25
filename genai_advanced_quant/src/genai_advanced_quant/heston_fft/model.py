"""Heston (1993) Stochastic Volatility & Carr-Madan (1999) FFT Option Calibration (Project 32).

Includes Fang-Oosterlee (2008) COS method and Albrecher et al. (2007) stable formulation.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import cmath
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.stats import norm


@dataclass
class HestonParameters:
    """Parameters for the Heston (1993) Stochastic Volatility Model.
    
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa * (theta - v_t) dt + xi * sqrt(v_t) dW_t^v
    corr(dW_t^S, dW_t^v) = rho
    """
    v0: float       # Initial variance (e.g. 0.04 -> 20% vol)
    kappa: float    # Mean-reversion speed (e.g. 2.0)
    theta: float    # Long-term variance (e.g. 0.04 -> 20% vol)
    xi: float       # Volatility of variance (e.g. 0.30)
    rho: float      # Correlation between price and volatility shocks (e.g. -0.70)
    r: float = 0.05 # Risk-free rate
    q: float = 0.00 # Continuous dividend yield
    
    @property
    def feller_ratio(self) -> float:
        """Feller condition ratio: 2 * kappa * theta / xi^2. Must be > 1 to guarantee strictly positive variance."""
        return (2.0 * self.kappa * self.theta) / (self.xi ** 2) if self.xi > 0 else float('inf')
        
    @property
    def is_feller_satisfied(self) -> bool:
        return self.feller_ratio > 1.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "v0": self.v0,
            "initial_vol": float(np.sqrt(self.v0)),
            "kappa": self.kappa,
            "theta": self.theta,
            "long_term_vol": float(np.sqrt(self.theta)),
            "xi": self.xi,
            "rho": self.rho,
            "rate": self.r,
            "dividend": self.q,
            "feller_ratio": self.feller_ratio,
            "feller_satisfied": self.is_feller_satisfied
        }


def heston_characteristic_function(
    u: complex,
    S0: float,
    T: float,
    params: HestonParameters,
) -> complex:
    """Evaluates the Heston characteristic function using the Albrecher et al. (2007) stable formulation.
    
    phi(u) = E[exp(i * u * ln(S_T))]
    """
    r = params.r
    q = params.q
    v0 = params.v0
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho
    
    xi_sq = max(1e-8, xi ** 2)
    
    # d = sqrt((kappa - i * rho * xi * u)^2 + xi^2 * (u^2 + i * u))
    d_arg = (kappa - 1j * rho * xi * u) ** 2 + xi_sq * (u ** 2 + 1j * u)
    d = cmath.sqrt(d_arg)
    
    # g = (kappa - i * rho * xi * u - d) / (kappa - i * rho * xi * u + d)
    g_denom = kappa - 1j * rho * xi * u + d
    g_numer = kappa - 1j * rho * xi * u - d
    g = g_numer / g_denom if g_denom != 0 else complex(0.0)
    
    # Stable Albrecher formulation
    exp_minus_dT = cmath.exp(-d * T)
    log_term = cmath.log((1.0 - g * exp_minus_dT) / (1.0 - g))
    
    C = (r - q) * 1j * u * T + (kappa * theta / xi_sq) * ((kappa - 1j * rho * xi * u - d) * T - 2.0 * log_term)
    D = (kappa - 1j * rho * xi * u - d) / xi_sq * ((1.0 - exp_minus_dT) / (1.0 - g * exp_minus_dT))
    
    log_S0 = np.log(S0)
    phi = cmath.exp(C + D * v0 + 1j * u * log_S0)
    return phi


def carr_madan_fft_price(
    S0: float,
    K_list: Union[List[float], np.ndarray],
    T: float,
    params: HestonParameters,
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
    option_type: str = "call"
) -> np.ndarray:
    """Prices European options for an array of strikes using Carr-Madan (1999) Fast Fourier Transform (FFT)."""
    r = params.r
    q = params.q
    
    lambda_spacing = (2.0 * np.pi) / (N * eta)
    b = (N * lambda_spacing) / 2.0
    k_grid = -b + np.arange(N) * lambda_spacing  # log-strikes
    strikes_fft = np.exp(k_grid)
    v_j = np.arange(N) * eta
    
    psi_vals = np.zeros(N, dtype=complex)
    for j in range(N):
        v = v_j[j]
        u = v - (alpha + 1.0) * 1j
        phi_val = heston_characteristic_function(u, S0, T, params)
        denom = alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v
        psi_vals[j] = np.exp(-r * T) * phi_val / denom
        
    simpson_weights = np.ones(N)
    simpson_weights[0] = 1.0 / 3.0
    simpson_weights[1::2] = 4.0 / 3.0
    simpson_weights[2::2] = 2.0 / 3.0
    simpson_weights[-1] = 1.0 / 3.0
    
    fft_input = np.exp(1j * b * v_j) * psi_vals * (eta * simpson_weights)
    fft_output = np.fft.fft(fft_input)
    
    call_prices_fft = np.exp(-alpha * k_grid) * np.real(fft_output) / np.pi
    call_prices_fft = np.maximum(0.0, call_prices_fft)
    
    valid_idx = (strikes_fft > S0 * 0.05) & (strikes_fft < S0 * 15.0)
    spline = CubicSpline(strikes_fft[valid_idx], call_prices_fft[valid_idx], extrapolate=True)
    target_strikes = np.asarray(K_list)
    call_target = np.maximum(0.0, spline(target_strikes))
    
    if option_type.lower() == "put":
        put_target = call_target - S0 * np.exp(-q * T) + target_strikes * np.exp(-r * T)
        return np.maximum(0.0, put_target)
    return call_target


def fang_oosterlee_cos_price(
    S0: float,
    K: float,
    T: float,
    params: HestonParameters,
    N: int = 128,
    L: float = 8.0,
    option_type: str = "call"
) -> float:
    """Prices European options via Fang-Oosterlee (2008) Fourier-Cosine (COS) expansion method."""
    r = params.r
    q = params.q
    v0 = params.v0
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho
    
    x = np.log(S0 / K)
    
    c1 = (r - q) * T + (1.0 - np.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T + x
    c2 = (1.0 / (8.0 * kappa ** 3)) * (
        xi * T * kappa * np.exp(-kappa * T) * (v0 - theta) * (8.0 * kappa * rho - 4.0 * xi)
        + kappa * rho * xi * (1.0 - np.exp(-kappa * T)) * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * T * (-4.0 * kappa * rho * xi + xi ** 2 + 4.0 * kappa ** 2)
        + xi ** 2 * ((theta - 2.0 * v0) * np.exp(-2.0 * kappa * T) + theta * (6.0 * np.exp(-kappa * T) - 7.0) + 2.0 * v0)
        + 8.0 * kappa ** 2 * (v0 - theta) * (1.0 - np.exp(-kappa * T))
    )
    c2 = max(1e-4, abs(c2))
    
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)
    
    k_vec = np.arange(N)
    u_k = k_vec * np.pi / (b - a)
    
    # Characteristic function of y = ln(S_T / K) = ln(S_T) - ln(K)
    phi_y = np.array([
        heston_characteristic_function(u, S0, T, params) * np.exp(-1j * u * np.log(K))
        for u in u_k
    ])
    
    def chi_k(k, c, d):
        omega = k * np.pi / (b - a)
        return (np.cos(omega * (d - a)) * np.exp(d) - np.cos(omega * (c - a)) * np.exp(c)
                + omega * np.sin(omega * (d - a)) * np.exp(d) - omega * np.sin(omega * (c - a)) * np.exp(c)) / (1.0 + omega ** 2)
                
    def psi_k(k, c, d):
        omega = k * np.pi / (b - a)
        if k == 0:
            return d - c
        return (np.sin(omega * (d - a)) - np.sin(omega * (c - a))) / omega
        
    if option_type.lower() == "call":
        # Call payoff on [0, b]
        H_k = 2.0 / (b - a) * (np.array([chi_k(k, 0.0, b) - psi_k(k, 0.0, b) for k in range(N)]))
    else:
        # Put payoff on [a, 0]
        H_k = 2.0 / (b - a) * (np.array([-chi_k(k, a, 0.0) + psi_k(k, a, 0.0) for k in range(N)]))
        
    term_0 = 0.5 * np.real(phi_y[0] * np.exp(-1j * u_k[0] * a)) * H_k[0]
    term_rest = np.sum(np.real(phi_y[1:] * np.exp(-1j * u_k[1:] * a)) * H_k[1:])
    
    price = K * np.exp(-r * T) * (term_0 + term_rest)
    return max(0.0, float(price))


@dataclass
class HestonCalibrationResult:
    """Summary of Heston model calibration against market option surface."""
    calibrated_params: HestonParameters
    rmse: float
    mae: float
    r_squared: float
    feller_ratio: float
    feller_satisfied: bool
    iterations: int
    pricing_comparison_df: pd.DataFrame


class HestonOptionPricer:
    """High-performance Heston pricing and surface calibration engine."""
    
    def __init__(self, params: Optional[HestonParameters] = None):
        self.params = params or HestonParameters(v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.70)
        
    def price_call(self, S0: float, K: float, T: float, method: str = "cos") -> float:
        """Prices single European Call option."""
        if method.lower() == "cos":
            return fang_oosterlee_cos_price(S0, K, T, self.params, option_type="call")
        return float(carr_madan_fft_price(S0, [K], T, self.params, option_type="call")[0])
        
    def price_put(self, S0: float, K: float, T: float, method: str = "cos") -> float:
        """Prices single European Put option."""
        if method.lower() == "cos":
            return fang_oosterlee_cos_price(S0, K, T, self.params, option_type="put")
        return float(carr_madan_fft_price(S0, [K], T, self.params, option_type="put")[0])
        
    def price_chain_fft(self, S0: float, strikes: List[float], T: float, option_type: str = "call") -> np.ndarray:
        """Prices an array of strikes simultaneously via Carr-Madan FFT."""
        return carr_madan_fft_price(S0, strikes, T, self.params, option_type=option_type)

    def implied_volatility_surface(
        self,
        S0: float = 100.0,
        moneyness_grid: Optional[np.ndarray] = None,
        expiries: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Generates a Black-Scholes implied volatility matrix from Heston prices across moneyness & expiry."""
        if moneyness_grid is None:
            moneyness_grid = np.array([0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30])
        if expiries is None:
            expiries = np.array([0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00])
            
        r = self.params.r
        q = self.params.q
        
        iv_matrix = np.zeros((len(expiries), len(moneyness_grid)))
        
        for i, T in enumerate(expiries):
            strikes = moneyness_grid * S0
            calls = self.price_chain_fft(S0, strikes, T, option_type="call")
            for j, (K, C) in enumerate(zip(strikes, calls)):
                iv_matrix[i, j] = self._invert_bs_iv(C, S0, K, T, r, q)
                
        df_iv = pd.DataFrame(iv_matrix, index=expiries, columns=moneyness_grid)
        df_iv.index.name = "Expiry"
        df_iv.columns.name = "Moneyness"
        return df_iv

    @staticmethod
    def _invert_bs_iv(C: float, S0: float, K: float, T: float, r: float, q: float) -> float:
        """Fast Newton-Raphson inversion of Black-Scholes formula for Implied Volatility."""
        intrinsic = max(0.0, S0 * np.exp(-q * T) - K * np.exp(-r * T))
        if C <= intrinsic:
            return 0.05
            
        sigma = 0.25
        for _ in range(30):
            d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            vega = max(1e-6, S0 * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1))
            diff = price - C
            if abs(diff) < 1e-6:
                return float(sigma)
            sigma = max(0.01, min(3.0, sigma - diff / vega))
        return float(sigma)

    def calibrate(
        self,
        market_surface_df: pd.DataFrame,
        spot: float = 100.0,
        r: float = 0.05,
        q: float = 0.01,
        enforce_feller: bool = True,
    ) -> HestonCalibrationResult:
        """Calibrates Heston parameters (v0, kappa, theta, xi, rho) against market option quotes."""
        market_prices = market_surface_df["Call_Mid"].values
        strikes = market_surface_df["Strike"].values
        expiries = market_surface_df["Expiry"].values
        n_quotes = len(market_prices)
        
        bounds = [
            (0.001, 0.50),  # v0
            (0.1, 10.0),    # kappa
            (0.001, 0.50),  # theta
            (0.01, 1.50),   # xi
            (-0.99, 0.99),  # rho
        ]
        init_guess = np.array([0.04, 2.0, 0.04, 0.35, -0.65])
        
        def loss_fn(p_vec):
            v0_c, kappa_c, theta_c, xi_c, rho_c = p_vec
            p_obj = HestonParameters(v0=v0_c, kappa=kappa_c, theta=theta_c, xi=xi_c, rho=rho_c, r=r, q=q)
            
            model_prices = np.zeros(n_quotes)
            for idx in range(n_quotes):
                model_prices[idx] = fang_oosterlee_cos_price(spot, strikes[idx], expiries[idx], p_obj, N=64)
                
            errors = model_prices - market_prices
            mse = np.mean(errors ** 2)
            
            feller_diff = 2.0 * kappa_c * theta_c - xi_c ** 2
            penalty = 0.0
            if enforce_feller and feller_diff < 0:
                penalty = 5.0 * (feller_diff ** 2)
            return mse + penalty
            
        opt_res = minimize(loss_fn, init_guess, bounds=bounds, method="L-BFGS-B", options={"maxiter": 100, "ftol": 1e-7})
        
        v0_opt, kappa_opt, theta_opt, xi_opt, rho_opt = opt_res.x
        best_params = HestonParameters(v0=v0_opt, kappa=kappa_opt, theta=theta_opt, xi=xi_opt, rho=rho_opt, r=r, q=q)
        
        model_final = np.zeros(n_quotes)
        for idx in range(n_quotes):
            model_final[idx] = fang_oosterlee_cos_price(spot, strikes[idx], expiries[idx], best_params, N=128)
            
        rmse = float(np.sqrt(np.mean((model_final - market_prices) ** 2)))
        mae = float(np.mean(np.abs(model_final - market_prices)))
        ss_tot = np.sum((market_prices - np.mean(market_prices)) ** 2)
        ss_res = np.sum((market_prices - model_final) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0
        
        comp_df = pd.DataFrame({
            "Expiry": expiries,
            "Strike": strikes,
            "Market_Price": market_prices,
            "Heston_Price": model_final,
            "Pricing_Error": model_final - market_prices,
            "Abs_Error_Pct": np.abs((model_final - market_prices) / market_prices) * 100.0
        })
        
        self.params = best_params
        
        return HestonCalibrationResult(
            calibrated_params=best_params,
            rmse=rmse,
            mae=mae,
            r_squared=r2,
            feller_ratio=best_params.feller_ratio,
            feller_satisfied=best_params.is_feller_satisfied,
            iterations=int(opt_res.nfev),
            pricing_comparison_df=comp_df
        )
