"""Exotic Option Pricing via Path-Dependent Monte Carlo and Longstaff-Schwartz LSM."""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from pricing_models.data.sample_market import BlackScholesAnalytical


@dataclass
class AsianOptionResult:
    price: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    option_type: str
    averaging_type: str
    n_simulations: int
    n_steps: int
    geometric_analytical_price: Optional[float] = None
    control_variate_used: bool = False


@dataclass
class BarrierOptionResult:
    price: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    option_type: str
    barrier_type: str
    barrier_level: float
    rebate: float
    hit_probability: float
    n_simulations: int
    n_steps: int


@dataclass
class LookbackOptionResult:
    price: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    option_type: str
    lookback_type: str
    strike: Optional[float]
    n_simulations: int
    n_steps: int


@dataclass
class LSMOptionResult:
    price: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    european_price: float
    early_exercise_premium: float
    option_type: str
    n_simulations: int
    n_steps: int
    polynomial_degree: int
    exercise_frequency_pct: float


class ExoticOptionPricer:
    """Pricer for Path-Dependent Exotic Derivatives and American Options via LSM."""

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> None:
        if S0 <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
            raise ValueError("Parameters S0, K, T, and sigma must be strictly positive.")

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)

    def _simulate_paths(
        self,
        n_simulations: int,
        n_steps: int,
        antithetic: bool = True,
        random_state: Optional[int] = 42,
    ) -> np.ndarray:
        """Simulates discrete multi-step paths: shape (M, n_steps + 1)."""
        if random_state is not None:
            np.random.seed(random_state)

        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        vol_sqrt_dt = self.sigma * np.sqrt(dt)

        if antithetic:
            half_n = n_simulations // 2
            Z = np.random.standard_normal((half_n, n_steps))
            Z = np.vstack([Z, -Z])
            if Z.shape[0] < n_simulations:
                extra = np.random.standard_normal((n_simulations - Z.shape[0], n_steps))
                Z = np.vstack([Z, extra])
        else:
            Z = np.random.standard_normal((n_simulations, n_steps))

        log_returns = drift + vol_sqrt_dt * Z
        log_paths = np.zeros((n_simulations, n_steps + 1))
        log_paths[:, 0] = np.log(self.S0)
        log_paths[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)

        paths = np.exp(log_paths)
        return paths

    def price_asian(
        self,
        option_type: str = "call",
        averaging_type: str = "arithmetic",
        n_simulations: int = 100000,
        n_steps: int = 252,
        antithetic: bool = True,
        control_variate: bool = True,
        random_state: Optional[int] = 42,
    ) -> AsianOptionResult:
        """Prices Asian options with optional Geometric Asian Control Variate."""
        is_call = option_type.lower() == "call"
        disc = np.exp(-self.r * self.T)

        paths = self._simulate_paths(n_simulations, n_steps, antithetic=antithetic, random_state=random_state)
        # Exclude t=0 in averaging or include depending on convention (standard is t_1 .. t_N)
        path_samples = paths[:, 1:]

        arithmetic_avg = np.mean(path_samples, axis=1)
        geometric_avg = np.exp(np.mean(np.log(path_samples), axis=1))

        if is_call:
            payoff_arith = np.maximum(arithmetic_avg - self.K, 0.0)
            payoff_geom = np.maximum(geometric_avg - self.K, 0.0)
        else:
            payoff_arith = np.maximum(self.K - arithmetic_avg, 0.0)
            payoff_geom = np.maximum(self.K - geometric_avg, 0.0)

        # Exact Geometric Asian analytical price (Kemna & Vorst 1990)
        n = n_steps
        sigma_g = self.sigma * np.sqrt((2.0 * n + 1.0) / (6.0 * (n + 1.0)))
        mu_g = (self.r - self.q - 0.5 * self.sigma ** 2) * (n + 1.0) / (2.0 * n) + 0.5 * (sigma_g ** 2)
        d1_g = (np.log(self.S0 / self.K) + (mu_g + 0.5 * sigma_g ** 2) * self.T) / (sigma_g * np.sqrt(self.T))
        d2_g = d1_g - sigma_g * np.sqrt(self.T)

        if is_call:
            exact_geom_px = float(
                np.exp(-self.r * self.T) * (self.S0 * np.exp(mu_g * self.T) * norm.cdf(d1_g) - self.K * norm.cdf(d2_g))
            )
        else:
            exact_geom_px = float(
                np.exp(-self.r * self.T) * (self.K * norm.cdf(-d2_g) - self.S0 * np.exp(mu_g * self.T) * norm.cdf(-d1_g))
            )

        if averaging_type.lower() == "geometric":
            disc_payoffs = disc * payoff_geom
            sample_mean = float(np.mean(disc_payoffs))
            sample_var = float(np.var(disc_payoffs, ddof=1))
            used_cv = False
        else:
            # Arithmetic Asian
            disc_arith = disc * payoff_arith
            disc_geom = disc * payoff_geom

            if control_variate:
                cov_mat = np.cov(disc_arith, disc_geom)
                cov_xy = cov_mat[0, 1]
                var_y = cov_mat[1, 1]
                c_star = cov_xy / var_y if var_y > 1e-12 else 1.0

                cv_payoffs = disc_arith - c_star * (disc_geom - exact_geom_px)
                sample_mean = float(np.mean(cv_payoffs))
                sample_var = float(np.var(cv_payoffs, ddof=1))
                used_cv = True
            else:
                sample_mean = float(np.mean(disc_arith))
                sample_var = float(np.var(disc_arith, ddof=1))
                used_cv = False

        se = float(np.sqrt(sample_var / n_simulations))
        ci_95 = (sample_mean - 1.95996 * se, sample_mean + 1.95996 * se)

        return AsianOptionResult(
            price=sample_mean,
            standard_error=se,
            confidence_interval_95=ci_95,
            option_type=option_type.lower(),
            averaging_type=averaging_type.lower(),
            n_simulations=n_simulations,
            n_steps=n_steps,
            geometric_analytical_price=exact_geom_px,
            control_variate_used=used_cv,
        )

    def price_barrier(
        self,
        option_type: str = "call",
        barrier_type: str = "up_and_out",
        barrier_level: float = 120.0,
        rebate: float = 0.0,
        n_simulations: int = 100000,
        n_steps: int = 252,
        brownian_bridge: bool = True,
        antithetic: bool = True,
        random_state: Optional[int] = 42,
    ) -> BarrierOptionResult:
        """Prices Barrier options with continuous Brownian Bridge crossing probability."""
        is_call = option_type.lower() == "call"
        b_type = barrier_type.lower()
        disc = np.exp(-self.r * self.T)
        dt = self.T / n_steps
        B = float(barrier_level)

        paths = self._simulate_paths(n_simulations, n_steps, antithetic=antithetic, random_state=random_state)
        S_T = paths[:, -1]

        # Plain vanilla terminal payoff
        vanilla_payoff = np.maximum(S_T - self.K, 0.0) if is_call else np.maximum(self.K - S_T, 0.0)

        # Check barrier breach
        if "up" in b_type:
            # Breach occurs if max(S) >= B
            max_discrete = np.max(paths, axis=1)
            hit_discrete = max_discrete >= B

            if brownian_bridge:
                # Probability of crossing B between step k and k+1
                S_from = paths[:, :-1]
                S_to = paths[:, 1:]
                # Only active if both endpoints are below barrier B
                valid_mask = (S_from < B) & (S_to < B)
                exponent = -2.0 * (B - S_from) * (B - S_to) / ((self.sigma ** 2) * dt * S_from * S_to)
                p_cross_step = np.where(valid_mask, np.exp(np.clip(exponent, -700, 0)), 0.0)
                p_no_cross = np.prod(1.0 - p_cross_step, axis=1)
                p_hit = 1.0 - np.where(~hit_discrete, p_no_cross, 0.0)
                is_hit = (p_hit >= np.random.uniform(0, 1, size=n_simulations)) | hit_discrete
            else:
                is_hit = hit_discrete
        else:
            # Down barrier: breach occurs if min(S) <= B
            min_discrete = np.min(paths, axis=1)
            hit_discrete = min_discrete <= B

            if brownian_bridge:
                S_from = paths[:, :-1]
                S_to = paths[:, 1:]
                valid_mask = (S_from > B) & (S_to > B)
                exponent = -2.0 * (S_from - B) * (S_to - B) / ((self.sigma ** 2) * dt * S_from * S_to)
                p_cross_step = np.where(valid_mask, np.exp(np.clip(exponent, -700, 0)), 0.0)
                p_no_cross = np.prod(1.0 - p_cross_step, axis=1)
                p_hit = 1.0 - np.where(~hit_discrete, p_no_cross, 0.0)
                is_hit = (p_hit >= np.random.uniform(0, 1, size=n_simulations)) | hit_discrete
            else:
                is_hit = hit_discrete

        # Determine active payoff
        if b_type == "up_and_out" or b_type == "down_and_out":
            # Knock-out: pays if NOT hit, otherwise rebate
            payoffs = np.where(~is_hit, vanilla_payoff, rebate)
        elif b_type == "up_and_in" or b_type == "down_and_in":
            # Knock-in: pays if HIT, otherwise rebate
            payoffs = np.where(is_hit, vanilla_payoff, rebate)
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")

        disc_payoffs = disc * payoffs
        sample_mean = float(np.mean(disc_payoffs))
        sample_var = float(np.var(disc_payoffs, ddof=1))
        se = float(np.sqrt(sample_var / n_simulations))
        ci_95 = (sample_mean - 1.95996 * se, sample_mean + 1.95996 * se)

        return BarrierOptionResult(
            price=sample_mean,
            standard_error=se,
            confidence_interval_95=ci_95,
            option_type=option_type.lower(),
            barrier_type=b_type,
            barrier_level=B,
            rebate=rebate,
            hit_probability=float(np.mean(is_hit)),
            n_simulations=n_simulations,
            n_steps=n_steps,
        )

    def price_lookback(
        self,
        option_type: str = "call",
        lookback_type: str = "floating",
        strike: Optional[float] = None,
        n_simulations: int = 100000,
        n_steps: int = 252,
        antithetic: bool = True,
        random_state: Optional[int] = 42,
    ) -> LookbackOptionResult:
        """Prices Floating & Fixed Strike Lookback options."""
        is_call = option_type.lower() == "call"
        l_type = lookback_type.lower()
        disc = np.exp(-self.r * self.T)
        K_val = strike if strike is not None else self.K

        paths = self._simulate_paths(n_simulations, n_steps, antithetic=antithetic, random_state=random_state)
        S_T = paths[:, -1]
        S_max = np.max(paths, axis=1)
        S_min = np.min(paths, axis=1)

        if l_type == "floating":
            # Floating strike
            payoffs = np.maximum(S_T - S_min, 0.0) if is_call else np.maximum(S_max - S_T, 0.0)
        elif l_type == "fixed":
            # Fixed strike
            payoffs = np.maximum(S_max - K_val, 0.0) if is_call else np.maximum(K_val - S_min, 0.0)
        else:
            raise ValueError(f"Unknown lookback type: {lookback_type}. Must be 'floating' or 'fixed'.")

        disc_payoffs = disc * payoffs
        sample_mean = float(np.mean(disc_payoffs))
        sample_var = float(np.var(disc_payoffs, ddof=1))
        se = float(np.sqrt(sample_var / n_simulations))
        ci_95 = (sample_mean - 1.95996 * se, sample_mean + 1.95996 * se)

        return LookbackOptionResult(
            price=sample_mean,
            standard_error=se,
            confidence_interval_95=ci_95,
            option_type=option_type.lower(),
            lookback_type=l_type,
            strike=K_val if l_type == "fixed" else None,
            n_simulations=n_simulations,
            n_steps=n_steps,
        )

    def price_american_lsm(
        self,
        option_type: str = "put",
        n_simulations: int = 50000,
        n_steps: int = 50,
        polynomial_degree: int = 3,
        antithetic: bool = True,
        random_state: Optional[int] = 42,
    ) -> LSMOptionResult:
        """Prices American options via Longstaff-Schwartz Least Squares Monte Carlo (LSM)."""
        is_call = option_type.lower() == "call"
        dt = self.T / n_steps
        disc_step = np.exp(-self.r * dt)

        # European baseline
        bs_eur = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, option_type)

        paths = self._simulate_paths(n_simulations, n_steps, antithetic=antithetic, random_state=random_state)

        # Terminal payoffs at step N
        S_T = paths[:, -1]
        cashflows = np.maximum(S_T - self.K, 0.0) if is_call else np.maximum(self.K - S_T, 0.0)
        exercise_times = np.full(n_simulations, n_steps, dtype=int)

        # Backward recursion
        for k in range(n_steps - 1, 0, -1):
            S_k = paths[:, k]
            intrinsic_k = np.maximum(S_k - self.K, 0.0) if is_call else np.maximum(self.K - S_k, 0.0)

            # In-The-Money (ITM) paths only
            itm_mask = intrinsic_k > 0.0
            if not np.any(itm_mask):
                continue

            itm_indices = np.where(itm_mask)[0]
            S_itm = S_k[itm_indices]
            # Discount future cashflow to step k
            time_steps_ahead = exercise_times[itm_indices] - k
            Y_disc = cashflows[itm_indices] * (disc_step ** time_steps_ahead)

            # Polynomial regression: basis functions 1, S, S^2, ...
            deg = min(polynomial_degree, len(itm_indices) - 1)
            reg_coeffs = np.polyfit(S_itm, Y_disc, deg)
            continuation_est = np.polyval(reg_coeffs, S_itm)

            # Early exercise decision
            early_ex_mask = intrinsic_k[itm_indices] > continuation_est
            if np.any(early_ex_mask):
                exercised_idx = itm_indices[early_ex_mask]
                cashflows[exercised_idx] = intrinsic_k[exercised_idx]
                exercise_times[exercised_idx] = k

        # Discount optimal cashflows to t=0
        discount_factors = disc_step ** exercise_times
        pv_cashflows = cashflows * discount_factors

        sample_mean = float(np.mean(pv_cashflows))
        sample_var = float(np.var(pv_cashflows, ddof=1))
        se = float(np.sqrt(sample_var / n_simulations))
        ci_95 = (sample_mean - 1.95996 * se, sample_mean + 1.95996 * se)
        early_prem = float(max(sample_mean - bs_eur, 0.0))
        ex_freq = float(np.mean(exercise_times < n_steps) * 100.0)

        return LSMOptionResult(
            price=sample_mean,
            standard_error=se,
            confidence_interval_95=ci_95,
            european_price=bs_eur,
            early_exercise_premium=early_prem,
            option_type=option_type.lower(),
            n_simulations=n_simulations,
            n_steps=n_steps,
            polynomial_degree=polynomial_degree,
            exercise_frequency_pct=ex_freq,
        )
