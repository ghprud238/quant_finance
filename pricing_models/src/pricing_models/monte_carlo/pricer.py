"""Monte Carlo Option Pricing Engine with Antithetic and Control Variates."""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from pricing_models.data.sample_market import BlackScholesAnalytical


@dataclass
class MonteCarloGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    n_simulations: int
    option_type: str
    exercise_style: str
    antithetic_used: bool
    control_variate_used: bool
    variance_reduction_ratio: float
    analytical_bs_price: float
    greeks: Optional[MonteCarloGreeks] = None


class MonteCarloOptionPricer:
    """Monte Carlo Option Pricing Engine supporting standard and variance-reduced simulations."""

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> None:
        if S0 <= 0.0:
            raise ValueError(f"S0 must be positive, got {S0}")
        if K <= 0.0:
            raise ValueError(f"K must be positive, got {K}")
        if T <= 0.0:
            raise ValueError(f"T must be positive, got {T}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)

    def _simulate_terminal_prices(
        self,
        n_simulations: int,
        antithetic: bool = True,
        random_state: Optional[int] = None,
        S0: Optional[float] = None,
        sigma: Optional[float] = None,
        r: Optional[float] = None,
        T: Optional[float] = None,
    ) -> np.ndarray:
        """Simulates terminal asset prices under risk-neutral measure Q."""
        if random_state is not None:
            np.random.seed(random_state)

        S0_val = S0 if S0 is not None else self.S0
        sig_val = sigma if sigma is not None else self.sigma
        r_val = r if r is not None else self.r
        T_val = T if T is not None else self.T
        q_val = self.q

        drift = (r_val - q_val - 0.5 * sig_val ** 2) * T_val
        diffusion = sig_val * np.sqrt(T_val)

        if antithetic:
            half_n = n_simulations // 2
            Z = np.random.standard_normal(half_n)
            Z = np.concatenate([Z, -Z])
            if len(Z) < n_simulations:
                extra = np.random.standard_normal(n_simulations - len(Z))
                Z = np.concatenate([Z, extra])
        else:
            Z = np.random.standard_normal(n_simulations)

        S_T = S0_val * np.exp(drift + diffusion * Z)
        return S_T

    def price(
        self,
        option_type: str = "call",
        n_simulations: int = 100000,
        antithetic: bool = True,
        control_variate: bool = False,
        random_state: Optional[int] = 42,
    ) -> MonteCarloResult:
        """Prices European options via Monte Carlo with optional Antithetic & Control Variates."""
        is_call = option_type.lower() == "call"
        disc = np.exp(-self.r * self.T)
        bs_exact = BlackScholesAnalytical.price(self.S0, self.K, self.T, self.r, self.sigma, self.q, option_type)

        # Baseline Simulation
        S_T = self._simulate_terminal_prices(n_simulations, antithetic=antithetic, random_state=random_state)
        payoffs = np.maximum(S_T - self.K, 0.0) if is_call else np.maximum(self.K - S_T, 0.0)
        discounted_payoffs = disc * payoffs

        raw_var = float(np.var(discounted_payoffs, ddof=1))

        if control_variate:
            # Use underlying discounted asset price as control variate: E[e^{-rT} S_T] = S0 * e^{-qT}
            control_exact = self.S0 * np.exp(-self.q * self.T)
            control_sample = disc * S_T

            cov_matrix = np.cov(discounted_payoffs, control_sample)
            cov_xy = cov_matrix[0, 1]
            var_ctrl = cov_matrix[1, 1]
            c_star = cov_xy / var_ctrl if var_ctrl > 1e-12 else 0.0

            cv_payoffs = discounted_payoffs - c_star * (control_sample - control_exact)
            sample_mean = float(np.mean(cv_payoffs))
            sample_var = float(np.var(cv_payoffs, ddof=1))
        else:
            sample_mean = float(np.mean(discounted_payoffs))
            sample_var = raw_var

        se = float(np.sqrt(sample_var / n_simulations))
        ci_95 = (sample_mean - 1.95996 * se, sample_mean + 1.95996 * se)
        var_red_ratio = raw_var / max(sample_var, 1e-12)

        return MonteCarloResult(
            price=sample_mean,
            standard_error=se,
            confidence_interval_95=ci_95,
            n_simulations=n_simulations,
            option_type=option_type.lower(),
            exercise_style="european",
            antithetic_used=antithetic,
            control_variate_used=control_variate,
            variance_reduction_ratio=var_red_ratio,
            analytical_bs_price=bs_exact,
        )

    def greeks(
        self,
        option_type: str = "call",
        n_simulations: int = 100000,
        dS_pct: float = 0.01,
        dvol_pct: float = 0.01,
        dr_pct: float = 0.0001,
        dt_days: float = 1.0,
        random_state: int = 42,
    ) -> MonteCarloGreeks:
        """Estimates Greeks using finite difference bumping with common random numbers."""
        dS = self.S0 * dS_pct
        dvol = dvol_pct
        dr = dr_pct
        dt = dt_days / 365.0

        # Base price
        base_res = self.price(option_type, n_simulations, antithetic=True, random_state=random_state)
        p0 = base_res.price

        # Delta & Gamma via central difference in S0
        pricer_up = MonteCarloOptionPricer(self.S0 + dS, self.K, self.T, self.r, self.sigma, self.q)
        pricer_dn = MonteCarloOptionPricer(self.S0 - dS, self.K, self.T, self.r, self.sigma, self.q)
        p_up = pricer_up.price(option_type, n_simulations, antithetic=True, random_state=random_state).price
        p_dn = pricer_dn.price(option_type, n_simulations, antithetic=True, random_state=random_state).price

        delta = float((p_up - p_dn) / (2.0 * dS))
        gamma = float((p_up - 2.0 * p0 + p_dn) / (dS ** 2))

        # Vega via vol bump
        pricer_v_up = MonteCarloOptionPricer(self.S0, self.K, self.T, self.r, self.sigma + dvol, self.q)
        p_v_up = pricer_v_up.price(option_type, n_simulations, antithetic=True, random_state=random_state).price
        vega = float((p_v_up - p0) / dvol)  # per 100% vol

        # Theta via time decay
        if self.T > dt:
            pricer_t = MonteCarloOptionPricer(self.S0, self.K, self.T - dt, self.r, self.sigma, self.q)
            p_t = pricer_t.price(option_type, n_simulations, antithetic=True, random_state=random_state).price
            theta_ann = float((p_t - p0) / dt)
        else:
            theta_ann = 0.0

        # Rho via interest rate bump
        pricer_r = MonteCarloOptionPricer(self.S0, self.K, self.T, self.r + dr, self.sigma, self.q)
        p_r = pricer_r.price(option_type, n_simulations, antithetic=True, random_state=random_state).price
        rho = float((p_r - p0) / dr)

        return MonteCarloGreeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta_ann,
            rho=rho,
        )
