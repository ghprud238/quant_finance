"""Volatility Smile Modeling and Interpolation.

Implements:
- Parametric Stochastic Volatility Inspired (SVI) model (Gatheral 2004)
- Cubic Spline interpolation with boundary extrapolation
- Volatility skew and convexity metrics
- Smile curve generation matching market empirical features
"""

from typing import Union, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


@dataclass
class SVIParameters:
    """Parameters of the Raw SVI (Stochastic Volatility Inspired) model.

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where k = ln(K / F) is the log-moneyness.
    """
    a: float        # Vertical level of variance
    b: float        # Slopes of asymptotes (b >= 0)
    rho: float      # Asymmetry / skew of asymptotes (-1 < rho < 1)
    m: float        # Horizontal location shift
    sigma: float    # Vertex smoothness / curvature (sigma > 0)
    time_to_expiry: float

    def total_variance(self, log_moneyness: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates total implied variance w(k) = sigma_IV^2 * T."""
        k = np.asarray(log_moneyness, dtype=float)
        disc = np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        w = self.a + self.b * (self.rho * (k - self.m) + disc)
        # Numerical protection: ensure non-negative variance
        w = np.maximum(w, 1e-8)
        return float(w) if np.isscalar(log_moneyness) else w

    def implied_volatility(self, log_moneyness: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates implied volatility sigma_IV(k) = sqrt(w(k) / T)."""
        w = self.total_variance(log_moneyness)
        iv = np.sqrt(w / self.time_to_expiry)
        return float(iv) if np.isscalar(log_moneyness) else iv


class VolatilitySmile:
    """Volatility Smile representation for a single expiration horizon.

    Supports SVI parametric calibration and non-parametric Cubic Spline interpolation.
    """

    def __init__(
        self,
        spot: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
    ):
        """Initializes the Volatility Smile model.

        Args:
            spot: Underlying spot price S_0
            time_to_expiry: Expiration in years T
            risk_free_rate: Continuously compounded interest rate r
            dividend_yield: Continuous dividend yield q
        """
        self.spot = float(spot)
        self.time_to_expiry = float(time_to_expiry)
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yield = float(dividend_yield)
        self.forward = self.spot * np.exp((self.risk_free_rate - self.dividend_yield) * self.time_to_expiry)

        self.svi_params: Optional[SVIParameters] = None
        self._spline: Optional[CubicSpline] = None
        self.market_strikes: Optional[np.ndarray] = None
        self.market_ivs: Optional[np.ndarray] = None

    def log_moneyness(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates forward log-moneyness k = ln(K / F)."""
        k = np.asarray(strike, dtype=float)
        lm = np.log(k / self.forward)
        return float(lm) if np.isscalar(strike) else lm

    def strike_from_log_moneyness(self, log_moneyness: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates strike K from forward log-moneyness: K = F * exp(k)."""
        lm = np.asarray(log_moneyness, dtype=float)
        k = self.forward * np.exp(lm)
        return float(k) if np.isscalar(log_moneyness) else k

    def fit_svi(
        self,
        strikes: Union[List[float], np.ndarray, pd.Series],
        implied_vols: Union[List[float], np.ndarray, pd.Series],
        initial_guess: Optional[List[float]] = None,
    ) -> SVIParameters:
        """Fits the Gatheral Raw SVI parameterization to observed market strikes & IVs.

        Minimizes sum of squared errors between market total variance and SVI total variance.
        """
        stk = np.asarray(strikes, dtype=float)
        ivs = np.asarray(implied_vols, dtype=float)

        # Filter NaNs and invalid points
        valid = (~np.isnan(stk)) & (~np.isnan(ivs)) & (ivs > 0) & (stk > 0)
        stk = stk[valid]
        ivs = ivs[valid]

        if len(stk) < 3:
            raise ValueError(f"At least 3 valid strike/IV pairs required to fit SVI, got {len(stk)}")

        self.market_strikes = np.sort(stk)
        self.market_ivs = ivs[np.argsort(stk)]

        k_pts = self.log_moneyness(stk)
        w_mkt = (ivs ** 2) * self.time_to_expiry

        # ATM total variance estimate
        atm_idx = np.argmin(np.abs(k_pts))
        w_atm = w_mkt[atm_idx]

        if initial_guess is None:
            # Initial guess: [a, b, rho, m, sigma]
            x0 = [w_atm * 0.8, 0.1, -0.2, 0.0, 0.1]
        else:
            x0 = list(initial_guess)

        bounds = [
            (-1.0, 5.0),    # a
            (1e-5, 5.0),    # b >= 0
            (-0.999, 0.999),# -1 < rho < 1
            (-2.0, 2.0),    # m
            (1e-5, 2.0),    # sigma > 0
        ]

        def objective(p: List[float]) -> float:
            a, b, rho, m, sig = p
            # Penalty for violating no-arbitrage non-negativity constraint
            if a + b * sig * np.sqrt(1.0 - rho ** 2) < 0:
                return 1e6

            disc = np.sqrt((k_pts - m) ** 2 + sig ** 2)
            w_model = a + b * (rho * (k_pts - m) + disc)
            residuals = w_model - w_mkt
            return float(np.sum(residuals ** 2))

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        a, b, rho, m, sig = res.x
        self.svi_params = SVIParameters(
            a=float(a),
            b=float(b),
            rho=float(rho),
            m=float(m),
            sigma=float(sig),
            time_to_expiry=self.time_to_expiry,
        )
        return self.svi_params

    def fit_spline(
        self,
        strikes: Union[List[float], np.ndarray, pd.Series],
        implied_vols: Union[List[float], np.ndarray, pd.Series],
    ) -> CubicSpline:
        """Fits a direct Cubic Spline across strike / moneyness for smooth non-parametric IV."""
        stk = np.asarray(strikes, dtype=float)
        ivs = np.asarray(implied_vols, dtype=float)

        valid = (~np.isnan(stk)) & (~np.isnan(ivs)) & (ivs > 0) & (stk > 0)
        stk = stk[valid]
        ivs = ivs[valid]

        sort_idx = np.argsort(stk)
        self.market_strikes = stk[sort_idx]
        self.market_ivs = ivs[sort_idx]

        # Natural boundary condition cubic spline with flat linear extrapolation
        self._spline = CubicSpline(self.market_strikes, self.market_ivs, bc_type="natural", extrapolate=True)
        return self._spline

    def get_iv(
        self,
        strike: Union[float, np.ndarray],
        method: str = "svi",
    ) -> Union[float, np.ndarray]:
        """Evaluates implied volatility at arbitrary strike(s).

        Args:
            strike: Strike price(s)
            method: 'svi' (default) or 'spline'
        """
        if method.lower() == "svi":
            if self.svi_params is None:
                if self._spline is not None:
                    return self.get_iv(strike, method="spline")
                raise RuntimeError("SVI model has not been fitted yet. Call fit_svi() first.")
            k = self.log_moneyness(strike)
            return self.svi_params.implied_volatility(k)

        elif method.lower() == "spline":
            if self._spline is None:
                if self.svi_params is not None:
                    return self.get_iv(strike, method="svi")
                raise RuntimeError("Spline model has not been fitted yet. Call fit_spline() first.")
            iv = self._spline(strike)
            return float(np.clip(iv, 0.01, 5.0)) if np.isscalar(strike) else np.clip(iv, 0.01, 5.0)

        else:
            raise ValueError(f"Unknown interpolation method '{method}', choose 'svi' or 'spline'")

    def get_atm_vol(self, method: str = "svi") -> float:
        """Returns At-The-Money (ATM, K = Forward) implied volatility."""
        return float(self.get_iv(self.forward, method=method))

    def get_skew(self, strike_down: Optional[float] = None, strike_up: Optional[float] = None) -> float:
        """Calculates implied volatility skew: (IV(K_low) - IV(K_high)) / (K_high - K_low).

        Default evaluates 90% vs 110% moneyness skew.
        """
        k_down = strike_down if strike_down is not None else 0.90 * self.spot
        k_up = strike_up if strike_up is not None else 1.10 * self.spot
        iv_down = self.get_iv(k_down)
        iv_up = self.get_iv(k_up)
        return float((iv_down - iv_up) / ((k_up - k_down) / self.spot))

    def get_convexity(self, delta_k: float = 0.05) -> float:
        """Calculates second derivative / smile curvature around ATM: (IV(ATM+d) + IV(ATM-d) - 2*IV(ATM)) / d^2."""
        k_atm = self.forward
        d = delta_k * self.spot
        iv_atm = self.get_iv(k_atm)
        iv_up = self.get_iv(k_atm + d)
        iv_down = self.get_iv(k_atm - d)
        return float((iv_up + iv_down - 2.0 * iv_atm) / (d ** 2))

    def generate_smile_curve(
        self,
        moneyness_range: Tuple[float, float] = (0.6, 1.4),
        n_points: int = 100,
        method: str = "svi",
    ) -> pd.DataFrame:
        """Generates a dense DataFrame of the smile curve across moneyness and strike range."""
        m_min, m_max = moneyness_range
        m_grid = np.linspace(m_min, m_max, n_points)
        k_grid = m_grid * self.spot
        iv_grid = self.get_iv(k_grid, method=method)

        return pd.DataFrame({
            "moneyness": m_grid,
            "strike": k_grid,
            "implied_vol": iv_grid,
            "time_to_expiry": self.time_to_expiry,
        })
