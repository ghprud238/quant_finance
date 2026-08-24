"""Volatility Surface Modeling, 2D/3D Interpolation & Local Volatility.

Implements:
- 2D/3D Volatility Surface parameterized by Strike/Moneyness and Time-to-Expiry (T)
- Term structure of SVI smiles with non-arbitrage variance interpolation
- Total Implied Variance w(K, T) = sigma_IV^2 * T
- Dupire Local Volatility extraction from implied variance
- 3D Surface meshgrid generation for visualization
"""

from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d
from .smile import VolatilitySmile, SVIParameters


class VolatilitySurface:
    """2D/3D Implied Volatility Surface across Moneyness and Expiration Horizontals."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
    ):
        """Initializes the Volatility Surface.

        Args:
            spot: Underlying asset spot price S_0
            risk_free_rate: Risk-free rate r
            dividend_yield: Dividend yield q
        """
        self.spot = float(spot)
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yield = float(dividend_yield)

        self.smiles: Dict[float, VolatilitySmile] = {}
        self.expiries: List[float] = []
        self._2d_spline: Optional[RectBivariateSpline] = None
        self._grid_moneyness: Optional[np.ndarray] = None
        self._grid_expiries: Optional[np.ndarray] = None
        self._grid_ivs: Optional[np.ndarray] = None

    def add_smile(self, time_to_expiry: float, smile: VolatilitySmile) -> None:
        """Registers a pre-fitted VolatilitySmile at expiration T."""
        t = float(time_to_expiry)
        self.smiles[t] = smile
        self.expiries = sorted(list(self.smiles.keys()))
        self._rebuild_interpolator()

    def fit_from_option_table(
        self,
        df_options: pd.DataFrame,
        strike_col: str = "strike",
        expiry_col: str = "expiry",
        iv_col: str = "implied_vol",
        method: str = "svi",
    ) -> None:
        """Fits the entire surface from an options chain DataFrame containing multiple expirations."""
        unique_expiries = np.sort(df_options[expiry_col].unique())
        self.smiles.clear()

        for t in unique_expiries:
            df_slice = df_options[df_options[expiry_col] == t]
            stk = df_slice[strike_col].values
            ivs = df_slice[iv_col].values

            smile = VolatilitySmile(
                spot=self.spot,
                time_to_expiry=float(t),
                risk_free_rate=self.risk_free_rate,
                dividend_yield=self.dividend_yield,
            )
            if method.lower() == "svi":
                try:
                    smile.fit_svi(stk, ivs)
                except Exception:
                    smile.fit_spline(stk, ivs)
            else:
                smile.fit_spline(stk, ivs)

            self.smiles[float(t)] = smile

        self.expiries = sorted(list(self.smiles.keys()))
        self._rebuild_interpolator()

    def _rebuild_interpolator(self, n_moneyness_pts: int = 60) -> None:
        """Reconstructs the smooth 2D bivariate spline across (moneyness, expiry)."""
        if len(self.expiries) < 2:
            return

        m_grid = np.linspace(0.5, 1.6, n_moneyness_pts)
        t_arr = np.array(self.expiries)
        iv_matrix = np.zeros((len(m_grid), len(t_arr)))

        for j, t in enumerate(t_arr):
            smile = self.smiles[t]
            k_pts = m_grid * self.spot
            iv_matrix[:, j] = smile.get_iv(k_pts)

        self._grid_moneyness = m_grid
        self._grid_expiries = t_arr
        self._grid_ivs = iv_matrix

        # 2D Bivariate spline for smooth C2 interpolation
        self._2d_spline = RectBivariateSpline(
            m_grid,
            t_arr,
            iv_matrix,
            kx=min(3, len(m_grid) - 1),
            ky=min(2 if len(t_arr) > 2 else 1, len(t_arr) - 1),
        )

    def get_iv(
        self,
        strike: Union[float, np.ndarray],
        time_to_expiry: float,
    ) -> Union[float, np.ndarray]:
        """Interpolates implied volatility at arbitrary (K, T)."""
        t = float(time_to_expiry)

        # If exact match exists in registered smiles
        if t in self.smiles:
            return self.smiles[t].get_iv(strike)

        # If 2D spline is ready, interpolate
        if self._2d_spline is not None:
            m = np.asarray(strike, dtype=float) / self.spot
            t_eval = np.clip(t, self.expiries[0], self.expiries[-1])
            iv = self._2d_spline(m, t_eval, grid=False)
            iv = np.clip(iv, 0.01, 5.0)
            return float(iv) if np.isscalar(strike) else iv

        # Fallback to nearest smile if only 1 smile registered
        if len(self.expiries) == 1:
            return self.smiles[self.expiries[0]].get_iv(strike)

        raise RuntimeError("No valid smiles loaded on the volatility surface.")

    def total_variance(
        self,
        strike: Union[float, np.ndarray],
        time_to_expiry: float,
    ) -> Union[float, np.ndarray]:
        """Calculates total implied variance w(K, T) = sigma_IV(K, T)^2 * T."""
        iv = self.get_iv(strike, time_to_expiry)
        return (iv ** 2) * time_to_expiry

    def generate_mesh(
        self,
        moneyness_range: Tuple[float, float] = (0.6, 1.4),
        expiry_range: Tuple[float, float] = (0.08, 2.0),
        n_moneyness: int = 50,
        n_expiries: int = 30,
    ) -> Dict[str, np.ndarray]:
        """Generates structured 2D coordinate meshgrids for 3D visualization.

        Returns:
            Dictionary containing:
            - 'moneyness_grid': 2D grid of K / S_0
            - 'expiry_grid': 2D grid of T (years)
            - 'strike_grid': 2D grid of Strikes K
            - 'iv_grid': 2D grid of Implied Volatility sigma_IV
            - 'total_variance_grid': 2D grid of Total Variance w = sigma_IV^2 * T
        """
        m_vec = np.linspace(moneyness_range[0], moneyness_range[1], n_moneyness)
        t_vec = np.linspace(expiry_range[0], expiry_range[1], n_expiries)

        M_grid, T_grid = np.meshgrid(m_vec, t_vec, indexing="ij")
        K_grid = M_grid * self.spot
        IV_grid = np.zeros_like(M_grid)

        for j, t_val in enumerate(t_vec):
            k_slice = m_vec * self.spot
            IV_grid[:, j] = self.get_iv(k_slice, t_val)

        TotalVar_grid = (IV_grid ** 2) * T_grid

        return {
            "moneyness_grid": M_grid,
            "expiry_grid": T_grid,
            "strike_grid": K_grid,
            "iv_grid": IV_grid,
            "total_variance_grid": TotalVar_grid,
            "moneyness_vec": m_vec,
            "expiry_vec": t_vec,
        }

    def dupire_local_volatility(
        self,
        strike: float,
        time_to_expiry: float,
        dk: float = 0.01,
        dt: float = 0.005,
    ) -> float:
        """Extracts Dupire Local Volatility sigma_local(K, T) via finite differences on total variance w(k, T)."""
        k0 = np.log(strike / self.spot)
        t0 = max(time_to_expiry, dt * 2)

        # Finite difference step
        k_up = strike * np.exp(dk)
        k_down = strike * np.exp(-dk)

        w_center = self.total_variance(strike, t0)
        w_t_up = self.total_variance(strike, t0 + dt)
        w_t_down = self.total_variance(strike, t0 - dt)

        w_k_up = self.total_variance(k_up, t0)
        w_k_down = self.total_variance(k_down, t0)

        # Derivatives w.r.t maturity T and log-moneyness k
        dw_dt = (w_t_up - w_t_down) / (2.0 * dt)
        dw_dk = (w_k_up - w_k_down) / (2.0 * dk)
        d2w_dk2 = (w_k_up - 2.0 * w_center + w_k_down) / (dk ** 2)

        # Dupire denominator formula in total variance terms
        denom = 1.0 - (k0 / w_center) * dw_dk + 0.25 * (-0.25 - 1.0 / w_center + (k0 ** 2) / (w_center ** 2)) * (dw_dk ** 2) + 0.5 * d2w_dk2
        if denom <= 0 or dw_dt <= 0:
            return float(self.get_iv(strike, t0))

        loc_var = dw_dt / denom
        return float(np.sqrt(max(loc_var, 1e-4)))

    @classmethod
    def create_synthetic_market_surface(
        cls,
        spot: float = 100.0,
        risk_free_rate: float = 0.03,
        dividend_yield: float = 0.01,
        atm_vol: float = 0.20,
        skew_slope: float = -0.15,
        convexity: float = 0.25,
        term_slope: float = 0.03,
    ) -> "VolatilitySurface":
        """Factory method generating a realistic equity-like volatility surface with negative skew and upward term structure."""
        surface = cls(spot=spot, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield)
        expiries = [1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0]
        moneyness_pts = np.linspace(0.65, 1.35, 15)
        strikes = moneyness_pts * spot

        for t in expiries:
            # Term structure dampening for skew: skew scales with 1 / sqrt(T)
            t_scale = 1.0 / np.sqrt(t)
            atm_t = atm_vol + term_slope * np.log(1.0 + t)

            k_log = np.log(moneyness_pts)
            # SVI style shape: ATM level + Skew * k + Convexity * k^2
            iv_slice = atm_t + (skew_slope * t_scale) * k_log + (convexity * t_scale) * (k_log ** 2)
            iv_slice = np.maximum(iv_slice, 0.05)

            smile = VolatilitySmile(
                spot=spot,
                time_to_expiry=t,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
            smile.fit_svi(strikes, iv_slice)
            surface.add_smile(t, smile)

        return surface
