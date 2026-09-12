"""
Fixed Income Term Structure, Nelson-Siegel, NSS, Zero-Coupon Bootstrapping & PCA.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


@dataclass
class NelsonSiegelParameters:
    """Parameters for the Nelson-Siegel (1987) Term Structure Model."""
    beta0: float  # Level (long-term rate asymptote)
    beta1: float  # Slope (short-to-long spread)
    beta2: float  # Curvature (medium-term hump)
    lambda_param: float  # Decay scale parameter

    def spot_rate(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate spot zero rate y(tau)."""
        t = np.maximum(tau, 1e-6)
        factor1 = (1.0 - np.exp(-t / self.lambda_param)) / (t / self.lambda_param)
        factor2 = factor1 - np.exp(-t / self.lambda_param)
        return self.beta0 + self.beta1 * factor1 + self.beta2 * factor2

    def forward_rate(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate instantaneous forward rate f(tau)."""
        t = np.maximum(tau, 1e-6)
        exp_term = np.exp(-t / self.lambda_param)
        return self.beta0 + self.beta1 * exp_term + self.beta2 * (t / self.lambda_param) * exp_term

    def discount_factor(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate discount factor P(0, tau) = exp(-y(tau)*tau)."""
        y = self.spot_rate(tau)
        return np.exp(-y * tau / 100.0)


@dataclass
class NSSParameters:
    """Parameters for the Nelson-Siegel-Svensson (1994) 6-Parameter Model."""
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lambda1: float
    lambda2: float

    def spot_rate(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate NSS spot zero rate y(tau)."""
        t = np.maximum(tau, 1e-6)
        factor1 = (1.0 - np.exp(-t / self.lambda1)) / (t / self.lambda1)
        factor2 = factor1 - np.exp(-t / self.lambda1)
        factor3 = (1.0 - np.exp(-t / self.lambda2)) / (t / self.lambda2) - np.exp(-t / self.lambda2)
        return self.beta0 + self.beta1 * factor1 + self.beta2 * factor2 + self.beta3 * factor3


class NelsonSiegelCalibrator:
    """Calibrates Nelson-Siegel and NSS term structure models to market yields."""

    @classmethod
    def fit_nelson_siegel(
        cls,
        maturities: np.ndarray,
        yields: np.ndarray,
    ) -> NelsonSiegelParameters:
        """Calibrate Nelson-Siegel parameters using non-linear least squares."""
        b0_init = float(yields[-1])
        b1_init = float(yields[0] - yields[-1])
        b2_init = float(2.0 * yields[len(yields)//2] - (yields[0] + yields[-1]))
        lam_init = 1.50
        init_guess = [b0_init, b1_init, b2_init, lam_init]

        def residuals(p: List[float]) -> np.ndarray:
            b0, b1, b2, lam = p
            t = np.maximum(maturities, 1e-6)
            f1 = (1.0 - np.exp(-t / lam)) / (t / lam)
            f2 = f1 - np.exp(-t / lam)
            model_y = b0 + b1 * f1 + b2 * f2
            return model_y - yields

        bounds = (
            [-10.0, -30.0, -30.0, 0.05],
            [30.0, 30.0, 30.0, 20.0],
        )

        res = least_squares(residuals, init_guess, bounds=bounds, ftol=1e-8, xtol=1e-8)
        b0, b1, b2, lam = res.x
        return NelsonSiegelParameters(beta0=float(b0), beta1=float(b1), beta2=float(b2), lambda_param=float(lam))

    @classmethod
    def fit_nss(
        cls,
        maturities: np.ndarray,
        yields: np.ndarray,
    ) -> NSSParameters:
        """Calibrate Nelson-Siegel-Svensson parameters."""
        init_guess = [float(yields[-1]), float(yields[0] - yields[-1]), 0.0, 0.0, 1.5, 5.0]

        def residuals(p: List[float]) -> np.ndarray:
            b0, b1, b2, b3, l1, l2 = p
            t = np.maximum(maturities, 1e-6)
            f1 = (1.0 - np.exp(-t / l1)) / (t / l1)
            f2 = f1 - np.exp(-t / l1)
            f3 = (1.0 - np.exp(-t / l2)) / (t / l2) - np.exp(-t / l2)
            model_y = b0 + b1 * f1 + b2 * f2 + b3 * f3
            return model_y - yields

        bounds = (
            [-10.0, -30.0, -30.0, -30.0, 0.05, 0.05],
            [30.0, 30.0, 30.0, 30.0, 20.0, 30.0],
        )

        res = least_squares(residuals, init_guess, bounds=bounds, ftol=1e-8, xtol=1e-8)
        b0, b1, b2, b3, l1, l2 = res.x
        return NSSParameters(
            beta0=float(b0),
            beta1=float(b1),
            beta2=float(b2),
            beta3=float(b3),
            lambda1=float(l1),
            lambda2=float(l2),
        )


@dataclass
class BootstrappedCurve:
    """Bootstrapped zero-coupon yield curve."""
    maturities: np.ndarray
    zero_rates: np.ndarray
    discount_factors: np.ndarray

    def get_discount_factor(self, tau: float) -> float:
        """Interpolate discount factor at arbitrary maturity."""
        pchip = PchipInterpolator(self.maturities, self.discount_factors)
        return float(np.clip(pchip(tau), 1e-5, 1.0))

    def get_zero_rate(self, tau: float) -> float:
        """Interpolate zero rate at arbitrary maturity."""
        pchip = PchipInterpolator(self.maturities, self.zero_rates)
        return float(pchip(tau))


class YieldCurveBootstrapper:
    """Recursively bootstraps zero-coupon spot rates from par coupon bonds."""

    @classmethod
    def bootstrap_par_yields(
        cls,
        maturities: np.ndarray,
        par_yields: np.ndarray,
        coupon_freq: int = 2,
    ) -> BootstrappedCurve:
        """
        Bootstrap discount factors P(0, T_n) and zero rates z(T_n) from par yields.
        Formula for par bond: 1 = C/m * sum_{k=1}^{m*T_n-1} P(0, t_k) + (1 + C/m) P(0, T_n).
        """
        n_tenors = len(maturities)
        discount_factors = np.zeros(n_tenors)
        zero_rates = np.zeros(n_tenors)

        known_maturities = []
        known_dfs = []

        for i in range(n_tenors):
            t_n = maturities[i]
            c_n = par_yields[i] / 100.0  # decimal coupon rate
            m = coupon_freq

            # Check if money market / zero coupon (maturity <= 1/m)
            if t_n <= (1.0 / m) + 1e-6:
                p_n = 1.0 / (1.0 + c_n * t_n)
            else:
                # All semi-annual coupon dates up to t_n - 1/m
                n_coupons = int(np.round(t_n * m))
                coupon_times = np.array([k / m for k in range(1, n_coupons)])

                if len(coupon_times) > 0 and len(known_maturities) > 0:
                    # Log-linear discount factor interpolation
                    interp_log_p = np.interp(
                        coupon_times,
                        np.array(known_maturities),
                        np.log(np.array(known_dfs)),
                    )
                    interp_dfs = np.exp(interp_log_p)
                    coupon_pv = (c_n / m) * np.sum(interp_dfs)
                else:
                    coupon_pv = 0.0

                p_n = (1.0 - coupon_pv) / (1.0 + (c_n / m))

            p_n = max(0.0001, min(1.0, float(p_n)))
            discount_factors[i] = p_n
            zero_rates[i] = (-np.log(p_n) / t_n) * 100.0

            known_maturities.append(t_n)
            known_dfs.append(p_n)

        return BootstrappedCurve(
            maturities=maturities,
            zero_rates=zero_rates,
            discount_factors=discount_factors,
        )


class YieldCurvePCA:
    """Performs Principal Component Analysis on daily sovereign yield curve shifts."""

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.mean_diff: Optional[np.ndarray] = None

    def fit(self, yield_matrix_df: pd.DataFrame) -> "YieldCurvePCA":
        """Fit PCA on daily yield changes delta y(tau)."""
        dy = yield_matrix_df.diff().dropna().values
        self.mean_diff = np.mean(dy, axis=0)
        dy_centered = dy - self.mean_diff

        cov_matrix = np.cov(dy_centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_matrix)

        # Sort descending
        idx = np.argsort(evals)[::-1]
        self.eigenvalues = evals[idx]
        self.eigenvectors = evecs[:, idx]
        total_var = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues[:self.n_components] / total_var

        # Sign convention: PC1 positive loading across tenors (Level)
        if np.mean(self.eigenvectors[:, 0]) < 0:
            self.eigenvectors[:, 0] *= -1.0
        # PC2 slope positive from short to long
        if self.eigenvectors[-1, 1] < self.eigenvectors[0, 1]:
            self.eigenvectors[:, 1] *= -1.0

        return self

    def summary(self) -> Dict[str, float]:
        """Return explained variance ratios for Level (PC1), Slope (PC2), and Curvature (PC3)."""
        if self.explained_variance_ratio is None:
            raise ValueError("Model not fitted.")
        return {
            "PC1_Level_Explained_Pct": float(self.explained_variance_ratio[0] * 100.0),
            "PC2_Slope_Explained_Pct": float(self.explained_variance_ratio[1] * 100.0),
            "PC3_Curvature_Explained_Pct": float(self.explained_variance_ratio[2] * 100.0),
            "Total_3PC_Explained_Pct": float(np.sum(self.explained_variance_ratio[:3]) * 100.0),
        }
