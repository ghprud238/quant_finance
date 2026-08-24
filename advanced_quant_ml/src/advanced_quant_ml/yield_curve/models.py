"""Parametric Nelson-Siegel, Nelson-Siegel-Svensson, and Yield Curve Bootstrapping Engine."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.interpolate import CubicSpline, PchipInterpolator


@dataclass
class NelsonSiegelFitResult:
    """Container for calibrated Nelson-Siegel parameters and curve values."""
    beta0: float  # Level
    beta1: float  # Slope
    beta2: float  # Curvature
    lambda_decay: float  # Scale / Decay parameter
    rmse: float
    r_squared: float
    maturities_fitted: np.ndarray
    yields_actual: np.ndarray
    yields_fitted: np.ndarray
    short_rate: float
    long_rate: float

    def predict_yield(self, maturities: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        tau = np.asarray(maturities, dtype=float)
        tau_safe = np.maximum(tau, 1e-6)
        factor1 = (1.0 - np.exp(-tau_safe / self.lambda_decay)) / (tau_safe / self.lambda_decay)
        factor2 = factor1 - np.exp(-tau_safe / self.lambda_decay)
        y = self.beta0 + self.beta1 * factor1 + self.beta2 * factor2
        return float(y) if np.isscalar(maturities) else y

    def predict_forward(self, maturities: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        tau = np.asarray(maturities, dtype=float)
        tau_safe = np.maximum(tau, 1e-6)
        f = self.beta0 + self.beta1 * np.exp(-tau_safe / self.lambda_decay) + self.beta2 * (tau_safe / self.lambda_decay) * np.exp(-tau_safe / self.lambda_decay)
        return float(f) if np.isscalar(maturities) else f

    def summary(self) -> Dict[str, float]:
        return {
            "Level (Beta0)": self.beta0,
            "Slope (Beta1)": self.beta1,
            "Curvature (Beta2)": self.beta2,
            "Decay (Lambda)": self.lambda_decay,
            "Short Rate (0Y)": self.short_rate,
            "Long Rate (30Y+)": self.long_rate,
            "Fit RMSE (%)": self.rmse,
            "R_Squared": self.r_squared,
        }


@dataclass
class NSSFitResult:
    """Container for calibrated Nelson-Siegel-Svensson parameters."""
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lambda1: float
    lambda2: float
    rmse: float
    r_squared: float

    def predict_yield(self, maturities: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        tau = np.asarray(maturities, dtype=float)
        tau_safe = np.maximum(tau, 1e-6)
        f1 = (1.0 - np.exp(-tau_safe / self.lambda1)) / (tau_safe / self.lambda1)
        f2 = f1 - np.exp(-tau_safe / self.lambda1)
        f3 = (1.0 - np.exp(-tau_safe / self.lambda2)) / (tau_safe / self.lambda2) - np.exp(-tau_safe / self.lambda2)
        y = self.beta0 + self.beta1 * f1 + self.beta2 * f2 + self.beta3 * f3
        return float(y) if np.isscalar(maturities) else y

    def predict_forward(self, maturities: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        tau = np.asarray(maturities, dtype=float)
        tau_safe = np.maximum(tau, 1e-6)
        f = (self.beta0 + self.beta1 * np.exp(-tau_safe / self.lambda1) + 
             self.beta2 * (tau_safe / self.lambda1) * np.exp(-tau_safe / self.lambda1) + 
             self.beta3 * (tau_safe / self.lambda2) * np.exp(-tau_safe / self.lambda2))
        return float(f) if np.isscalar(maturities) else f


@dataclass
class BootstrapResult:
    """Container for bootstrapped zero-coupon yield and discount curves."""
    maturities: np.ndarray
    par_yields: np.ndarray
    zero_rates: np.ndarray
    discount_factors: np.ndarray

    def get_discount_factor(self, maturity: float) -> float:
        """Interpolates discount factor P(0, tau)."""
        if maturity <= 0:
            return 1.0
        interp = PchipInterpolator(self.maturities, np.log(self.discount_factors))
        return float(np.exp(interp(maturity)))

    def get_zero_rate(self, maturity: float) -> float:
        """Interpolates continuously compounded zero rate z(tau)."""
        if maturity <= 0:
            return float(self.zero_rates[0])
        df = self.get_discount_factor(maturity)
        return float(-np.log(df) / maturity)


class NelsonSiegelModel:
    """Nelson-Siegel Parametric Term Structure Model.
    
    y(tau) = beta0 + beta1 * ((1 - exp(-tau/lambda)) / (tau/lambda)) + beta2 * (((1 - exp(-tau/lambda)) / (tau/lambda)) - exp(-tau/lambda))
    """

    def __init__(self, lambda_decay: Optional[float] = None) -> None:
        self.fixed_lambda = lambda_decay

    @staticmethod
    def _basis_functions(tau: np.ndarray, lambda_decay: float) -> Tuple[np.ndarray, np.ndarray]:
        tau_safe = np.maximum(tau, 1e-6)
        x = tau_safe / lambda_decay
        factor1 = (1.0 - np.exp(-x)) / x
        factor2 = factor1 - np.exp(-x)
        return factor1, factor2

    def fit(self, maturities: Union[np.ndarray, List[float]], yields: Union[np.ndarray, List[float]]) -> NelsonSiegelFitResult:
        """Fits Nelson-Siegel parameters (beta0, beta1, beta2, lambda) using Non-Linear Least Squares."""
        mat = np.asarray(maturities, dtype=float)
        y_act = np.asarray(yields, dtype=float)

        if len(mat) < 4:
            raise ValueError("At least 4 yield curve tenors are required to calibrate Nelson-Siegel model.")

        # Bounds: beta0 > 0, lambda > 0.05
        # Initial guess
        b0_init = float(y_act[-1])
        b1_init = float(y_act[0] - y_act[-1])
        b2_init = float(2.0 * y_act[len(y_act)//2] - y_act[0] - y_act[-1])
        lam_init = 1.5 if self.fixed_lambda is None else self.fixed_lambda

        if self.fixed_lambda is not None:
            # Linear OLS for given lambda
            f1, f2 = self._basis_functions(mat, self.fixed_lambda)
            X = np.column_stack([np.ones_like(mat), f1, f2])
            betas, residuals, rank, s = np.linalg.lstsq(X, y_act, rcond=None)
            b0, b1, b2 = betas
            lam = self.fixed_lambda
        else:
            def residuals(params: np.ndarray) -> np.ndarray:
                b0, b1, b2, lam = params
                f1, f2 = self._basis_functions(mat, lam)
                y_pred = b0 + b1 * f1 + b2 * f2
                return y_pred - y_act

            init_p = [b0_init, b1_init, b2_init, lam_init]
            bounds = ([-1.0, -15.0, -15.0, 0.05], [20.0, 15.0, 15.0, 10.0])

            res = least_squares(residuals, init_p, bounds=bounds, ftol=1e-9, xtol=1e-9)
            b0, b1, b2, lam = res.x

        f1_fit, f2_fit = self._basis_functions(mat, lam)
        y_fit = b0 + b1 * f1_fit + b2 * f2_fit
        rmse = float(np.sqrt(np.mean((y_fit - y_act) ** 2)))
        ss_tot = np.sum((y_act - np.mean(y_act)) ** 2)
        ss_res = np.sum((y_act - y_fit) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))

        return NelsonSiegelFitResult(
            beta0=float(b0),
            beta1=float(b1),
            beta2=float(b2),
            lambda_decay=float(lam),
            rmse=rmse,
            r_squared=r2,
            maturities_fitted=mat,
            yields_actual=y_act,
            yields_fitted=y_fit,
            short_rate=float(b0 + b1),
            long_rate=float(b0),
        )

    def fit_time_series(self, yield_df: pd.DataFrame, maturities: np.ndarray) -> pd.DataFrame:
        """Calibrates Nelson-Siegel parameters for every historical date in yield_df."""
        results = []
        for date, row in yield_df.iterrows():
            y_arr = row.values.astype(float)
            fit_res = self.fit(maturities, y_arr)
            results.append({
                "Date": date,
                "Level_Beta0": fit_res.beta0,
                "Slope_Beta1": fit_res.beta1,
                "Curvature_Beta2": fit_res.beta2,
                "Lambda": fit_res.lambda_decay,
                "Short_Rate": fit_res.short_rate,
                "Long_Rate": fit_res.long_rate,
                "RMSE": fit_res.rmse,
                "R_Squared": fit_res.r_squared,
            })
        df_ts = pd.DataFrame(results).set_index("Date")
        return df_ts


class NelsonSiegelSvenssonModel:
    """Nelson-Siegel-Svensson (NSS) 6-parameter Term Structure Model."""

    @staticmethod
    def _basis_functions(tau: np.ndarray, lam1: float, lam2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau_safe = np.maximum(tau, 1e-6)
        x1 = tau_safe / lam1
        x2 = tau_safe / lam2
        f1 = (1.0 - np.exp(-x1)) / x1
        f2 = f1 - np.exp(-x1)
        f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
        return f1, f2, f3

    def fit(self, maturities: Union[np.ndarray, List[float]], yields: Union[np.ndarray, List[float]]) -> NSSFitResult:
        mat = np.asarray(maturities, dtype=float)
        y_act = np.asarray(yields, dtype=float)

        if len(mat) < 6:
            raise ValueError("At least 6 yield curve tenors are required to calibrate NSS model.")

        def residuals(params: np.ndarray) -> np.ndarray:
            b0, b1, b2, b3, lam1, lam2 = params
            f1, f2, f3 = self._basis_functions(mat, lam1, lam2)
            y_pred = b0 + b1 * f1 + b2 * f2 + b3 * f3
            return y_pred - y_act

        init_p = [float(y_act[-1]), float(y_act[0] - y_act[-1]), 0.0, 0.0, 1.5, 5.0]
        bounds = ([-2.0, -15.0, -15.0, -15.0, 0.1, 0.1], [25.0, 15.0, 15.0, 15.0, 15.0, 20.0])

        res = least_squares(residuals, init_p, bounds=bounds, ftol=1e-9, xtol=1e-9)
        b0, b1, b2, b3, lam1, lam2 = res.x

        f1, f2, f3 = self._basis_functions(mat, lam1, lam2)
        y_fit = b0 + b1 * f1 + b2 * f2 + b3 * f3
        rmse = float(np.sqrt(np.mean((y_fit - y_act) ** 2)))
        ss_tot = np.sum((y_act - np.mean(y_act)) ** 2)
        ss_res = np.sum((y_act - y_fit) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))

        return NSSFitResult(
            beta0=float(b0),
            beta1=float(b1),
            beta2=float(b2),
            beta3=float(b3),
            lambda1=float(lam1),
            lambda2=float(lam2),
            rmse=rmse,
            r_squared=r2,
        )


class YieldCurveBootstrapper:
    """Bootstraps Zero-Coupon Spot Rates and Discount Factors from Par Yields."""

    @staticmethod
    def bootstrap_par_yields(maturities: np.ndarray, par_yields: np.ndarray, coupon_freq: int = 2) -> BootstrapResult:
        """Bootstraps spot zero rates z(tau) from a curve of par yields (in percent)."""
        mat = np.asarray(maturities, dtype=float)
        # Convert percent to decimal
        y_par = np.asarray(par_yields, dtype=float) / 100.0
        n_tenors = len(mat)

        discount_factors = np.zeros(n_tenors)
        zero_rates = np.zeros(n_tenors)

        for i in range(n_tenors):
            tau = mat[i]
            c = y_par[i]

            if tau <= 1.0 / coupon_freq:
                # Money market / short zero coupon rate: P(0, tau) = 1 / (1 + c * tau)
                df = 1.0 / (1.0 + c * tau)
            else:
                # Semi-annual / annual coupon bond priced at par: 1.0 = (c/freq) * sum(P) + (1 + c/freq)*P(tau)
                dt = 1.0 / coupon_freq
                n_coupons = int(round(tau * coupon_freq))
                coupon_times = np.linspace(dt, tau, n_coupons)

                # Interpolate previously solved discount factors
                if i > 0:
                    prev_interp = PchipInterpolator(mat[:i], np.log(discount_factors[:i]))
                    # coupon payments before tau
                    prev_times = coupon_times[:-1]
                    prev_dfs = np.exp(prev_interp(prev_times))
                    sum_pv_coupons = (c / coupon_freq) * np.sum(prev_dfs)
                else:
                    sum_pv_coupons = 0.0

                df = (1.0 - sum_pv_coupons) / (1.0 + c / coupon_freq)

            df = max(df, 1e-6)
            discount_factors[i] = df
            # Continuously compounded zero rate (in percent)
            zero_rates[i] = (-np.log(df) / tau) * 100.0

        return BootstrapResult(
            maturities=mat,
            par_yields=np.asarray(par_yields, dtype=float),
            zero_rates=zero_rates,
            discount_factors=discount_factors,
        )


class YieldCurvePCA:
    """Principal Component Analysis on daily yield curve changes (Level, Slope, Curvature)."""

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None

    def fit(self, yield_df: pd.DataFrame) -> 'YieldCurvePCA':
        """Fits PCA on daily yield changes delta y(tau)."""
        delta_y = yield_df.diff().dropna()
        cov_matrix = np.cov(delta_y.values, rowvar=False)

        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        # Sort descending
        idx = np.argsort(eigenvals)[::-1]
        self.eigenvalues = eigenvals[idx][:self.n_components]
        self.eigenvectors = eigenvecs[:, idx][:, :self.n_components]

        # Ensure consistent sign convention:
        # PC1 (Level): all positive loadings
        if np.sum(self.eigenvectors[:, 0]) < 0:
            self.eigenvectors[:, 0] *= -1.0
        # PC2 (Slope): short tenor positive, long tenor negative
        if self.eigenvectors[0, 1] < 0:
            self.eigenvectors[:, 1] *= -1.0

        total_var = np.sum(eigenvals)
        self.explained_variance_ratio = self.eigenvalues / total_var
        return self

    def summary(self) -> pd.DataFrame:
        if self.explained_variance_ratio is None:
            raise RuntimeError("PCA not fitted yet.")
        labels = ["PC1 (Level Shift)", "PC2 (Slope Tilt)", "PC3 (Curvature Twist)"][:self.n_components]
        return pd.DataFrame({
            "Factor": labels,
            "Eigenvalue": self.eigenvalues,
            "Variance_Explained_Pct": self.explained_variance_ratio * 100.0,
            "Cumulative_Pct": np.cumsum(self.explained_variance_ratio) * 100.0,
        }).set_index("Factor")
