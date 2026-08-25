"""Green Bond Valuation & Greenium Decomposition Engine (Project 37).

Implements the institutional framework for green bond pricing and sustainability premia:
1. Matched-pair Twin Bond Greenium calculation ($\text{Greenium} = y_{\text{vanilla}} - y_{\text{green}}$ in bps).
2. Four-component yield spread decomposition (Maturity mismatch, Duration/Convexity, Liquidity, Pure Greenium).
3. Multivariate factor attribution regression (Liquidity differential, Credit rating, ESG score, Carbon intensity).
4. Nelson-Siegel parametric term structure calibration for the Greenium across tenors (1Y to 30Y).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t


@dataclass
class GreeniumDecompositionResult:
    """Dataclass holding granular spread decomposition for a single green bond pair."""
    pair_id: str
    issuer: str
    raw_spread_bps: float
    maturity_adjustment_bps: float
    duration_adjustment_bps: float
    liquidity_adjustment_bps: float
    pure_greenium_bps: float
    greenium_share_pct: float

    def summary(self) -> Dict[str, Any]:
        return {
            "Pair ID": self.pair_id,
            "Issuer": self.issuer,
            "Raw Spread (bps)": round(self.raw_spread_bps, 2),
            "Maturity Adj (bps)": round(self.maturity_adjustment_bps, 2),
            "Duration Adj (bps)": round(self.duration_adjustment_bps, 2),
            "Liquidity Adj (bps)": round(self.liquidity_adjustment_bps, 2),
            "Pure Fundamental Greenium (bps)": round(self.pure_greenium_bps, 2),
            "Pure Greenium Share (%)": f"{self.greenium_share_pct:.1f}%",
        }


@dataclass
class GreeniumFactorAttribution:
    """Dataclass holding multivariate regression attribution results."""
    coefficients: Dict[str, float]
    t_statistics: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_pvalue: float
    n_observations: int
    factor_importance_pct: Dict[str, float]

    def summary_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Factor": list(self.coefficients.keys()),
            "Coefficient": list(self.coefficients.values()),
            "t_Stat": list(self.t_statistics.values()),
            "p_Value": list(self.p_values.values()),
            "Importance_Pct": [self.factor_importance_pct.get(k, 0.0) for k in self.coefficients.keys()],
        })
        return df


@dataclass
class NelsonSiegelGreeniumFit:
    """Dataclass holding Nelson-Siegel term structure calibration parameters."""
    beta0_long_term: float
    beta1_short_spread: float
    beta2_curvature: float
    lambda_decay: float
    rmse_bps: float
    r_squared: float

    def predict(self, maturities: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        tau = np.asarray(maturities)
        lam = self.lambda_decay
        tau_scaled = np.maximum(tau / lam, 1e-6)
        
        factor1 = (1.0 - np.exp(-tau_scaled)) / tau_scaled
        factor2 = factor1 - np.exp(-tau_scaled)
        
        greenium = self.beta0_long_term + self.beta1_short_spread * factor1 + self.beta2_curvature * factor2
        if isinstance(maturities, (int, float)):
            return float(greenium)
        return greenium


class GreenBondValuationEngine:
    """Institutional Green Bond Pricing, Greenium Decomposition & Term Structure Engine."""

    def __init__(self, default_yield_curve_slope: float = 0.08):
        self.default_yield_curve_slope = default_yield_curve_slope

    # =========================================================================
    # 1. GREENIUM CALCULATION & SPREAD DECOMPOSITION
    # =========================================================================

    @staticmethod
    def compute_greenium_bps(
        vanilla_yield_pct: Union[float, np.ndarray, pd.Series],
        green_yield_pct: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Computes raw Greenium in basis points: Greenium = (y_vanilla - y_green) * 100."""
        return (vanilla_yield_pct - green_yield_pct) * 100.0

    def decompose_pair(
        self,
        pair_row: Union[pd.Series, Dict[str, Any]],
        yield_curve_slope_bps_per_yr: Optional[float] = None,
        duration_factor: float = 0.5,
    ) -> GreeniumDecompositionResult:
        """Decomposes matched-pair yield spread into fundamental drivers."""
        slope = yield_curve_slope_bps_per_yr if yield_curve_slope_bps_per_yr is not None else (self.default_yield_curve_slope * 100.0)

        # Extract values
        pair_id = str(pair_row.get("Pair_ID", "PAIR_001"))
        issuer = str(pair_row.get("Issuer", "Generic Issuer"))
        y_green = float(pair_row.get("Green_Yield_Pct", 3.0))
        y_vanilla = float(pair_row.get("Vanilla_Yield_Pct", 3.05))
        
        t_green = float(pair_row.get("Maturity_Years", 10.0))
        t_vanilla = float(pair_row.get("Vanilla_Maturity_Years", t_green))
        
        d_green = float(pair_row.get("Green_Duration", t_green * 0.88))
        d_vanilla = float(pair_row.get("Vanilla_Duration", t_vanilla * 0.87))
        
        ba_green = float(pair_row.get("Green_Bid_Ask_bps", 3.0))
        ba_vanilla = float(pair_row.get("Vanilla_Bid_Ask_bps", 2.2))

        # 1. Raw Spread
        raw_spread_bps = (y_vanilla - y_green) * 100.0

        # 2. Maturity adjustment (if maturities differ slightly)
        maturity_adj_bps = slope * (t_vanilla - t_green)

        # 3. Duration adjustment
        duration_adj_bps = duration_factor * (d_vanilla - d_green)

        # 4. Liquidity penalty (green bonds often have slightly wider bid-ask spreads)
        # Wider green bid-ask makes observed green yield artificially higher, so pure greenium is even higher
        liquidity_adj_bps = 0.5 * (ba_green - ba_vanilla)

        # 5. Pure Greenium
        pure_greenium_bps = raw_spread_bps - maturity_adj_bps - duration_adj_bps + liquidity_adj_bps
        pure_greenium_bps = max(0.1, pure_greenium_bps)
        
        share_pct = (pure_greenium_bps / (raw_spread_bps + 1e-6)) * 100.0 if raw_spread_bps > 0 else 100.0

        return GreeniumDecompositionResult(
            pair_id=pair_id,
            issuer=issuer,
            raw_spread_bps=round(raw_spread_bps, 2),
            maturity_adjustment_bps=round(maturity_adj_bps, 2),
            duration_adjustment_bps=round(duration_adj_bps, 2),
            liquidity_adjustment_bps=round(liquidity_adj_bps, 2),
            pure_greenium_bps=round(pure_greenium_bps, 2),
            greenium_share_pct=round(share_pct, 1),
        )

    def decompose_universe(self, df_pairs: pd.DataFrame) -> pd.DataFrame:
        """Decomposes greenium across an entire universe of twin bond pairs."""
        results = [self.decompose_pair(row).summary() for _, row in df_pairs.iterrows()]
        return pd.DataFrame(results)

    # =========================================================================
    # 2. MULTIVARIATE FACTOR ATTRIBUTION REGRESSION
    # =========================================================================

    def attribute_factors(self, df_pairs: pd.DataFrame) -> GreeniumFactorAttribution:
        """Performs OLS attribution regression of the Greenium against structural drivers.
        
        Greenium = α + β1*LiquidityDiff + β2*RatingScore + β3*ESGScore + β4*CarbonIntensity + β5*Maturity
        """
        df = df_pairs.copy()
        
        # Dependent variable: Raw Greenium in bps
        if "Raw_Greenium_bps" in df.columns:
            y = df["Raw_Greenium_bps"].values
        else:
            y = ((df["Vanilla_Yield_Pct"] - df["Green_Yield_Pct"]) * 100.0).values

        # Rating score numerical mapping
        rating_map = {"AAA": 10.0, "AA+": 9.0, "AA": 8.0, "AA-": 7.0, "A+": 6.0, "A": 5.0, "A-": 4.0, "BBB+": 3.0, "BBB": 2.0, "BBB-": 1.0}
        rating_num = df["Credit_Rating"].map(lambda r: rating_map.get(str(r), 5.0)).values

        # Feature matrix
        liq_diff = (df["Green_Bid_Ask_bps"] - df["Vanilla_Bid_Ask_bps"]).values
        esg_score = df.get("ESG_Score", pd.Series(85.0, index=df.index)).values
        carbon_int = df.get("Carbon_Intensity_tCO2_EUR_M", pd.Series(100.0, index=df.index)).values
        maturity = df.get("Maturity_Years", pd.Series(10.0, index=df.index)).values

        feature_names = ["Intercept", "Liquidity_Diff_bps", "Credit_Rating_Score", "ESG_Score", "Carbon_Intensity", "Maturity_Years"]
        
        N = len(y)
        X = np.column_stack([np.ones(N), liq_diff, rating_num, esg_score, carbon_int, maturity])
        
        # OLS estimation: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)

        y_pred = X @ beta
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-12))
        
        k = X.shape[1] - 1
        adj_r_squared = 1.0 - ((1.0 - r_squared) * (N - 1) / max(1, N - k - 1))
        
        # Standard errors & t-stats
        sigma_sq = ss_res / max(1, N - k - 1)
        cov_beta = sigma_sq * np.linalg.pinv(X.T @ X)
        se_beta = np.sqrt(np.maximum(1e-12, np.diag(cov_beta)))
        t_stats = beta / se_beta
        p_vals = [float(2.0 * (1.0 - student_t.cdf(abs(t), df=N - k - 1))) for t in t_stats]

        # F-statistic
        f_stat = ((ss_tot - ss_res) / k) / (ss_res / max(1, N - k - 1)) if ss_res > 0 else 0.0
        from scipy.stats import f as f_dist
        f_pvalue = float(1.0 - f_dist.cdf(f_stat, k, N - k - 1))

        # Relative feature importance
        importances = {}
        for name, b in zip(feature_names[1:], beta[1:]):
            importances[name] = round(float(abs(b) / (np.sum(np.abs(beta[1:])) + 1e-12) * 100.0), 1)

        coef_dict = {name: round(float(b), 4) for name, b in zip(feature_names, beta)}
        t_dict = {name: round(float(t), 2) for name, t in zip(feature_names, t_stats)}
        p_dict = {name: round(float(p), 4) for name, p in zip(feature_names, p_vals)}

        return GreeniumFactorAttribution(
            coefficients=coef_dict,
            t_statistics=t_dict,
            p_values=p_dict,
            r_squared=round(float(r_squared), 4),
            adjusted_r_squared=round(float(adj_r_squared), 4),
            f_statistic=round(float(f_stat), 2),
            f_pvalue=round(float(f_pvalue), 6),
            n_observations=N,
            factor_importance_pct=importances,
        )

    # =========================================================================
    # 3. NELSON-SIEGEL GREENIUM TERM STRUCTURE CALIBRATION
    # =========================================================================

    def fit_nelson_siegel_term_structure(
        self,
        maturities: Union[List[float], np.ndarray],
        greeniums_bps: Union[List[float], np.ndarray],
        fixed_lambda: Optional[float] = None,
    ) -> NelsonSiegelGreeniumFit:
        """Calibrates Nelson-Siegel parametric curve to the Greenium across maturities (1Y to 30Y).
        
        Greenium(τ) = β0 + β1 * ((1 - e^(-τ/λ)) / (τ/λ)) + β2 * (((1 - e^(-τ/λ)) / (τ/λ)) - e^(-τ/λ))
        """
        taus = np.asarray(maturities, dtype=float)
        y_obs = np.asarray(greeniums_bps, dtype=float)

        def objective(params):
            if fixed_lambda is not None:
                b0, b1, b2 = params
                lam = fixed_lambda
            else:
                b0, b1, b2, lam = params
                if lam <= 0.1 or lam > 30.0:
                    return 1e6

            tau_scaled = np.maximum(taus / lam, 1e-6)
            f1 = (1.0 - np.exp(-tau_scaled)) / tau_scaled
            f2 = f1 - np.exp(-tau_scaled)
            pred = b0 + b1 * f1 + b2 * f2
            return np.sum((y_obs - pred) ** 2)

        init_params = [5.0, -2.0, 1.5] if fixed_lambda is not None else [5.0, -2.0, 1.5, 3.0]
        bounds = [(0.0, 20.0), (-15.0, 15.0), (-15.0, 15.0)] if fixed_lambda is not None else [(0.0, 20.0), (-15.0, 15.0), (-15.0, 15.0), (0.5, 15.0)]

        res = minimize(objective, init_params, bounds=bounds, method="L-BFGS-B")
        
        if fixed_lambda is not None:
            b0, b1, b2 = res.x
            lam = fixed_lambda
        else:
            b0, b1, b2, lam = res.x

        fit_model = NelsonSiegelGreeniumFit(
            beta0_long_term=round(float(b0), 4),
            beta1_short_spread=round(float(b1), 4),
            beta2_curvature=round(float(b2), 4),
            lambda_decay=round(float(lam), 4),
            rmse_bps=round(float(np.sqrt(res.fun / len(taus))), 4),
            r_squared=round(float(1.0 - (res.fun / (np.sum((y_obs - np.mean(y_obs))**2) + 1e-12))), 4),
        )
        return fit_model
