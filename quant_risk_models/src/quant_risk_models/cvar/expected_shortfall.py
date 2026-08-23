"""Expected Shortfall (Conditional Value-at-Risk) and Tail Risk Models.

Implements:
- Historical Expected Shortfall
- Parametric Gaussian Expected Shortfall
- Parametric Student's t Expected Shortfall
- Monte Carlo Expected Shortfall
- Component CVaR and Marginal Contribution to Risk (MCR)
- Statistical Backtesting (Kupiec POF & Christoffersen Independence Tests)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class KupiecBacktestResult:
    """Results from Kupiec Proportion of Failures (POF) Likelihood Ratio test."""
    confidence_level: float
    total_observations: int
    expected_exceptions: float
    actual_exceptions: int
    empirical_failure_rate: float
    lr_stat: float
    p_value: float
    is_rejected: bool  # True if model is rejected at alpha=0.05
    verdict: str


@dataclass
class ChristoffersenBacktestResult:
    """Results from Christoffersen Independence Likelihood Ratio test."""
    n00: int
    n01: int
    n10: int
    n11: int
    pi01: float
    pi11: float
    lr_ind_stat: float
    p_value_ind: float
    lr_cc_stat: float  # Combined conditional coverage (POF + Independence)
    p_value_cc: float
    is_rejected_ind: bool
    is_rejected_cc: bool
    verdict: str


@dataclass
class RiskBacktestReport:
    """Comprehensive VaR / ES backtest diagnostic report."""
    kupiec: KupiecBacktestResult
    christoffersen: ChristoffersenBacktestResult
    confidence_level: float
    exceptions_mask: pd.Series

    def summary(self) -> pd.DataFrame:
        data = {
            "Metric": [
                "Confidence Level",
                "Total Sample Size",
                "Expected Exceptions",
                "Actual Exceptions",
                "Failure Rate",
                "Kupiec POF LR Stat",
                "Kupiec p-value",
                "Kupiec Unconditional Coverage",
                "Christoffersen Ind LR Stat",
                "Christoffersen Ind p-value",
                "Independence Test",
                "Combined CC LR Stat",
                "Combined CC p-value",
                "Overall Backtest Verdict",
            ],
            "Value": [
                f"{self.confidence_level:.1%}",
                f"{self.kupiec.total_observations:,}",
                f"{self.kupiec.expected_exceptions:.1f}",
                f"{self.kupiec.actual_exceptions:,}",
                f"{self.kupiec.empirical_failure_rate:.2%}",
                f"{self.kupiec.lr_stat:.4f}",
                f"{self.kupiec.p_value:.4f}",
                "PASSED (Accurate Coverage)" if not self.kupiec.is_rejected else "FAILED (Inaccurate Coverage)",
                f"{self.christoffersen.lr_ind_stat:.4f}",
                f"{self.christoffersen.p_value_ind:.4f}",
                "PASSED (No Clustering)" if not self.christoffersen.is_rejected_ind else "FAILED (Clustered)",
                f"{self.christoffersen.lr_cc_stat:.4f}",
                f"{self.christoffersen.p_value_cc:.4f}",
                "PASSED (Valid Risk Model)" if (not self.kupiec.is_rejected and not self.christoffersen.is_rejected_ind) else "FAILED",
            ]
        }
        return pd.DataFrame(data)


@dataclass
class ComponentCVaRReport:
    """Multi-asset CVaR risk attribution report via Euler decomposition."""
    portfolio_cvar: float
    marginal_cvar: pd.Series
    component_cvar: pd.Series
    percentage_cvar: pd.Series
    weights: pd.Series
    method: str

    def summary_table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Weight": self.weights,
            "Marginal_CVaR": self.marginal_cvar,
            "Component_CVaR": self.component_cvar,
            "Pct_CVaR_Contribution": self.percentage_cvar * 100.0,
        })
        return df.sort_values(by="Component_CVaR", ascending=False)


class ExpectedShortfallModel:
    """Comprehensive Expected Shortfall (CVaR) and Value-at-Risk Engine."""

    def __init__(self, confidence_level: float = 0.95):
        if not (0.50 <= confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in [0.50, 1.0), got {confidence_level}")
        self.confidence_level = confidence_level

    def _prepare_returns(self, returns: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 5:
            raise ValueError(f"At least 5 valid return observations are required, got {len(arr)}")
        return arr

    # =========================================================================
    # 1. HISTORICAL ESTIMATORS
    # =========================================================================
    def historical_var(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: Optional[float] = None,
        as_loss: bool = True,
    ) -> float:
        """Calculates empirical Historical Value-at-Risk."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        alpha = 1.0 - cl
        return_var = float(np.percentile(arr, alpha * 100.0))
        return -return_var if as_loss else return_var

    def historical_es(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: Optional[float] = None,
        as_loss: bool = True,
    ) -> float:
        """Calculates empirical Historical Expected Shortfall (CVaR)."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        alpha = 1.0 - cl
        return_var = np.percentile(arr, alpha * 100.0)
        tail_returns = arr[arr <= return_var]
        if len(tail_returns) == 0:
            return_es = return_var
        else:
            return_es = float(np.mean(tail_returns))
        return -return_es if as_loss else return_es

    # =========================================================================
    # 2. PARAMETRIC GAUSSIAN ESTIMATORS
    # =========================================================================
    def parametric_gaussian_var(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: Optional[float] = None,
        as_loss: bool = True,
    ) -> float:
        """Calculates Parametric Gaussian Value-at-Risk."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        z = float(stats.norm.ppf(1.0 - cl))
        return_var = mu + z * sigma
        return -return_var if as_loss else return_var

    def parametric_gaussian_es(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: Optional[float] = None,
        as_loss: bool = True,
    ) -> float:
        """Calculates Parametric Gaussian Expected Shortfall."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        alpha = 1.0 - cl
        z_alpha = float(stats.norm.ppf(alpha))
        # E[R | R <= q] = mu - sigma * (phi(z_alpha) / alpha)
        ratio = float(stats.norm.pdf(z_alpha) / alpha)
        return_es = mu - sigma * ratio
        return -return_es if as_loss else return_es

    # =========================================================================
    # 3. PARAMETRIC STUDENT'S T ESTIMATORS
    # =========================================================================
    def parametric_student_t_var(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: Optional[float] = None,
        df: Optional[float] = None,
        as_loss: bool = True,
    ) -> float:
        """Calculates Parametric Student's t Value-at-Risk."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        if df is None:
            # Fit Student-t
            nu, mu, scale = stats.t.fit(arr)
            nu = max(nu, 2.01)
        else:
            nu = max(float(df), 2.01)
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1))
            scale = sigma * np.sqrt((nu - 2.0) / nu)

        alpha = 1.0 - cl
        t_alpha = float(stats.t.ppf(alpha, df=nu))
        return_var = mu + scale * t_alpha
        return -return_var if as_loss else return_var

    def parametric_student_t_es(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: Optional[float] = None,
        df: Optional[float] = None,
        as_loss: bool = True,
    ) -> float:
        """Calculates Parametric Student's t Expected Shortfall."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        if df is None:
            nu, mu, scale = stats.t.fit(arr)
            nu = max(nu, 2.01)
        else:
            nu = max(float(df), 2.01)
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1))
            scale = sigma * np.sqrt((nu - 2.0) / nu)

        alpha = 1.0 - cl
        t_alpha = float(stats.t.ppf(alpha, df=nu))
        pdf_val = float(stats.t.pdf(t_alpha, df=nu))
        factor = ((nu + t_alpha**2) / (nu - 1.0)) * (pdf_val / alpha)
        return_es = mu - scale * factor
        return -return_es if as_loss else return_es

    # =========================================================================
    # 4. CORNISH-FISHER EXPANSION
    # =========================================================================
    def cornish_fisher_var(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: Optional[float] = None,
        as_loss: bool = True,
    ) -> float:
        """Calculates Cornish-Fisher modified VaR adjusting for skewness and kurtosis."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        s = float(stats.skew(arr))
        k = float(stats.kurtosis(arr, fisher=True))  # excess kurtosis

        z = float(stats.norm.ppf(1.0 - cl))
        z_cf = (
            z
            + (1.0 / 6.0) * (z**2 - 1.0) * s
            + (1.0 / 24.0) * (z**3 - 3.0 * z) * k
            - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (s**2)
        )
        return_var = mu + z_cf * sigma
        return -return_var if as_loss else return_var

    # =========================================================================
    # 5. MONTE CARLO ESTIMATORS
    # =========================================================================
    def monte_carlo_es(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: Optional[float] = None,
        n_simulations: int = 100_000,
        distribution: str = "t",
        as_loss: bool = True,
        random_state: int = 42,
    ) -> Tuple[float, float]:
        """Calculates Monte Carlo Value-at-Risk and Expected Shortfall."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        rng = np.random.default_rng(random_state)

        if distribution.lower() == "t":
            nu, mu, scale = stats.t.fit(arr)
            sim_returns = stats.t.rvs(df=nu, loc=mu, scale=scale, size=n_simulations, random_state=rng)
        else:
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1))
            sim_returns = rng.normal(loc=mu, scale=sigma, size=n_simulations)

        alpha = 1.0 - cl
        return_var = float(np.percentile(sim_returns, alpha * 100.0))
        tail = sim_returns[sim_returns <= return_var]
        return_es = float(np.mean(tail)) if len(tail) > 0 else return_var

        if as_loss:
            return -return_var, -return_es
        return return_var, return_es

    # =========================================================================
    # 6. MULTI-ASSET COMPONENT CVaR / MCR (EULER DECOMPOSITION)
    # =========================================================================
    def component_cvar(
        self,
        returns_matrix: Union[pd.DataFrame, np.ndarray],
        weights: Optional[Union[pd.Series, np.ndarray, Dict[str, float]]] = None,
        confidence_level: Optional[float] = None,
        method: str = "historical",
    ) -> ComponentCVaRReport:
        """Computes Component CVaR and Marginal Contribution to Risk (MCR)."""
        cl = confidence_level or self.confidence_level

        if isinstance(returns_matrix, pd.DataFrame):
            cols = list(returns_matrix.columns)
            arr = returns_matrix.values
        else:
            arr = np.asarray(returns_matrix, dtype=float)
            cols = [f"Asset_{i}" for i in range(arr.shape[1])]

        n_obs, n_assets = arr.shape
        if weights is None:
            w = np.ones(n_assets) / n_assets
        elif isinstance(weights, dict):
            w = np.array([weights.get(c, 0.0) for c in cols], dtype=float)
        elif isinstance(weights, pd.Series):
            w = np.array([weights.get(c, 0.0) for c in cols], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)

        w = w / np.sum(w)
        w_series = pd.Series(w, index=cols)

        # Portfolio return series
        port_returns = arr @ w

        if method.lower() == "historical":
            alpha = 1.0 - cl
            var_thresh = np.percentile(port_returns, alpha * 100.0)
            tail_mask = port_returns <= var_thresh
            if np.sum(tail_mask) == 0:
                tail_mask = np.array([True] * len(port_returns))

            port_cvar = -float(np.mean(port_returns[tail_mask]))
            # MCR_i = -E[R_{i,t} | t in tail]
            mcr = -np.mean(arr[tail_mask, :], axis=0)

        elif method.lower() == "parametric":
            mu_vec = np.mean(arr, axis=0)
            cov_mat = np.cov(arr, rowvar=False)
            port_mu = float(w @ mu_vec)
            port_sigma = float(np.sqrt(w @ cov_mat @ w))

            alpha = 1.0 - cl
            z_alpha = float(stats.norm.ppf(alpha))
            ratio = float(stats.norm.pdf(z_alpha) / alpha)

            port_cvar = -port_mu + port_sigma * ratio
            # MCR = -mu + (cov @ w / port_sigma) * ratio
            marginal_vol = (cov_mat @ w) / max(port_sigma, 1e-12)
            mcr = -mu_vec + marginal_vol * ratio
        else:
            raise ValueError(f"Unsupported method '{method}'. Use 'historical' or 'parametric'.")

        mcr_series = pd.Series(mcr, index=cols)
        component_series = w_series * mcr_series
        pct_series = component_series / max(port_cvar, 1e-12)

        return ComponentCVaRReport(
            portfolio_cvar=port_cvar,
            marginal_cvar=mcr_series,
            component_cvar=component_series,
            percentage_cvar=pct_series,
            weights=w_series,
            method=method,
        )

    # =========================================================================
    # 7. STATISTICAL BACKTESTING (KUPIEC & CHRISTOFFERSEN)
    # =========================================================================
    def backtest_var(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        var_forecasts: Union[pd.Series, np.ndarray, List[float], float],
        confidence_level: Optional[float] = None,
    ) -> RiskBacktestReport:
        """Runs Kupiec POF and Christoffersen Independence Likelihood Ratio Tests."""
        cl = confidence_level or self.confidence_level
        arr = self._prepare_returns(returns)
        p = 1.0 - cl
        n = len(arr)

        if np.isscalar(var_forecasts):
            var_arr = np.full(n, float(var_forecasts))
        else:
            var_arr = np.asarray(var_forecasts, dtype=float)
            if len(var_arr) != n:
                raise ValueError(f"Length mismatch: returns ({n}) vs var_forecasts ({len(var_arr)})")

        # In standard loss notation, an exception is when Return <= -VaR (if VaR is positive)
        # or Return <= VaR (if VaR is negative return)
        if np.mean(var_arr) > 0:
            exceptions = (arr <= -var_arr).astype(int)
        else:
            exceptions = (arr <= var_arr).astype(int)

        x = int(np.sum(exceptions))
        p_hat = x / n

        # 1. Kupiec POF Test
        if x == 0:
            lr_pof = -2.0 * n * np.log(1.0 - p)
        elif x == n:
            lr_pof = -2.0 * n * np.log(p)
        else:
            # LR = 2 * [ x ln(p_hat/p) + (N-x) ln((1-p_hat)/(1-p)) ]
            lr_pof = 2.0 * (
                x * np.log(p_hat / p) + (n - x) * np.log((1.0 - p_hat) / (1.0 - p))
            )

        lr_pof = max(0.0, float(lr_pof))
        p_val_pof = float(1.0 - stats.chi2.cdf(lr_pof, df=1))
        is_rejected_pof = p_val_pof < 0.05
        verdict_pof = "REJECTED" if is_rejected_pof else "ACCEPT"

        kupiec_res = KupiecBacktestResult(
            confidence_level=cl,
            total_observations=n,
            expected_exceptions=n * p,
            actual_exceptions=x,
            empirical_failure_rate=p_hat,
            lr_stat=lr_pof,
            p_value=p_val_pof,
            is_rejected=is_rejected_pof,
            verdict=verdict_pof,
        )

        # 2. Christoffersen Independence Test
        # Markov transitions
        i_prev = exceptions[:-1]
        i_curr = exceptions[1:]

        n00 = int(np.sum((i_prev == 0) & (i_curr == 0)))
        n01 = int(np.sum((i_prev == 0) & (i_curr == 1)))
        n10 = int(np.sum((i_prev == 1) & (i_curr == 0)))
        n11 = int(np.sum((i_prev == 1) & (i_curr == 1)))

        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

        if pi01 == 0 or pi == 0 or (1.0 - pi01) == 0 or (1.0 - pi) == 0:
            lr_ind = 0.0
        else:
            term1 = n00 * np.log((1.0 - pi01) / (1.0 - pi)) if (1.0 - pi01) > 0 and (1.0 - pi) > 0 else 0.0
            term2 = n01 * np.log(pi01 / pi) if pi01 > 0 and pi > 0 else 0.0
            term3 = n10 * np.log((1.0 - pi11) / (1.0 - pi)) if (1.0 - pi11) > 0 and (1.0 - pi) > 0 and n10 > 0 else 0.0
            term4 = n11 * np.log(pi11 / pi) if pi11 > 0 and pi > 0 and n11 > 0 else 0.0
            lr_ind = 2.0 * (term1 + term2 + term3 + term4)

        lr_ind = max(0.0, float(lr_ind))
        p_val_ind = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
        is_rejected_ind = p_val_ind < 0.05
        verdict_ind = "REJECTED (Clustered)" if is_rejected_ind else "ACCEPT (Independent)"

        # Combined Conditional Coverage
        lr_cc = lr_pof + lr_ind
        p_val_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
        is_rejected_cc = p_val_cc < 0.05

        christoffersen_res = ChristoffersenBacktestResult(
            n00=n00,
            n01=n01,
            n10=n10,
            n11=n11,
            pi01=pi01,
            pi11=pi11,
            lr_ind_stat=lr_ind,
            p_value_ind=p_val_ind,
            lr_cc_stat=lr_cc,
            p_value_cc=p_val_cc,
            is_rejected_ind=is_rejected_ind,
            is_rejected_cc=is_rejected_cc,
            verdict=verdict_ind,
        )

        idx = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(len(returns))
        mask_series = pd.Series(exceptions.astype(bool), index=idx, name="VaR_Exception")

        return RiskBacktestReport(
            kupiec=kupiec_res,
            christoffersen=christoffersen_res,
            confidence_level=cl,
            exceptions_mask=mask_series,
        )
