"""
Risk: Value-at-Risk (VaR), Conditional VaR (Expected Shortfall) & Kupiec Backtesting.
"""

from typing import Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
import scipy.stats as stats


class RiskMetricsEngine:
    """Calculates Historical, Parametric, Cornish-Fisher and Monte Carlo VaR / CVaR."""

    @staticmethod
    def historical_var(returns: Union[pd.Series, np.ndarray], confidence: float = 0.95) -> float:
        """Non-parametric empirical quantile VaR (expressed as positive loss)."""
        clean = np.asarray(returns, dtype=float).flatten()
        clean = clean[~np.isnan(clean)]
        if len(clean) == 0:
            return 0.0
        q = np.percentile(clean, (1.0 - confidence) * 100.0)
        return float(-q)

    @staticmethod
    def historical_cvar(returns: Union[pd.Series, np.ndarray], confidence: float = 0.95) -> float:
        """Historical Expected Shortfall (CVaR)."""
        clean = np.asarray(returns, dtype=float).flatten()
        clean = clean[~np.isnan(clean)]
        if len(clean) == 0:
            return 0.0
        cutoff = np.percentile(clean, (1.0 - confidence) * 100.0)
        tail = clean[clean <= cutoff]
        return float(-np.mean(tail)) if len(tail) > 0 else float(-cutoff)

    @staticmethod
    def parametric_gaussian_var(
        returns: Union[pd.Series, np.ndarray], confidence: float = 0.95, horizon_days: int = 1
    ) -> float:
        """Gaussian Delta-Normal VaR."""
        clean = np.asarray(returns, dtype=float).flatten()
        clean = clean[~np.isnan(clean)]
        mu = np.mean(clean)
        sigma = np.std(clean, ddof=1)
        z = stats.norm.ppf(confidence)
        var_1d = -(mu - z * sigma)
        return float(var_1d * np.sqrt(horizon_days))

    @staticmethod
    def parametric_cornish_fisher_var(
        returns: Union[pd.Series, np.ndarray], confidence: float = 0.95
    ) -> float:
        """Cornish-Fisher expansion VaR adjusting for skewness and excess kurtosis."""
        clean = np.asarray(returns, dtype=float).flatten()
        clean = clean[~np.isnan(clean)]
        mu = np.mean(clean)
        sigma = np.std(clean, ddof=1)
        s = float(stats.skew(clean))
        k = float(stats.kurtosis(clean, fisher=True))

        z = stats.norm.ppf(confidence)
        z_cf = z + (1.0/6.0)*(z**2 - 1.0)*s + (1.0/24.0)*(z**3 - 3.0*z)*k - (1.0/36.0)*(2.0*z**3 - 5.0*z)*(s**2)

        return float(-(mu - z_cf * sigma))

    @staticmethod
    def monte_carlo_var_cvar(
        returns: Union[pd.Series, np.ndarray],
        confidence: float = 0.95,
        n_sims: int = 100000,
        horizon_days: int = 252,
    ) -> Tuple[float, float, np.ndarray]:
        """Simulate GBM price paths and calculate terminal VaR and CVaR."""
        clean = np.asarray(returns, dtype=float).flatten()
        clean = clean[~np.isnan(clean)]
        mu = float(np.mean(clean)) * 252.0
        sigma = float(np.std(clean, ddof=1)) * np.sqrt(252.0)

        dt = 1.0 / 252.0
        drift = (mu - 0.5 * sigma**2) * (horizon_days / 252.0)
        diffusion = sigma * np.sqrt(horizon_days / 252.0) * np.random.standard_normal(n_sims)

        terminal_values = np.exp(drift + diffusion)
        terminal_returns = terminal_values - 1.0

        q = np.percentile(terminal_returns, (1.0 - confidence) * 100.0)
        var_mc = float(-q)
        cvar_mc = float(-np.mean(terminal_returns[terminal_returns <= q]))

        return var_mc, cvar_mc, terminal_returns

    @staticmethod
    def kupiec_pof_test(
        returns: Union[pd.Series, np.ndarray], var_threshold: float, confidence: float = 0.95
    ) -> Tuple[int, float, float, bool]:
        """
        Kupiec Proportion of Failures (POF) Likelihood Ratio Test.
        Returns: (n_exceptions, failure_rate, lr_statistic, p_value, is_passed).
        """
        clean = np.asarray(returns, dtype=float).flatten()
        clean = clean[~np.isnan(clean)]
        n = len(clean)
        x = np.sum(clean < -abs(var_threshold))
        p = 1.0 - confidence

        if x == 0:
            return 0, 0.0, 0.0, 1.0, True

        pi = x / n
        num = (1.0 - p)**(n - x) * (p**x)
        den = (1.0 - pi)**(n - x) * (pi**x)
        lr = -2.0 * np.log(max(num / den, 1e-300))
        p_val = float(1.0 - stats.chi2.cdf(lr, df=1))
        is_passed = bool(p_val > 0.05)

        return int(x), float(pi), float(lr), p_val, is_passed
