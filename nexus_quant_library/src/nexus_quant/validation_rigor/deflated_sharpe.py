"""
The Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR) (Bailey & López de Prado 2014).
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PSRReport:
    observed_sharpe: float
    benchmark_sharpe: float
    psr_value: float
    sample_length: int
    skewness: float
    kurtosis: float
    is_significant_95: bool


@dataclass
class DSRReport:
    observed_sharpe: float
    expected_max_sharpe: float
    deflated_sharpe_ratio: float
    p_value: float
    num_trials: int
    effective_trials: float
    sample_length: int
    skewness: float
    kurtosis: float
    is_significant_95: bool
    is_significant_99: bool
    min_track_record_length_years: float

    def summary(self) -> str:
        status = "GENUINE ALPHA (Passed DSR > 0.95)" if self.is_significant_95 else "OVERFITTED / FALSE DISCOVERY (Failed DSR <= 0.95)"
        return f"""Deflated Sharpe Ratio (DSR) Audit Report:
  - Observed Strategy Sharpe:         {self.observed_sharpe:.3f}
  - Multiple Testing Trials (N):      {self.num_trials:,}
  - Expected Max Sharpe from Noise:   {self.expected_max_sharpe:.3f}
  - Deflated Sharpe Ratio (DSR):      {self.deflated_sharpe_ratio:.2%}
  - Statistical Significance:         {status}
  - Return Skewness:                  {self.skewness:+.2f}
  - Return Pearson Kurtosis:          {self.kurtosis:.2f}
  - Min Track Record Length Required: {self.min_track_record_length_years:.2f} years"""


class DeflatedSharpeRatioCalculator:
    """Computes the Deflated Sharpe Ratio (DSR) correcting for selection bias and non-normality."""

    EULER_MASCHERONI = 0.5772156649015328606

    @classmethod
    def compute_psr(
        cls,
        observed_sharpe: float,
        benchmark_sharpe: float = 0.0,
        sample_length: int = 252,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        periods_per_year: int = 252,
    ) -> PSRReport:
        sr = observed_sharpe / np.sqrt(periods_per_year)
        sr_bm = benchmark_sharpe / np.sqrt(periods_per_year)
        T = float(sample_length)

        # Standard error of Sharpe ratio under non-normality
        denom = np.sqrt(1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * (sr ** 2))
        z = ((sr - sr_bm) * np.sqrt(T - 1.0)) / max(denom, 1e-6)
        psr = float(stats.norm.cdf(z))

        return PSRReport(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            psr_value=psr,
            sample_length=sample_length,
            skewness=skewness,
            kurtosis=kurtosis,
            is_significant_95=psr >= 0.95,
        )

    @classmethod
    def expected_max_sharpe(
        cls,
        num_trials: int,
        var_sharpe_trials: float = 0.25,
        mean_sharpe_trials: float = 0.0,
        average_trial_correlation: float = 0.0,
    ) -> Tuple[float, float]:
        """Extreme value theory approximation of expected maximum Sharpe ratio under N trials."""
        n_eff = 1.0 + (num_trials - 1.0) * (1.0 - average_trial_correlation)
        n_eff = max(1.0, n_eff)
        sigma_sr = np.sqrt(var_sharpe_trials)

        if n_eff <= 1.0:
            return mean_sharpe_trials, 1.0

        z1 = stats.norm.ppf(1.0 - 1.0 / n_eff)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_eff * np.e))
        em_sr = mean_sharpe_trials + sigma_sr * ((1.0 - cls.EULER_MASCHERONI) * z1 + cls.EULER_MASCHERONI * z2)
        return float(em_sr), float(n_eff)

    @classmethod
    def min_track_record_length(
        cls,
        observed_sharpe: float,
        benchmark_sharpe: float = 0.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        confidence_level: float = 0.95,
        periods_per_year: int = 252,
    ) -> float:
        """Calculates Minimum Track Record Length (MinTRL) in years."""
        sr = observed_sharpe / np.sqrt(periods_per_year)
        sr_bm = benchmark_sharpe / np.sqrt(periods_per_year)
        if sr <= sr_bm:
            return np.inf

        z_alpha = stats.norm.ppf(confidence_level)
        var_sr = 1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * (sr ** 2)
        min_t_periods = 1.0 + var_sr * ((z_alpha / (sr - sr_bm)) ** 2)
        return float(min_t_periods / periods_per_year)

    def compute_dsr(
        self,
        best_sharpe_ratio: float,
        num_trials: int = 100,
        var_sharpe_trials: float = 0.25,
        sample_length: int = 252,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        periods_per_year: int = 252,
        average_trial_correlation: float = 0.0,
    ) -> DSRReport:
        e_max_sr, n_eff = self.expected_max_sharpe(
            num_trials=num_trials,
            var_sharpe_trials=var_sharpe_trials,
            average_trial_correlation=average_trial_correlation,
        )

        psr_rep = self.compute_psr(
            observed_sharpe=best_sharpe_ratio,
            benchmark_sharpe=e_max_sr,
            sample_length=sample_length,
            skewness=skewness,
            kurtosis=kurtosis,
            periods_per_year=periods_per_year,
        )

        min_trl = self.min_track_record_length(
            observed_sharpe=best_sharpe_ratio,
            benchmark_sharpe=0.0,
            skewness=skewness,
            kurtosis=kurtosis,
            periods_per_year=periods_per_year,
        )

        return DSRReport(
            observed_sharpe=best_sharpe_ratio,
            expected_max_sharpe=e_max_sr,
            deflated_sharpe_ratio=psr_rep.psr_value,
            p_value=1.0 - psr_rep.psr_value,
            num_trials=num_trials,
            effective_trials=n_eff,
            sample_length=sample_length,
            skewness=skewness,
            kurtosis=kurtosis,
            is_significant_95=psr_rep.psr_value >= 0.95,
            is_significant_99=psr_rep.psr_value >= 0.99,
            min_track_record_length_years=min_trl,
        )
