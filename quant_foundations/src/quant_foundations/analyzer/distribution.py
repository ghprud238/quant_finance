"""Statistical distribution analysis for asset returns.

Provides skewness, kurtosis, Jarque-Bera normality testing, and parametric distribution
fitting (Gaussian vs. Student-t) with goodness-of-fit metrics (Log-Likelihood, AIC, BIC, KS test).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def skewness(
    returns: Union[pd.Series, pd.DataFrame],
    bias: bool = False,
) -> Union[float, pd.Series]:
    r"""Calculate sample skewness of returns distribution.

    Skewness measures the degree of asymmetry:
    .. math::
        S = \frac{\frac{1}{N} \sum_{t=1}^N (R_t - \bar{R})^3}{\left( \frac{1}{N} \sum_{t=1}^N (R_t - \bar{R})^2 \right)^{3/2}}

    If `bias=False`, sample size correction is applied:
    .. math::
        S_{\text{unbiased}} = \frac{\sqrt{N(N-1)}}{N-2} S

    Parameters
    ----------
    returns : Union[pd.Series, pd.DataFrame]
        Asset return series or DataFrame.
    bias : bool, default False
        If False, computes unbiased sample skewness.

    Returns
    -------
    Union[float, pd.Series]
        Skewness value as float (for Series) or Series (for DataFrame).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: skewness(col, bias=bias))

    clean = returns.dropna()
    if len(clean) < 3:
        return np.nan
    return float(stats.skew(clean.values, bias=bias))


def kurtosis(
    returns: Union[pd.Series, pd.DataFrame],
    excess: bool = True,
    bias: bool = False,
) -> Union[float, pd.Series]:
    r"""Calculate kurtosis (or excess kurtosis) of returns distribution.

    Kurtosis measures the fatness of tails:
    .. math::
        K = \frac{\frac{1}{N} \sum_{t=1}^N (R_t - \bar{R})^4}{\left( \frac{1}{N} \sum_{t=1}^N (R_t - \bar{R})^2 \right)^2}

    If `excess=True`, returns :math:`K_{\text{excess}} = K - 3`.

    Parameters
    ----------
    returns : Union[pd.Series, pd.DataFrame]
        Asset return series or DataFrame.
    excess : bool, default True
        If True, returns excess kurtosis (Gaussian = 0). If False, returns raw kurtosis (Gaussian = 3).
    bias : bool, default False
        If False, computes sample-bias corrected kurtosis.

    Returns
    -------
    Union[float, pd.Series]
        Kurtosis value as float (for Series) or Series (for DataFrame).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: kurtosis(col, excess=excess, bias=bias))

    clean = returns.dropna()
    if len(clean) < 4:
        return np.nan
    return float(stats.kurtosis(clean.values, fisher=excess, bias=bias))


def jarque_bera_test(
    returns: Union[pd.Series, pd.DataFrame],
    alpha: float = 0.05,
) -> Union[Tuple[float, float, bool], pd.DataFrame]:
    r"""Perform Jarque-Bera test for normality.

    Tests the null hypothesis :math:`H_0` that the data is normally distributed based
    on matching theoretical skewness and kurtosis:
    .. math::
        JB = \frac{N}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right) \sim \chi^2(2)

    Parameters
    ----------
    returns : Union[pd.Series, pd.DataFrame]
        Asset return series or DataFrame.
    alpha : float, default 0.05
        Significance level for hypothesis test.

    Returns
    -------
    Union[Tuple[float, float, bool], pd.DataFrame]
        For Series: tuple of (jb_statistic, p_value, is_normal) where is_normal is True if p_value > alpha.
        For DataFrame: DataFrame with columns ['statistic', 'p_value', 'is_normal'].
    """
    if isinstance(returns, pd.DataFrame):
        results = []
        for col in returns.columns:
            stat, pval, is_norm = jarque_bera_test(returns[col], alpha=alpha)
            results.append({"ticker": col, "statistic": stat, "p_value": pval, "is_normal": is_norm})
        df_res = pd.DataFrame(results).set_index("ticker")
        return df_res

    clean = returns.dropna()
    if len(clean) < 8:
        return (np.nan, np.nan, False)

    jb_res = stats.jarque_bera(clean.values)
    stat = float(jb_res.statistic)
    p_val = float(jb_res.pvalue)
    is_normal = bool(p_val > alpha)
    return (stat, p_val, is_normal)


def fit_distributions(
    returns: Union[pd.Series, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Fit Gaussian and Student-t distributions to the return series and compare fit quality.

    Computes maximum likelihood parameters, Log-Likelihood, Akaike Information Criterion (AIC),
    Bayesian Information Criterion (BIC), and Kolmogorov-Smirnov (KS) test statistics.

    Parameters
    ----------
    returns : Union[pd.Series, np.ndarray]
        1D series or array of asset returns.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary containing fitted parameters and goodness-of-fit metrics for
        'gaussian' and 'student_t' distributions.
    """
    if isinstance(returns, pd.Series):
        data = returns.dropna().values
    else:
        data = np.asarray(returns)
        data = data[~np.isnan(data)]

    n = len(data)
    if n < 5:
        raise ValueError("At least 5 observations required to fit distributions.")

    # 1. Fit Gaussian (Normal) Distribution
    # Parameters: mean (mu), std (sigma)
    mu, sigma = stats.norm.fit(data)
    ll_norm = float(np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma)))
    k_norm = 2  # mu, sigma
    aic_norm = float(2 * k_norm - 2 * ll_norm)
    bic_norm = float(k_norm * np.log(n) - 2 * ll_norm)
    ks_norm = stats.kstest(data, "norm", args=(mu, sigma))

    # 2. Fit Student-t Distribution
    # Parameters: df (nu), loc (mu), scale (sigma)
    df_t, loc_t, scale_t = stats.t.fit(data)
    ll_t = float(np.sum(stats.t.logpdf(data, df=df_t, loc=loc_t, scale=scale_t)))
    k_t = 3  # df, loc, scale
    aic_t = float(2 * k_t - 2 * ll_t)
    bic_t = float(k_t * np.log(n) - 2 * ll_t)
    ks_t = stats.kstest(data, "t", args=(df_t, loc_t, scale_t))

    return {
        "gaussian": {
            "mean": float(mu),
            "std": float(sigma),
            "log_likelihood": ll_norm,
            "aic": aic_norm,
            "bic": bic_norm,
            "ks_stat": float(ks_norm.statistic),
            "ks_pvalue": float(ks_norm.pvalue),
        },
        "student_t": {
            "df": float(df_t),
            "loc": float(loc_t),
            "scale": float(scale_t),
            "log_likelihood": ll_t,
            "aic": aic_t,
            "bic": bic_t,
            "ks_stat": float(ks_t.statistic),
            "ks_pvalue": float(ks_t.pvalue),
        },
    }
