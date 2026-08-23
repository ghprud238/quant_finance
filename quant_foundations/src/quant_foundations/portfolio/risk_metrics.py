"""Portfolio Risk Metrics Engine.

This module provides quantitative risk, return, and performance metrics
for asset and portfolio evaluation, including drawdown analytics,
Value at Risk (VaR), Conditional Value at Risk (CVaR/Expected Shortfall),
risk-adjusted return ratios, and benchmark-relative metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


def _to_series(data: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]], name: str = "returns") -> pd.Series:
    """Convert input return series to a 1D clean pandas Series of floats."""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            series = data.iloc[:, 0].astype(float)
        else:
            raise ValueError(f"Expected single-column return data, got DataFrame with shape {data.shape}")
    elif isinstance(data, pd.Series):
        series = data.astype(float)
    elif isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=float).squeeze()
        if arr.ndim > 1:
            raise ValueError(f"Expected 1D array, got shape {arr.shape}")
        series = pd.Series(arr, name=name)
    elif isinstance(data, (list, tuple)):
        series = pd.Series(list(data), dtype=float, name=name)
    else:
        series = pd.Series(data, dtype=float, name=name)

    # Drop NaNs or infinite values from the series
    series = series.dropna()
    return series


def _align_series(
    portfolio_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    benchmark_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
) -> Tuple[pd.Series, pd.Series]:
    """Align portfolio and benchmark returns on shared indices or matching lengths."""
    rp = _to_series(portfolio_returns, name="portfolio")
    rb = _to_series(benchmark_returns, name="benchmark")

    # If both have meaningful indices that overlap
    if isinstance(rp.index, pd.Index) and isinstance(rb.index, pd.Index):
        common_idx = rp.index.intersection(rb.index)
        if len(common_idx) > 0 and not (isinstance(rp.index, pd.RangeIndex) and isinstance(rb.index, pd.RangeIndex)):
            rp_aligned = rp.loc[common_idx]
            rb_aligned = rb.loc[common_idx]
            return rp_aligned, rb_aligned

    # Fallback to position-based truncation if lengths differ
    min_len = min(len(rp), len(rb))
    if min_len == 0:
        return rp.iloc[:0], rb.iloc[:0]
    return rp.iloc[-min_len:], rb.iloc[-min_len:]


def annualized_return(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    periods_per_year: int = 252,
    geometric: bool = True,
) -> float:
    """Calculate the annualized return (CAGR or arithmetic).

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns (e.g. daily simple returns).
    periods_per_year : int, default 252
        Number of periods in a full trading year (252 for daily, 52 for weekly, 12 for monthly).
    geometric : bool, default True
        If True, calculates the Compound Annual Growth Rate (CAGR).
        If False, calculates arithmetic annualized return.

    Returns
    -------
    float
        Annualized return as a decimal (e.g. 0.125 for 12.5%).
    """
    r = _to_series(returns)
    n = len(r)
    if n == 0:
        return 0.0

    if geometric:
        wealth_factor = float(np.prod(1.0 + r.values))
        if wealth_factor <= 0.0:
            return -1.0
        return float(wealth_factor ** (periods_per_year / n) - 1.0)
    else:
        return float(r.mean() * periods_per_year)


def annualized_volatility(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """Calculate the annualized volatility (sample standard deviation scaled by sqrt(T)).

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    periods_per_year : int, default 252
        Number of periods per year.
    ddof : int, default 1
        Delta Degrees of Freedom.

    Returns
    -------
    float
        Annualized volatility as a decimal.
    """
    r = _to_series(returns)
    if len(r) <= ddof:
        return 0.0
    sample_std = float(r.std(ddof=ddof))
    if abs(sample_std) < 1e-14 or np.isnan(sample_std):
        return 0.0
    return float(sample_std * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sharpe Ratio.

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate (e.g. 0.02 for 2%).
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Annualized Sharpe Ratio.
    """
    r = _to_series(returns)
    if len(r) < 2:
        return 0.0

    rf_periodic = risk_free_rate / periods_per_year
    excess_returns = r - rf_periodic
    vol = float(r.std(ddof=1))
    if vol <= 1e-14 or np.isnan(vol):
        return 0.0

    return float((excess_returns.mean() / vol) * np.sqrt(periods_per_year))


def downside_deviation(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    target_return: float = 0.0,
    periods_per_year: int = 252,
    ddof: int = 0,
) -> float:
    """Calculate annualized downside deviation (semi-deviation below target).

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    target_return : float, default 0.0
        Annualized Minimum Acceptable Return (MAR) or target return.
    periods_per_year : int, default 252
        Number of periods per year.
    ddof : int, default 0
        Degrees of freedom adjustment.

    Returns
    -------
    float
        Annualized downside deviation.
    """
    r = _to_series(returns)
    n = len(r)
    if n <= ddof:
        return 0.0

    target_periodic = target_return / periods_per_year
    underperformance = np.minimum(r.values - target_periodic, 0.0)
    downside_variance = np.sum(underperformance ** 2) / (n - ddof)
    if downside_variance <= 1e-14:
        return 0.0
    return float(np.sqrt(downside_variance) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sortino Ratio.

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate.
    target_return : float, default 0.0
        Annualized target return / Minimum Acceptable Return (MAR).
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Annualized Sortino Ratio.
    """
    r = _to_series(returns)
    if len(r) < 2:
        return 0.0

    rf_periodic = risk_free_rate / periods_per_year
    excess_mean_annualized = float((r.mean() - rf_periodic) * periods_per_year)
    d_dev = downside_deviation(r, target_return=target_return, periods_per_year=periods_per_year)

    if d_dev <= 1e-14:
        if excess_mean_annualized > 0:
            return float("inf")
        elif excess_mean_annualized < 0:
            return float("-inf")
        return 0.0

    return float(excess_mean_annualized / d_dev)


def drawdown_series(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
) -> pd.DataFrame:
    """Calculate the wealth index, high-water mark, drawdown series, and durations.

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'cumulative_returns': Wealth index minus 1 (or compound total return)
        - 'high_water_mark': Peak cumulative wealth factor
        - 'drawdown_pct': Current drawdown percentage (values <= 0.0)
        - 'drawdown_duration': Current number of periods in drawdown
        - 'max_drawdown': Overall maximum drawdown percentage across series
        - 'max_drawdown_duration': Maximum contiguous drawdown duration (periods)
    """
    r = _to_series(returns)
    if len(r) == 0:
        return pd.DataFrame(
            columns=[
                "cumulative_returns",
                "high_water_mark",
                "drawdown_pct",
                "drawdown_duration",
                "max_drawdown",
                "max_drawdown_duration",
            ]
        )

    wealth_index = (1.0 + r).cumprod()
    high_water_mark = wealth_index.cummax()
    drawdown_pct = (wealth_index - high_water_mark) / high_water_mark

    # Calculate continuous drawdown duration in periods
    durations = []
    current_dur = 0
    for dd in drawdown_pct.values:
        if dd < 0.0:
            current_dur += 1
        else:
            current_dur = 0
        durations.append(current_dur)

    duration_series = pd.Series(durations, index=r.index, dtype=int)
    max_dd = float(drawdown_pct.min()) if len(drawdown_pct) > 0 else 0.0
    max_dur = int(duration_series.max()) if len(duration_series) > 0 else 0

    df = pd.DataFrame(
        {
            "cumulative_returns": wealth_index - 1.0,
            "high_water_mark": high_water_mark,
            "drawdown_pct": drawdown_pct,
            "drawdown_duration": duration_series,
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dur,
        },
        index=r.index,
    )
    return df


def max_drawdown(returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]]) -> float:
    """Calculate the maximum drawdown percentage (as a signed non-positive float, e.g. -0.15 for -15%)."""
    dd_df = drawdown_series(returns)
    if dd_df.empty:
        return 0.0
    return float(dd_df["drawdown_pct"].min())


def max_drawdown_duration(returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]]) -> int:
    """Calculate the longest contiguous drawdown duration in periods."""
    dd_df = drawdown_series(returns)
    if dd_df.empty:
        return 0
    return int(dd_df["drawdown_duration"].max())


def calmar_ratio(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    periods_per_year: int = 252,
) -> float:
    """Calculate the Calmar Ratio (Annualized Return / Absolute Max Drawdown).

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Calmar Ratio.
    """
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    mdd = abs(max_drawdown(returns))

    if mdd <= 1e-14:
        if ann_ret > 0:
            return float("inf")
        elif ann_ret < 0:
            return float("-inf")
        return 0.0

    return float(ann_ret / mdd)


def skewness(returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]]) -> float:
    """Calculate sample skewness (3rd standardized moment)."""
    r = _to_series(returns)
    if len(r) < 3:
        return 0.0
    return float(stats.skew(r.values, bias=False))


def excess_kurtosis(returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]]) -> float:
    """Calculate sample excess kurtosis (Fisher definition where normal = 0.0)."""
    r = _to_series(returns)
    if len(r) < 4:
        return 0.0
    return float(stats.kurtosis(r.values, fisher=True, bias=False))


def value_at_risk(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    confidence_level: float = 0.95,
    method: str = "historical",
    **kwargs: Any,
) -> float:
    """Calculate Value at Risk (VaR) at a given confidence level.

    The return cutoff is returned (e.g. -0.025 for a 2.5% loss threshold).

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    confidence_level : float, default 0.95
        Confidence level (e.g. 0.95 for 95% VaR).
    method : {'historical', 'parametric', 'cornish_fisher', 'monte_carlo'}, default 'historical'
        Method of VaR computation:
        - 'historical': Empirical quantile at (1 - confidence_level).
        - 'parametric': Gaussian parametric VaR: mu - z_alpha * sigma.
        - 'cornish_fisher': Modified VaR adjusting for skewness S and excess kurtosis K:
          z_tilde = z + 1/6*(z^2 - 1)*S + 1/24*(z^3 - 3z)*K - 1/36*(2z^3 - 5z)*S^2
          VaR = mu - z_tilde * sigma.
        - 'monte_carlo': Simulated paths from normal or Student-t distribution.
    **kwargs : Any
        Additional options for Monte Carlo:
        - `n_simulations`: int, default 100,000
        - `dist`: {'normal', 't'}, default 'normal'
        - `random_state`: Optional[int], seed for reproducibility

    Returns
    -------
    float
        Value at Risk cutoff return (typically negative).
    """
    r = _to_series(returns)
    if len(r) == 0:
        return 0.0

    method_clean = method.lower().strip()
    alpha = 1.0 - confidence_level

    if method_clean in ("historical", "hist"):
        return float(np.percentile(r.values, alpha * 100.0))

    mu = float(r.mean())
    sigma = float(r.std(ddof=1)) if len(r) > 1 else 0.0

    if sigma <= 1e-14:
        return mu

    z = float(stats.norm.ppf(confidence_level))

    if method_clean in ("parametric", "gaussian"):
        return float(mu - z * sigma)

    elif method_clean in ("cornish_fisher", "modified", "cf"):
        s = float(stats.skew(r.values, bias=False)) if len(r) >= 3 else 0.0
        k = float(stats.kurtosis(r.values, fisher=True, bias=False)) if len(r) >= 4 else 0.0

        z_tilde = (
            z
            + (1.0 / 6.0) * (z**2 - 1.0) * s
            + (1.0 / 24.0) * (z**3 - 3.0 * z) * k
            - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (s**2)
        )
        return float(mu - z_tilde * sigma)

    elif method_clean in ("monte_carlo", "mc"):
        n_sim = int(kwargs.get("n_simulations", 100_000))
        dist_type = str(kwargs.get("dist", "normal")).lower()
        random_state = kwargs.get("random_state", None)
        rng = np.random.default_rng(random_state)

        if dist_type in ("t", "student_t"):
            if len(r) >= 4:
                df_param, loc_param, scale_param = stats.t.fit(r.values)
                sim_returns = stats.t.rvs(
                    df=df_param,
                    loc=loc_param,
                    scale=scale_param,
                    size=n_sim,
                    random_state=rng,
                )
            else:
                sim_returns = rng.normal(mu, sigma, size=n_sim)
        else:
            sim_returns = rng.normal(mu, sigma, size=n_sim)

        return float(np.percentile(sim_returns, alpha * 100.0))

    else:
        raise ValueError(
            f"Unknown VaR method '{method}'. Supported methods: 'historical', 'parametric', 'cornish_fisher', 'monte_carlo'"
        )


def conditional_value_at_risk(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    confidence_level: float = 0.95,
    method: str = "historical",
    **kwargs: Any,
) -> float:
    """Calculate Conditional Value at Risk (CVaR / Expected Shortfall: E[R | R <= VaR]).

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    confidence_level : float, default 0.95
        Confidence level.
    method : {'historical', 'parametric'}, default 'historical'
        - 'historical': Mean of returns less than or equal to historical VaR.
        - 'parametric': Analytical Gaussian Expected Shortfall: mu - sigma * (phi(z) / (1 - confidence_level)).
    **kwargs : Any
        Passed to underlying VaR calculation if applicable.

    Returns
    -------
    float
        Conditional Value at Risk (Expected Shortfall) return.
    """
    r = _to_series(returns)
    if len(r) == 0:
        return 0.0

    method_clean = method.lower().strip()
    if method_clean in ("historical", "hist"):
        var = value_at_risk(r, confidence_level=confidence_level, method="historical")
        tail = r[r <= var]
        if len(tail) == 0:
            return var
        return float(tail.mean())

    elif method_clean in ("parametric", "gaussian"):
        mu = float(r.mean())
        sigma = float(r.std(ddof=1)) if len(r) > 1 else 0.0
        if sigma <= 1e-14:
            return mu
        z = float(stats.norm.ppf(confidence_level))
        alpha = 1.0 - confidence_level
        pdf_z = float(stats.norm.pdf(z))
        es = mu - sigma * (pdf_z / alpha)
        return float(es)

    else:
        raise ValueError(f"Unknown CVaR method '{method}'. Supported methods: 'historical', 'parametric'")


def realized_beta(
    portfolio_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    benchmark_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
) -> float:
    """Calculate Realized Beta: Cov(R_p, R_b) / Var(R_b).

    Parameters
    ----------
    portfolio_returns : Series, DataFrame, array, or sequence
        Portfolio periodic returns.
    benchmark_returns : Series, DataFrame, array, or sequence
        Benchmark periodic returns.

    Returns
    -------
    float
        Realized Beta.
    """
    rp, rb = _align_series(portfolio_returns, benchmark_returns)
    if len(rp) < 2 or len(rb) < 2:
        return 0.0

    var_b = float(np.var(rb.values, ddof=1))
    if var_b <= 1e-14 or np.isnan(var_b):
        return 0.0

    cov = float(np.cov(rp.values, rb.values, ddof=1)[0, 1])
    return float(cov / var_b)


def jensens_alpha(
    portfolio_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    benchmark_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Annualized Jensen's Alpha relative to a benchmark.

    Alpha = (mean(R_p) - rf_p)*periods_per_year - Beta * (mean(R_b) - rf_p)*periods_per_year.

    Parameters
    ----------
    portfolio_returns : Series, DataFrame, array, or sequence
        Portfolio periodic returns.
    benchmark_returns : Series, DataFrame, array, or sequence
        Benchmark periodic returns.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate.
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Annualized Jensen's Alpha.
    """
    rp, rb = _align_series(portfolio_returns, benchmark_returns)
    if len(rp) < 2:
        return 0.0

    beta = realized_beta(rp, rb)
    rf_periodic = risk_free_rate / periods_per_year

    rp_excess_mean = float(rp.mean() - rf_periodic)
    rb_excess_mean = float(rb.mean() - rf_periodic)

    alpha_periodic = rp_excess_mean - beta * rb_excess_mean
    return float(alpha_periodic * periods_per_year)


def tracking_error(
    portfolio_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    benchmark_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    periods_per_year: int = 252,
) -> float:
    """Calculate Annualized Tracking Error: std(R_p - R_b) * sqrt(periods_per_year).

    Parameters
    ----------
    portfolio_returns : Series, DataFrame, array, or sequence
        Portfolio periodic returns.
    benchmark_returns : Series, DataFrame, array, or sequence
        Benchmark periodic returns.
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Annualized tracking error.
    """
    rp, rb = _align_series(portfolio_returns, benchmark_returns)
    if len(rp) < 2:
        return 0.0

    diff = rp.values - rb.values
    diff_std = float(np.std(diff, ddof=1))
    if diff_std <= 1e-14 or np.isnan(diff_std):
        return 0.0
    return float(diff_std * np.sqrt(periods_per_year))


def information_ratio(
    portfolio_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    benchmark_returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    periods_per_year: int = 252,
) -> float:
    """Calculate the Information Ratio: (mean(R_p) - mean(R_b)) * periods_per_year / Tracking Error.

    Parameters
    ----------
    portfolio_returns : Series, DataFrame, array, or sequence
        Portfolio periodic returns.
    benchmark_returns : Series, DataFrame, array, or sequence
        Benchmark periodic returns.
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Information Ratio.
    """
    rp, rb = _align_series(portfolio_returns, benchmark_returns)
    if len(rp) < 2:
        return 0.0

    te = tracking_error(rp, rb, periods_per_year=periods_per_year)
    active_return = float((rp.mean() - rb.mean()) * periods_per_year)

    if te <= 1e-14:
        if active_return > 0:
            return float("inf")
        elif active_return < 0:
            return float("-inf")
        return 0.0

    return float(active_return / te)


def omega_ratio(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    threshold: float = 0.0,
) -> float:
    """Calculate the Omega Ratio relative to a return threshold.

    Omega = Sum(max(R - L, 0)) / Sum(max(L - R, 0)).

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    threshold : float, default 0.0
        Periodic threshold return L.

    Returns
    -------
    float
        Omega Ratio.
    """
    r = _to_series(returns)
    if len(r) == 0:
        return 0.0

    diff = r.values - threshold
    gains = diff[diff > 0.0]
    losses = -diff[diff < 0.0]

    sum_gains = float(np.sum(gains))
    sum_losses = float(np.sum(losses))

    if sum_losses <= 1e-14:
        if sum_gains > 0.0:
            return float("inf")
        return 0.0

    return float(sum_gains / sum_losses)


def tail_ratio(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    upper_p: float = 95.0,
    lower_p: float = 5.0,
) -> float:
    """Calculate the Tail Ratio: Percentile(returns, upper_p) / |Percentile(returns, lower_p)|.

    Parameters
    ----------
    returns : Series, DataFrame, array, or sequence
        Periodic returns.
    upper_p : float, default 95.0
        Upper percentile (e.g. 95th percentile).
    lower_p : float, default 5.0
        Lower percentile (e.g. 5th percentile).

    Returns
    -------
    float
        Tail Ratio.
    """
    r = _to_series(returns)
    if len(r) == 0:
        return 0.0

    upper_val = float(np.percentile(r.values, upper_p))
    lower_val = abs(float(np.percentile(r.values, lower_p)))

    if lower_val <= 1e-14:
        if upper_val > 0.0:
            return float("inf")
        return 0.0

    return float(upper_val / lower_val)


def win_rate(returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]]) -> float:
    """Calculate the proportion of positive return periods (Win Rate / Hit Ratio)."""
    r = _to_series(returns)
    if len(r) == 0:
        return 0.0
    return float(np.mean(r.values > 0.0))


def gain_loss_ratio(returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]]) -> float:
    """Calculate the average gain over average loss ratio."""
    r = _to_series(returns)
    if len(r) == 0:
        return 0.0
    gains = r[r > 0.0]
    losses = r[r < 0.0]
    avg_gain = float(gains.mean()) if len(gains) > 0 else 0.0
    avg_loss = abs(float(losses.mean())) if len(losses) > 0 else 0.0

    if avg_loss <= 1e-14:
        return float("inf") if avg_gain > 0.0 else 0.0
    return float(avg_gain / avg_loss)
