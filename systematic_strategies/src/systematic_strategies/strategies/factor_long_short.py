"""Factor-Based Long/Short Equity Strategy (Project 14).

Implements a systematic multi-factor quantitative equity strategy:
1. Cross-sectional multi-factor scoring (Value, Momentum, Quality, Low-Vol, Size).
2. Robust winsorization and cross-sectional Z-score standardization.
3. Quantile sorting into Long (Top Q5) and Short (Bottom Q1) portfolios.
4. Dollar-neutral (+0.5/-0.5) and Beta-neutral portfolio construction.
5. Periodic rebalancing with turnover smoothing and transaction/borrow cost modeling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FactorBacktestResult:
    """Container for Factor Long/Short backtest results."""

    returns: pd.Series  # Net strategy spread returns
    long_returns: pd.Series  # Long leg returns
    short_returns: pd.Series  # Short leg returns (as realized by strategy)
    gross_returns: pd.Series  # Gross spread returns before transaction/borrow costs
    benchmark_returns: Optional[pd.Series]
    weights: pd.DataFrame  # Time series of asset weights
    turnover: pd.Series  # Daily turnover
    metrics: Dict[str, float]  # Summary performance statistics

    def summary_table(self) -> pd.DataFrame:
        """Returns performance metrics formatted as a pandas DataFrame."""
        rows = []
        for k, v in self.metrics.items():
            if "Return" in k or "Volatility" in k or "Drawdown" in k or "Cost" in k or "Turnover" in k or "Rate" in k:
                val_str = f"{v:+.2%}" if "Return" in k or "Drawdown" in k else f"{v:.2%}"
            elif "Ratio" in k or "Beta" in k or "Alpha" in k or "T-Stat" in k or "Factor" in k:
                val_str = f"{v:.2f}"
            else:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            rows.append({"Metric": k, "Value": val_str})
        return pd.DataFrame(rows)


class FactorLongShortStrategy:
    """Factor-Based Cross-Sectional Long/Short Strategy.

    Parameters
    ----------
    factor_weights : Dict[str, float], optional
        Weights for each factor in composite score. Default:
        {'value': 0.25, 'momentum': 0.25, 'quality': 0.20, 'low_vol': 0.20, 'size': 0.10}
    n_quantiles : int, default 5
        Number of cross-sectional quantile buckets (5 for quintiles, 10 for deciles).
    long_quantile : int, default 5
        Target quantile for long positions (highest factor score).
    short_quantile : int, default 1
        Target quantile for short positions (lowest factor score).
    dollar_neutral : bool, default True
        Enforces dollar neutrality (sum(w_long) = +0.5, sum(w_short) = -0.5, net=0.0).
    beta_neutral : bool, default False
        Enforces market beta neutrality (sum(w_i * beta_i) = 0).
    rebalance_freq : int or str, default 21
        Rebalance frequency in trading days (e.g. 21 for monthly, 5 for weekly, 1 for daily).
    turnover_smoothing : float, default 1.0
        Smoothing parameter alpha in (0, 1]. w_t = (1-alpha)*w_{t-1} + alpha*w_target.
    winsorize_limits : Tuple[float, float], default (0.01, 0.01)
        Quantile bounds for cross-sectional winsorization.
    zscore_clip : float, default 3.0
        Cap on absolute standardized z-scores (+/- 3.0 sigma).
    transaction_cost_bps : float, default 5.0
        One-way transaction cost per unit of turnover (basis points).
    borrow_cost_bps : float, default 50.0
        Annualized short borrow financing fee (basis points).
    """

    DEFAULT_FACTOR_WEIGHTS = {
        "value": 0.25,
        "momentum": 0.25,
        "quality": 0.20,
        "low_vol": 0.20,
        "size": 0.10,
    }

    def __init__(
        self,
        factor_weights: Optional[Dict[str, float]] = None,
        n_quantiles: int = 5,
        long_quantile: int = 5,
        short_quantile: int = 1,
        dollar_neutral: bool = True,
        beta_neutral: bool = False,
        rebalance_freq: Union[int, str] = 21,
        turnover_smoothing: float = 1.0,
        winsorize_limits: Tuple[float, float] = (0.01, 0.01),
        zscore_clip: float = 3.0,
        transaction_cost_bps: float = 5.0,
        borrow_cost_bps: float = 50.0,
    ):
        if factor_weights is None:
            self.factor_weights = self.DEFAULT_FACTOR_WEIGHTS.copy()
        else:
            total = sum(factor_weights.values())
            self.factor_weights = {k: v / total for k, v in factor_weights.items()}

        if n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if not (1 <= short_quantile < long_quantile <= n_quantiles):
            raise ValueError(f"Invalid quantiles: short={short_quantile}, long={long_quantile} for n_quantiles={n_quantiles}")
        if not (0.0 < turnover_smoothing <= 1.0):
            raise ValueError("turnover_smoothing must be in (0.0, 1.0]")

        self.n_quantiles = n_quantiles
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile
        self.dollar_neutral = dollar_neutral
        self.beta_neutral = beta_neutral
        self.rebalance_freq = rebalance_freq
        self.turnover_smoothing = turnover_smoothing
        self.winsorize_limits = winsorize_limits
        self.zscore_clip = zscore_clip
        self.transaction_cost_bps = transaction_cost_bps
        self.borrow_cost_bps = borrow_cost_bps

    def standardize_cross_section(
        self,
        values: pd.Series,
        winsorize: bool = True,
        clip_val: Optional[float] = None,
    ) -> pd.Series:
        """Applies cross-sectional winsorization and Z-score standardization.

        Z_i = (X_i - mean(X)) / std(X)
        """
        if values.dropna().empty or len(values.dropna()) < 2:
            return pd.Series(0.0, index=values.index)

        valid = values.dropna().copy()
        if winsorize and len(valid) >= 5:
            low_q = valid.quantile(self.winsorize_limits[0])
            high_q = valid.quantile(1.0 - self.winsorize_limits[1])
            valid = valid.clip(lower=low_q, upper=high_q)

        mean_val = valid.mean()
        std_val = valid.std(ddof=1)

        if std_val < 1e-8 or np.isnan(std_val):
            z = pd.Series(0.0, index=values.index)
        else:
            z = (valid - mean_val) / std_val

        clip = clip_val if clip_val is not None else self.zscore_clip
        if clip is not None:
            z = z.clip(lower=-clip, upper=clip)
            final_std = z.std(ddof=1)
            if final_std > 1e-8:
                z = (z - z.mean()) / final_std
            else:
                z = z - z.mean()

        return z.reindex(values.index).fillna(0.0)

    def compute_composite_scores(
        self,
        factor_cross_section: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Computes composite multi-factor score for a cross-section of assets.

        Composite Score S_i = sum_k w_k * Z_{i,k}
        """
        w_dict = weights or self.factor_weights
        total_w = sum(w_dict.get(k, 0.0) for k in factor_cross_section.keys())
        if total_w < 1e-8:
            total_w = 1.0

        all_tickers = list(next(iter(factor_cross_section.values())).index)
        composite = pd.Series(0.0, index=all_tickers)

        for factor_name, raw_series in factor_cross_section.items():
            if factor_name in w_dict and w_dict[factor_name] > 0:
                weight = w_dict[factor_name] / total_w
                z_scores = self.standardize_cross_section(raw_series, winsorize=False)
                composite += weight * z_scores.reindex(all_tickers).fillna(0.0)

        return self.standardize_cross_section(composite, winsorize=False)

    def assign_quantiles(
        self,
        scores: pd.Series,
        n_quantiles: Optional[int] = None,
    ) -> pd.Series:
        """Assigns integer quantiles [1 .. n_quantiles] based on factor scores."""
        n_q = n_quantiles or self.n_quantiles
        valid_scores = scores.dropna()
        if len(valid_scores) < n_q:
            ranks = scores.rank(ascending=True, method="first")
            return pd.Series(
                np.where(ranks > len(scores) / 2, self.long_quantile, self.short_quantile),
                index=scores.index,
            )

        try:
            quantiles = pd.qcut(scores, q=n_q, labels=range(1, n_q + 1), duplicates="drop")
            return quantiles.astype(int)
        except Exception:
            ranks = scores.rank(pct=True)
            quantiles = (ranks * n_q).clip(upper=n_q).apply(np.ceil).astype(int)
            return quantiles

    def construct_portfolio_weights(
        self,
        scores: pd.Series,
        betas: Optional[pd.Series] = None,
        dollar_neutral: Optional[bool] = None,
        beta_neutral: Optional[bool] = None,
        gross_leverage: float = 1.0,
    ) -> pd.Series:
        """Constructs long/short portfolio weights.

        Parameters
        ----------
        scores : pd.Series
            Cross-sectional composite factor scores.
        betas : pd.Series, optional
            Market betas for beta-neutral construction.
        dollar_neutral : bool, optional
            Overrides instance setting.
        beta_neutral : bool, optional
            Overrides instance setting.
        gross_leverage : float, default 1.0
            Total gross exposure sum(|w_i|) = 1.0 (50% long, 50% short).
        """
        is_dn = self.dollar_neutral if dollar_neutral is None else dollar_neutral
        is_bn = self.beta_neutral if beta_neutral is None else beta_neutral

        quantiles = self.assign_quantiles(scores)
        long_mask = quantiles == self.long_quantile
        short_mask = quantiles == self.short_quantile

        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())

        weights = pd.Series(0.0, index=scores.index)

        if n_long == 0 and n_short == 0:
            return weights

        long_target = gross_leverage / 2.0
        short_target = -gross_leverage / 2.0

        if n_long > 0:
            weights[long_mask] = long_target / n_long
        if n_short > 0:
            weights[short_mask] = short_target / n_short

        if is_bn and betas is not None and n_long > 0 and n_short > 0:
            asset_betas = betas.reindex(scores.index).fillna(1.0)
            long_beta = (weights[long_mask] * asset_betas[long_mask]).sum()
            short_beta = (weights[short_mask] * asset_betas[short_mask]).sum()

            if abs(long_beta) > 1e-6 and abs(short_beta) > 1e-6:
                scale_long = 1.0 / long_beta
                scale_short = -1.0 / short_beta

                w_long_bn = weights[long_mask] * scale_long
                w_short_bn = weights[short_mask] * scale_short

                total_gross = w_long_bn.abs().sum() + w_short_bn.abs().sum()
                if total_gross > 1e-8:
                    norm_factor = gross_leverage / total_gross
                    weights[long_mask] = w_long_bn * norm_factor
                    weights[short_mask] = w_short_bn * norm_factor

        return weights

    def backtest(
        self,
        prices: pd.DataFrame,
        factor_data: Dict[str, pd.DataFrame],
        betas: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
    ) -> FactorBacktestResult:
        """Executes full historical backtest of the factor long/short strategy."""
        returns_df = prices.pct_change().fillna(0.0)
        dates = prices.index
        n_days = len(dates)
        tickers = list(prices.columns)

        if isinstance(self.rebalance_freq, int):
            rebalance_step = self.rebalance_freq
            rebal_indices = set(range(0, n_days, rebalance_step))
        else:
            rebal_dates = pd.date_range(dates[0], dates[-1], freq=self.rebalance_freq)
            rebal_indices = {dates.get_indexer([d], method="nearest")[0] for d in rebal_dates}

        weights_matrix = np.zeros((n_days, len(tickers)))
        current_weights = pd.Series(0.0, index=tickers)

        daily_long_ret = np.zeros(n_days)
        daily_short_ret = np.zeros(n_days)
        daily_gross_ret = np.zeros(n_days)
        daily_net_ret = np.zeros(n_days)
        daily_turnover = np.zeros(n_days)

        cost_per_turnover = self.transaction_cost_bps / 10000.0
        daily_borrow_rate = (self.borrow_cost_bps / 10000.0) / 252.0

        for t in range(n_days):
            date = dates[t]

            if t in rebal_indices or t == 0:
                current_factors = {}
                for f_name, f_matrix in factor_data.items():
                    if date in f_matrix.index:
                        current_factors[f_name] = f_matrix.loc[date].dropna()

                if len(current_factors) > 0:
                    composite_scores = self.compute_composite_scores(current_factors)
                    current_betas = betas.loc[date] if betas is not None and date in betas.index else None

                    target_w = self.construct_portfolio_weights(
                        scores=composite_scores,
                        betas=current_betas,
                    )

                    if self.turnover_smoothing < 1.0 and t > 0:
                        new_weights = (1.0 - self.turnover_smoothing) * current_weights + self.turnover_smoothing * target_w
                    else:
                        new_weights = target_w

                    turnover = (new_weights - current_weights).abs().sum()
                    rebalance_cost = turnover * cost_per_turnover
                    current_weights = new_weights
                else:
                    turnover = 0.0
                    rebalance_cost = 0.0
            else:
                turnover = 0.0
                rebalance_cost = 0.0

            weights_matrix[t, :] = current_weights.values
            daily_turnover[t] = turnover

            r_t = returns_df.iloc[t].values
            w_t = current_weights.values

            w_long = np.where(w_t > 0, w_t, 0.0)
            w_short = np.where(w_t < 0, w_t, 0.0)

            r_long = np.sum(w_long * r_t)
            r_short = np.sum(w_short * r_t)

            gross_spread = r_long + r_short
            borrow_cost = np.sum(np.abs(w_short)) * daily_borrow_rate
            net_spread = gross_spread - rebalance_cost - borrow_cost

            daily_long_ret[t] = r_long
            daily_short_ret[t] = r_short
            daily_gross_ret[t] = gross_spread
            daily_net_ret[t] = net_spread

            if t < n_days - 1:
                evolved_w = w_t * (1.0 + r_t)
                total_abs = np.sum(np.abs(evolved_w))
                if total_abs > 1e-8:
                    current_weights = pd.Series(evolved_w / total_abs, index=tickers)

        weights_df = pd.DataFrame(weights_matrix, index=dates, columns=tickers)
        net_series = pd.Series(daily_net_ret, index=dates)
        long_series = pd.Series(daily_long_ret, index=dates)
        short_series = pd.Series(daily_short_ret, index=dates)
        gross_series = pd.Series(daily_gross_ret, index=dates)
        turnover_series = pd.Series(daily_turnover, index=dates)

        metrics = self._compute_performance_metrics(
            net_returns=net_series,
            long_returns=long_series,
            short_returns=short_series,
            turnover=turnover_series,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
        )

        return FactorBacktestResult(
            returns=net_series,
            long_returns=long_series,
            short_returns=short_series,
            gross_returns=gross_series,
            benchmark_returns=benchmark_returns,
            weights=weights_df,
            turnover=turnover_series,
            metrics=metrics,
        )

    def _compute_performance_metrics(
        self,
        net_returns: pd.Series,
        long_returns: pd.Series,
        short_returns: pd.Series,
        turnover: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, float]:
        clean_ret = net_returns.dropna()
        n_periods = len(clean_ret)
        if n_periods < 2:
            return {}

        ann_factor = 252.0
        daily_rf = risk_free_rate / ann_factor

        ann_return = float((1.0 + clean_ret.mean()) ** ann_factor - 1.0)
        ann_vol = float(clean_ret.std(ddof=1) * np.sqrt(ann_factor))
        excess_ret = clean_ret - daily_rf
        sharpe = float(np.sqrt(ann_factor) * excess_ret.mean() / clean_ret.std(ddof=1)) if ann_vol > 1e-8 else 0.0

        downside_diff = clean_ret[clean_ret < daily_rf] - daily_rf
        downside_dev = float(np.sqrt(np.mean(downside_diff**2)) * np.sqrt(ann_factor)) if len(downside_diff) > 0 else 1e-8
        sortino = float(np.sqrt(ann_factor) * excess_ret.mean() / downside_dev) if downside_dev > 1e-8 else 0.0

        cum_wealth = (1.0 + clean_ret).cumprod()
        peak = cum_wealth.cummax()
        drawdown = (cum_wealth - peak) / peak
        max_drawdown = float(drawdown.min())
        calmar = float(ann_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-8 else 0.0

        wins = clean_ret[clean_ret > 0]
        losses = clean_ret[clean_ret < 0]
        win_rate = float(len(wins) / len(clean_ret)) if len(clean_ret) > 0 else 0.0
        profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-8 else np.nan

        long_ann_ret = float((1.0 + long_returns.mean()) ** ann_factor - 1.0)
        short_ann_ret = float((1.0 + short_returns.mean()) ** ann_factor - 1.0)
        avg_turnover = float(turnover.mean() * ann_factor)

        metrics = {
            "Annualized Return (Net)": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Long Leg Ann Return": long_ann_ret,
            "Short Leg Ann Return": short_ann_ret,
            "Annualized Turnover": avg_turnover,
        }

        if benchmark_returns is not None:
            bench_aligned = benchmark_returns.reindex(clean_ret.index).fillna(0.0)
            cov_matrix = np.cov(clean_ret, bench_aligned)
            bench_var = cov_matrix[1, 1]
            if bench_var > 1e-8:
                beta = float(cov_matrix[0, 1] / bench_var)
                bench_ann = float((1.0 + bench_aligned.mean()) ** ann_factor - 1.0)
                alpha = float(ann_return - (risk_free_rate + beta * (bench_ann - risk_free_rate)))
                active_ret = clean_ret - bench_aligned
                tracking_err = float(active_ret.std(ddof=1) * np.sqrt(ann_factor))
                info_ratio = float(active_ret.mean() * ann_factor / tracking_err) if tracking_err > 1e-8 else 0.0

                metrics.update({
                    "Realized Market Beta": beta,
                    "Jensen Alpha": alpha,
                    "Tracking Error": tracking_err,
                    "Information Ratio": info_ratio,
                })

        return metrics
