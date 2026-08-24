"""Module 25: Alternative Data Alpha Model.

Pre-processes and orthogonalizes alternative datasets (consumer sentiment, web traffic,
satellite/supply chain activity), evaluates multi-horizon Information Coefficient (IC) decay,
and executes dollar-neutral systematic factor strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ICDecayReport:
    """Report summarizing multi-horizon Information Coefficient (IC) analysis."""
    horizons: List[int]
    mean_ic: Dict[int, float]
    mean_rank_ic: Dict[int, float]
    ic_std: Dict[int, float]
    ic_ir: Dict[int, float]          # Information Ratio: mean_ic / std_ic
    ic_t_stat: Dict[int, float]     # t-statistic: mean_ic / (std_ic / sqrt(N))
    ic_p_value: Dict[int, float]    # p-value of two-tailed test
    daily_ic_series: Dict[int, pd.Series]

    def summary_table(self) -> pd.DataFrame:
        """Formats IC metrics into a structured summary DataFrame."""
        rows = []
        for h in self.horizons:
            rows.append({
                "Horizon (Days)": h,
                "Mean IC": self.mean_ic[h],
                "Mean Rank IC": self.mean_rank_ic[h],
                "IC Std Dev": self.ic_std[h],
                "IC IR": self.ic_ir[h],
                "t-Statistic": self.ic_t_stat[h],
                "p-Value": self.ic_p_value[h],
                "Significant (p<0.05)": self.ic_p_value[h] < 0.05,
            })
        return pd.DataFrame(rows).set_index("Horizon (Days)")


@dataclass
class QuantilePerformance:
    """Analysis of forward return distribution sorted across factor quantiles."""
    n_quantiles: int
    mean_returns: Dict[int, float]
    annualized_returns: Dict[int, float]
    annualized_volatilities: Dict[int, float]
    sharpe_ratios: Dict[int, float]
    cumulative_curves: Dict[int, pd.Series]
    spread_q_top_bottom: float        # Q_top - Q_bottom annualized return spread
    is_monotonic: bool                # True if Q1 < Q2 < ... < Q_top
    dates: pd.DatetimeIndex

    def summary_table(self) -> pd.DataFrame:
        """Formats quantile performance table."""
        rows = []
        for q in range(1, self.n_quantiles + 1):
            rows.append({
                "Quantile": f"Q{q}",
                "Daily Mean Return (%)": self.mean_returns[q] * 100,
                "Annualized Return (%)": self.annualized_returns[q] * 100,
                "Annualized Volatility (%)": self.annualized_volatilities[q] * 100,
                "Sharpe Ratio": self.sharpe_ratios[q],
            })
        return pd.DataFrame(rows).set_index("Quantile")


@dataclass
class AlternativeAlphaBacktestResult:
    """Results from systematic long/short strategy execution on alternative signal."""
    strategy_name: str
    dates: pd.DatetimeIndex
    gross_returns: pd.Series
    net_returns: pd.Series
    cumulative_returns: pd.Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    metrics: Dict[str, float]

    def summary_table(self) -> pd.DataFrame:
        """Formats performance metrics into a clean DataFrame."""
        rows = [
            ("Annualized Return (Gross)", f"{self.metrics.get('cagr_gross', 0.0):+.2%}"),
            ("Annualized Return (Net)", f"{self.metrics.get('cagr_net', 0.0):+.2%}"),
            ("Annualized Volatility", f"{self.metrics.get('annualized_volatility', 0.0):.2%}"),
            ("Sharpe Ratio (Gross)", f"{self.metrics.get('sharpe_gross', 0.0):.2f}"),
            ("Sharpe Ratio (Net)", f"{self.metrics.get('sharpe_net', 0.0):.2f}"),
            ("Sortino Ratio", f"{self.metrics.get('sortino_ratio', 0.0):.2f}"),
            ("Calmar Ratio", f"{self.metrics.get('calmar_ratio', 0.0):.2f}"),
            ("Maximum Drawdown", f"{self.metrics.get('max_drawdown', 0.0):.2%}"),
            ("Win Rate (Daily)", f"{self.metrics.get('win_rate', 0.0):.1%}"),
            ("Profit Factor", f"{self.metrics.get('profit_factor', 0.0):.2f}"),
            ("Annualized Turnover", f"{self.metrics.get('annualized_turnover', 0.0):.1%}"),
            ("Annual Cost Drag (bps)", f"{self.metrics.get('annual_cost_bps', 0.0):.1f} bps"),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")


class AlternativeDataAlphaModel:
    """Alternative Data Alpha Model (Project 25).

    Ingests, smooths, standardizes, and factor-neutralizes alternative datasets,
    computes multi-horizon IC decay dynamics, and simulates dollar-neutral alpha portfolios.
    """

    def __init__(
        self,
        decay_factor: float = 0.85,
        winsorize_limits: Tuple[float, float] = (0.01, 0.01),
        zscore_clip: float = 3.0,
        n_quantiles: int = 5,
        rebalance_freq: int = 5,
        transaction_cost_bps: float = 5.0,
        borrow_cost_bps: float = 50.0,
        risk_free_rate: float = 0.02,
    ) -> None:
        self.decay_factor = decay_factor
        self.winsorize_limits = winsorize_limits
        self.zscore_clip = zscore_clip
        self.n_quantiles = n_quantiles
        self.rebalance_freq = rebalance_freq
        self.transaction_cost_bps = transaction_cost_bps
        self.borrow_cost_bps = borrow_cost_bps
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def generate_synthetic_data(
        n_stocks: int = 30,
        n_days: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates realistic multi-stock price histories, alternative datasets, and risk factors.

        Datasets generated:
        - : Daily closing stock prices.
        - : NLP news/sentiment index with short-term predictive alpha (1-5 days).
        - : Digital traffic & app growth with medium-term predictive alpha (5-21 days).
        - : Logistics velocity with quarterly predictive alpha (10-42 days).
        - : Traditional factor loadings (Market Beta, Momentum, Size) for neutralization.
        """
        np.random.seed(seed)
        dates = pd.bdate_range(end="2024-12-31", periods=n_days, freq="B")
        tickers = [f"STK_{i:02d}" for i in range(n_stocks)]

        # 1. Base stock return generation via factor model + idiosyncratic alpha
        market_returns = np.random.normal(0.0004, 0.012, size=n_days)
        momentum_factor = np.random.normal(0.0001, 0.008, size=n_days)
        size_factor = np.random.normal(0.0000, 0.006, size=n_days)

        # True alpha vectors (underlying unobserved firm quality trends)
        alpha_base = np.random.randn(n_stocks)
        alpha_base /= np.std(alpha_base)

        stock_betas_mkt = np.random.uniform(0.7, 1.4, size=n_stocks)
        stock_betas_mom = np.random.uniform(-0.5, 0.5, size=n_stocks)
        stock_betas_size = np.random.uniform(-0.6, 0.6, size=n_stocks)

        # Alternative data raw signals (true signal + noise)
        sentiment_mat = np.zeros((n_days, n_stocks))
        web_mat = np.zeros((n_days, n_stocks))
        supply_mat = np.zeros((n_days, n_stocks))

        # Returns matrix
        ret_mat = np.zeros((n_days, n_stocks))

        # Dynamic state progression
        state_sent = np.zeros(n_stocks)
        state_web = np.zeros(n_stocks)
        state_supply = np.zeros(n_stocks)

        for t in range(n_days):
            # Evolve latent alternative signals with mean-reversion
            state_sent = 0.80 * state_sent + 0.20 * np.random.randn(n_stocks) + 0.15 * alpha_base
            state_web = 0.92 * state_web + 0.08 * np.random.randn(n_stocks) + 0.10 * alpha_base
            state_supply = 0.96 * state_supply + 0.04 * np.random.randn(n_stocks) + 0.08 * alpha_base

            sentiment_mat[t, :] = state_sent + 0.4 * np.random.randn(n_stocks)
            web_mat[t, :] = state_web + 0.3 * np.random.randn(n_stocks)
            supply_mat[t, :] = state_supply + 0.3 * np.random.randn(n_stocks)

            # Predictive return contribution (lagged lead)
            idio_alpha = (
                0.0018 * state_sent +
                0.0015 * state_web +
                0.0012 * state_supply +
                np.random.normal(0, 0.015, size=n_stocks)
            )

            # Stock return = Beta * Market + Beta * Mom + Beta * Size + Idiosyncratic Alpha
            ret_mat[t, :] = (
                stock_betas_mkt * market_returns[t] +
                stock_betas_mom * momentum_factor[t] +
                stock_betas_size * size_factor[t] +
                idio_alpha
            )

        # Price paths from returns
        initial_prices = np.random.uniform(50.0, 150.0, size=n_stocks)
        prices_mat = np.zeros((n_days, n_stocks))
        prices_mat[0, :] = initial_prices
        for t in range(1, n_days):
            prices_mat[t, :] = prices_mat[t-1, :] * (1.0 + ret_mat[t, :])

        prices_df = pd.DataFrame(prices_mat, index=dates, columns=tickers)
        sentiment_df = pd.DataFrame(sentiment_mat, index=dates, columns=tickers)
        web_df = pd.DataFrame(web_mat, index=dates, columns=tickers)
        supply_df = pd.DataFrame(supply_mat, index=dates, columns=tickers)

        # Factor loadings DataFrame (static or rolling)
        loadings_dict = {
            "Market_Beta": pd.DataFrame(np.tile(stock_betas_mkt, (n_days, 1)), index=dates, columns=tickers),
            "Momentum_Beta": pd.DataFrame(np.tile(stock_betas_mom, (n_days, 1)), index=dates, columns=tickers),
            "Size_Beta": pd.DataFrame(np.tile(stock_betas_size, (n_days, 1)), index=dates, columns=tickers),
        }

        return {
            "prices": prices_df,
            "sentiment": sentiment_df,
            "web_traffic": web_df,
            "supply_chain": supply_df,
            "risk_loadings": loadings_dict,
            "market_returns": pd.Series(market_returns, index=dates),
        }

    def combine_signals(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Combines multiple alternative data signals using cross-sectional z-score averaging."""
        if weights is None:
            weights = {k: 1.0 / len(signals_dict) for k in signals_dict.keys()}

        norm_weights = {k: v / sum(weights.values()) for k, v in weights.items()}

        composite = None
        for name, sig_df in signals_dict.items():
            w = norm_weights.get(name, 0.0)
            z_sig = self.standardize_cross_section(sig_df)
            if composite is None:
                composite = w * z_sig
            else:
                composite = composite + w * z_sig

        return composite

    def exponential_decay_smoothing(
        self,
        signal_df: pd.DataFrame,
        decay_factor: Optional[float] = None,
    ) -> pd.DataFrame:
        """Applies recursive exponential decay filter: S_decay(t) = lambda * S_decay(t-1) + (1-lambda) * S(t)."""
        lam = decay_factor if decay_factor is not None else self.decay_factor
        smoothed = signal_df.ewm(alpha=(1.0 - lam), adjust=False).mean()
        return smoothed

    def standardize_cross_section(
        self,
        signal_df: pd.DataFrame,
        winsorize_limits: Optional[Tuple[float, float]] = None,
        zscore_clip: Optional[float] = None,
    ) -> pd.DataFrame:
        """Performs cross-sectional Winsorization and z-score standardisation per date."""
        w_limits = winsorize_limits if winsorize_limits is not None else self.winsorize_limits
        clip_val = zscore_clip if zscore_clip is not None else self.zscore_clip

        standardized = pd.DataFrame(index=signal_df.index, columns=signal_df.columns, dtype=float)

        for date, row in signal_df.iterrows():
            vals = row.dropna().values
            if len(vals) < 3:
                standardized.loc[date, :] = 0.0
                continue

            # 1. Percentile Winsorization
            if w_limits[0] > 0 or w_limits[1] > 0:
                low_p = np.percentile(vals, w_limits[0] * 100)
                high_p = np.percentile(vals, (1.0 - w_limits[1]) * 100)
                vals_w = np.clip(vals, low_p, high_p)
            else:
                vals_w = vals

            # 2. Z-Score
            mean_v = np.mean(vals_w)
            std_v = np.std(vals_w)
            if std_v > 1e-8:
                z = (vals_w - mean_v) / std_v
                if clip_val is not None:
                    z = np.clip(z, -clip_val, clip_val)
            else:
                z = np.zeros_like(vals_w)

            standardized.loc[date, row.dropna().index] = z

        return standardized.fillna(0.0)

    def neutralize_factors(
        self,
        signal_df: pd.DataFrame,
        risk_loadings: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Residualizes alternative signal against risk factor loadings via cross-sectional OLS projection.

        Formula: S_neutral = S - X (X^T X)^(-1) X^T S
        """
        neutral_signal = pd.DataFrame(index=signal_df.index, columns=signal_df.columns, dtype=float)
        factor_names = list(risk_loadings.keys())

        for date in signal_df.index:
            y = signal_df.loc[date].values
            valid_mask = ~np.isnan(y)
            if np.sum(valid_mask) < len(factor_names) + 2:
                neutral_signal.loc[date, :] = y
                continue

            # Construct X matrix with constant (market intercept) + risk loadings
            x_cols = [np.ones(np.sum(valid_mask))]
            for fname in factor_names:
                f_row = risk_loadings[fname].loc[date].values[valid_mask]
                # Standardize factor column
                f_std = (f_row - np.mean(f_row)) / (np.std(f_row) + 1e-8)
                x_cols.append(f_std)

            X = np.column_stack(x_cols)
            y_sub = y[valid_mask]

            try:
                # OLS projection beta: (X^T X)^(-1) X^T y
                beta, _, _, _ = np.linalg.lstsq(X, y_sub, rcond=None)
                residuals = y_sub - X @ beta
                neutral_signal.loc[date, valid_mask] = residuals
            except Exception:
                neutral_signal.loc[date, valid_mask] = y_sub

        return self.standardize_cross_section(neutral_signal)

    def compute_ic_decay(
        self,
        signal_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        horizons: Optional[List[int]] = None,
        rank_ic: bool = True,
    ) -> ICDecayReport:
        """Computes Information Coefficient (IC) across multiple forward horizons."""
        if horizons is None:
            horizons = [1, 2, 5, 10, 21, 42, 63]

        mean_ic = {}
        mean_rank_ic = {}
        ic_std = {}
        ic_ir = {}
        ic_t_stat = {}
        ic_p_val = {}
        daily_ic_dict = {}

        for h in horizons:
            # Forward return: (P_{t+h} - P_t) / P_t
            fwd_returns = (prices_df.shift(-h) - prices_df) / prices_df

            daily_ic_list = []
            daily_rank_ic_list = []

            for date in signal_df.index:
                if date not in fwd_returns.index:
                    continue
                s_row = signal_df.loc[date].dropna()
                r_row = fwd_returns.loc[date].dropna()
                common = s_row.index.intersection(r_row.index)
                if len(common) < 5:
                    continue

                s_vals = s_row.loc[common].values
                r_vals = r_row.loc[common].values

                # Pearson IC
                if np.std(s_vals) > 1e-8 and np.std(r_vals) > 1e-8:
                    p_ic, _ = stats.pearsonr(s_vals, r_vals)
                    r_ic, _ = stats.spearmanr(s_vals, r_vals)
                    if not np.isnan(p_ic):
                        daily_ic_list.append((date, p_ic))
                    if not np.isnan(r_ic):
                        daily_rank_ic_list.append((date, r_ic))

            if daily_ic_list:
                ic_series = pd.Series([x[1] for x in daily_ic_list], index=[x[0] for x in daily_ic_list])
                rank_ic_series = pd.Series([x[1] for x in daily_rank_ic_list], index=[x[0] for x in daily_rank_ic_list])

                m_ic = float(ic_series.mean())
                m_rank = float(rank_ic_series.mean())
                s_ic = float(ic_series.std()) if len(ic_series) > 1 else 1e-4
                ir = m_ic / s_ic if s_ic > 1e-8 else 0.0
                t_stat = m_ic / (s_ic / np.sqrt(len(ic_series))) if s_ic > 1e-8 and len(ic_series) > 1 else 0.0
                p_val = float(2.0 * stats.t.sf(np.abs(t_stat), df=max(len(ic_series)-1, 1)))

                mean_ic[h] = m_ic
                mean_rank_ic[h] = m_rank
                ic_std[h] = s_ic
                ic_ir[h] = ir
                ic_t_stat[h] = t_stat
                ic_p_val[h] = p_val
                daily_ic_dict[h] = rank_ic_series if rank_ic else ic_series
            else:
                mean_ic[h] = 0.0
                mean_rank_ic[h] = 0.0
                ic_std[h] = 0.0
                ic_ir[h] = 0.0
                ic_t_stat[h] = 0.0
                ic_p_val[h] = 1.0
                daily_ic_dict[h] = pd.Series(dtype=float)

        return ICDecayReport(
            horizons=horizons,
            mean_ic=mean_ic,
            mean_rank_ic=mean_rank_ic,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_t_stat=ic_t_stat,
            ic_p_value=ic_p_val,
            daily_ic_series=daily_ic_dict,
        )

    def quantile_analysis(
        self,
        signal_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        n_quantiles: Optional[int] = None,
        forward_horizon: int = 1,
    ) -> QuantilePerformance:
        """Evaluates forward return distribution and monotonicity across factor quantiles."""
        nq = n_quantiles if n_quantiles is not None else self.n_quantiles
        returns_df = prices_df.pct_change().shift(-forward_horizon).dropna()

        quantile_daily_returns = {q: [] for q in range(1, nq + 1)}
        valid_dates = []

        for date in signal_df.index:
            if date not in returns_df.index:
                continue
            sig_row = signal_df.loc[date].dropna()
            ret_row = returns_df.loc[date].dropna()
            common = sig_row.index.intersection(ret_row.index)
            if len(common) < nq * 2:
                continue

            valid_dates.append(date)
            # Rank stocks into quantiles
            ranks = pd.qcut(sig_row.loc[common], q=nq, labels=False, duplicates="drop") + 1

            for q in range(1, nq + 1):
                q_assets = common[ranks == q]
                if len(q_assets) > 0:
                    q_ret = float(ret_row.loc[q_assets].mean())
                else:
                    q_ret = 0.0
                quantile_daily_returns[q].append(q_ret)

        dt_idx = pd.DatetimeIndex(valid_dates)
        mean_rets = {}
        ann_rets = {}
        ann_vols = {}
        sharpes = {}
        cum_curves = {}

        for q in range(1, nq + 1):
            s = pd.Series(quantile_daily_returns[q], index=dt_idx)
            m_r = float(s.mean())
            v_r = float(s.std() * np.sqrt(252)) if len(s) > 1 else 0.0
            ann_r = float(m_r * 252)
            sh = (ann_r - self.risk_free_rate) / v_r if v_r > 1e-8 else 0.0

            mean_rets[q] = m_r
            ann_rets[q] = ann_r
            ann_vols[q] = v_r
            sharpes[q] = sh
            cum_curves[q] = (1.0 + s).cumprod()

        spread = ann_rets[nq] - ann_rets[1]
        is_mono = all(mean_rets[i] <= mean_rets[i+1] for i in range(1, nq))

        return QuantilePerformance(
            n_quantiles=nq,
            mean_returns=mean_rets,
            annualized_returns=ann_rets,
            annualized_volatilities=ann_vols,
            sharpe_ratios=sharpes,
            cumulative_curves=cum_curves,
            spread_q_top_bottom=spread,
            is_monotonic=is_mono,
            dates=dt_idx,
        )

    def backtest_long_short(
        self,
        signal_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        n_quantiles: Optional[int] = None,
        rebalance_freq: Optional[int] = None,
        transaction_cost_bps: Optional[float] = None,
        borrow_cost_bps: Optional[float] = None,
        strategy_name: str = "Alternative Data Alpha Strategy",
    ) -> AlternativeAlphaBacktestResult:
        """Executes a dollar-neutral Long Top Quintile (Q5) / Short Bottom Quintile (Q1) strategy."""
        nq = n_quantiles if n_quantiles is not None else self.n_quantiles
        rfreq = rebalance_freq if rebalance_freq is not None else self.rebalance_freq
        fee_bps = transaction_cost_bps if transaction_cost_bps is not None else self.transaction_cost_bps
        borrow_bps = borrow_cost_bps if borrow_cost_bps is not None else self.borrow_cost_bps

        returns_df = prices_df.pct_change().fillna(0.0)
        dates = prices_df.index
        n_dates = len(dates)
        tickers = prices_df.columns

        weights = pd.DataFrame(0.0, index=dates, columns=tickers)

        # Generate target weights on rebalancing days
        target_w = pd.Series(0.0, index=tickers)
        for t in range(n_dates):
            date = dates[t]
            if t % rfreq == 0 and date in signal_df.index:
                sig_row = signal_df.loc[date].dropna()
                if len(sig_row) >= nq * 2:
                    ranks = pd.qcut(sig_row, q=nq, labels=False, duplicates="drop") + 1
                    long_assets = sig_row.index[ranks == nq]
                    short_assets = sig_row.index[ranks == 1]

                    target_w = pd.Series(0.0, index=tickers)
                    if len(long_assets) > 0:
                        target_w.loc[long_assets] = +0.5 / len(long_assets)
                    if len(short_assets) > 0:
                        target_w.loc[short_assets] = -0.5 / len(short_assets)

            weights.loc[date, :] = target_w

        # Lagged execution: apply t-1 weights to period t returns
        lagged_weights = weights.shift(1).fillna(0.0)
        gross_pnl = (lagged_weights * returns_df).sum(axis=1)

        # Turnover & Costs
        weight_changes = weights.diff().abs().sum(axis=1).fillna(0.0)
        trade_costs = weight_changes * (fee_bps / 10000.0)
        short_borrow_costs = (lagged_weights.clip(upper=0).abs().sum(axis=1)) * (borrow_bps / 10000.0 / 252.0)
        total_costs = trade_costs + short_borrow_costs

        net_pnl = gross_pnl - total_costs
        cum_ret = (1.0 + net_pnl).cumprod() - 1.0
        equity = 100000.0 * (1.0 + cum_ret)

        # Drawdowns
        hwm = equity.cummax()
        dd = (equity - hwm) / hwm
        max_dd = float(dd.min())

        # Metrics
        n_p = len(net_pnl)
        cagr_net = float(cum_ret.iloc[-1] ** (252.0 / n_p) - 1.0) if cum_ret.iloc[-1] > -0.99 else -0.99
        ann_vol = float(net_pnl.std() * np.sqrt(252)) if net_pnl.std() > 1e-8 else 1e-4
        sharpe_gross = float((gross_pnl.mean() * 252 - self.risk_free_rate) / ann_vol)
        sharpe_net = float((net_pnl.mean() * 252 - self.risk_free_rate) / ann_vol)

        downside_ret = net_pnl[net_pnl < 0]
        downside_vol = float(downside_ret.std() * np.sqrt(252)) if len(downside_ret) > 1 else 1e-4
        sortino = float((net_pnl.mean() * 252 - self.risk_free_rate) / downside_vol)
        calmar = abs(cagr_net / max_dd) if abs(max_dd) > 1e-4 else 0.0

        win_rate = float(np.mean(net_pnl > 0))
        gains = net_pnl[net_pnl > 0].sum()
        losses = abs(net_pnl[net_pnl < 0].sum())
        profit_factor = float(gains / losses) if losses > 1e-8 else 1.0
        ann_turnover = float(weight_changes.mean() * 252)
        ann_cost_bps = float(total_costs.mean() * 252 * 10000)

        metrics = {
            "cagr_gross": float(gross_pnl.mean() * 252),
            "cagr_net": cagr_net,
            "annualized_volatility": ann_vol,
            "sharpe_gross": sharpe_gross,
            "sharpe_net": sharpe_net,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "annualized_turnover": ann_turnover,
            "annual_cost_bps": ann_cost_bps,
        }

        return AlternativeAlphaBacktestResult(
            strategy_name=strategy_name,
            dates=dates,
            gross_returns=gross_pnl,
            net_returns=net_pnl,
            cumulative_returns=cum_ret,
            equity_curve=equity,
            drawdown_series=dd,
            turnover=weight_changes,
            weights=weights,
            metrics=metrics,
        )
