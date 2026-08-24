"""Multi-Asset Trend-Following Strategy (Project 15).

Implements a systematic multi-asset trend-following and managed futures strategy:
1. Multi-Horizon Time-Series Momentum (TSMOM: 1M, 3M, 6M, 12M) and MA crossover conviction.
2. Continuous conviction score in [-1, +1] per asset.
3. Volatility targeting per asset (scaling inverse to realized volatility).
4. Equal Risk Contribution (ERC) / Risk Parity allocation across 4 core asset classes:
   Equities, Fixed Income, Currencies (FX), and Commodities.
5. Overall portfolio volatility targeting and all-weather diversification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class MultiAssetTrendResult:
    """Container for Multi-Asset Trend strategy backtest results."""

    returns: pd.Series  # Net strategy portfolio returns
    gross_returns: pd.Series  # Gross strategy returns
    asset_class_returns: pd.DataFrame  # Daily returns per asset class bucket
    weights: pd.DataFrame  # Daily weights for each individual asset
    class_weights: pd.DataFrame  # Daily aggregate weight per asset class
    leverage: pd.Series  # Gross leverage over time
    turnover: pd.Series  # Daily turnover
    metrics: Dict[str, float]  # Performance statistics
    regime_breakdown: Optional[pd.DataFrame] = None  # Crisis / market phase performance

    def summary_table(self) -> pd.DataFrame:
        """Returns performance metrics formatted as a pandas DataFrame."""
        rows = []
        for k, v in self.metrics.items():
            if "Return" in k or "Volatility" in k or "Drawdown" in k or "Cost" in k or "Turnover" in k or "Rate" in k:
                val_str = f"{v:+.2%}" if "Return" in k or "Drawdown" in k else f"{v:.2%}"
            elif "Ratio" in k or "Beta" in k or "Alpha" in k or "Factor" in k or "Leverage" in k:
                val_str = f"{v:.2f}"
            else:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            rows.append({"Metric": k, "Value": val_str})
        return pd.DataFrame(rows)


class MultiAssetTrendStrategy:
    """Multi-Asset Multi-Horizon Trend-Following & Risk Parity Strategy.

    Parameters
    ----------
    asset_classes : Dict[str, str], optional
        Mapping of ticker to asset class ('Equities', 'Bonds', 'Currencies', 'Commodities').
    lookback_horizons : List[int], default [21, 63, 126, 252]
        Lookback windows in days for TSMOM trend detection (1M, 3M, 6M, 12M).
    horizon_weights : Optional[List[float]], optional
        Weights for combining horizons. If None, equal weights are used.
    target_asset_vol : float, default 0.10
        Target annualized volatility per individual asset (e.g. 10%).
    target_portfolio_vol : float, default 0.10
        Target annualized volatility for total strategy portfolio (e.g. 10%).
    vol_lookback : int, default 60
        Rolling window in days for estimating realized volatility.
    max_asset_leverage : float, default 2.0
        Maximum absolute leverage cap per individual asset.
    max_gross_leverage : float, default 3.5
        Maximum total gross leverage (sum of abs weights).
    use_risk_parity : bool, default True
        If True, solves for Equal Risk Contribution across asset classes.
    ma_fast : int, default 20
        Fast moving average window for crossover confirmation.
    ma_slow : int, default 100
        Slow moving average window for crossover confirmation.
    rebalance_freq : int, default 5
        Rebalancing frequency in trading days (5 for weekly, 1 for daily, 21 for monthly).
    transaction_cost_bps : float, default 3.0
        Transaction cost per unit of turnover (basis points).
    """

    DEFAULT_ASSET_CLASSES = {
        "SPY": "Equities",
        "QQQ": "Equities",
        "IWM": "Equities",
        "EEM": "Equities",
        "TLT": "Bonds",
        "IEF": "Bonds",
        "AGG": "Bonds",
        "UUP": "Currencies",
        "FXE": "Currencies",
        "FXY": "Currencies",
        "GLD": "Commodities",
        "SLV": "Commodities",
        "USO": "Commodities",
        "DBA": "Commodities",
    }

    def __init__(
        self,
        asset_classes: Optional[Dict[str, str]] = None,
        lookback_horizons: Optional[List[int]] = None,
        horizon_weights: Optional[List[float]] = None,
        target_asset_vol: float = 0.10,
        target_portfolio_vol: float = 0.10,
        vol_lookback: int = 60,
        max_asset_leverage: float = 2.0,
        max_gross_leverage: float = 3.5,
        use_risk_parity: bool = True,
        ma_fast: int = 20,
        ma_slow: int = 100,
        rebalance_freq: int = 5,
        transaction_cost_bps: float = 3.0,
    ):
        self.asset_classes = asset_classes or self.DEFAULT_ASSET_CLASSES.copy()
        self.lookback_horizons = lookback_horizons or [21, 63, 126, 252]

        if horizon_weights is None:
            self.horizon_weights = [1.0 / len(self.lookback_horizons)] * len(self.lookback_horizons)
        else:
            total_hw = sum(horizon_weights)
            self.horizon_weights = [w / total_hw for w in horizon_weights]

        self.target_asset_vol = target_asset_vol
        self.target_portfolio_vol = target_portfolio_vol
        self.vol_lookback = vol_lookback
        self.max_asset_leverage = max_asset_leverage
        self.max_gross_leverage = max_gross_leverage
        self.use_risk_parity = use_risk_parity
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.rebalance_freq = rebalance_freq
        self.transaction_cost_bps = transaction_cost_bps

    def compute_trend_conviction(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Computes continuous multi-horizon trend conviction scores C_{i,t} in [-1, +1]."""
        returns = prices.pct_change()
        vol = returns.rolling(window=self.vol_lookback, min_periods=20).std() * np.sqrt(252.0)
        vol = vol.clip(lower=0.04)

        # 1. Multi-horizon TSMOM signals
        tsmom_scores = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for h, w_h in zip(self.lookback_horizons, self.horizon_weights):
            h_ret = prices.pct_change(periods=h)
            h_vol = vol * np.sqrt(h / 252.0)
            norm_ret = h_ret / h_vol
            h_signal = np.tanh(norm_ret)
            tsmom_scores += w_h * h_signal.fillna(0.0)

        # 2. Moving Average Crossover signal
        ma_fast_df = prices.ewm(span=self.ma_fast, min_periods=self.ma_fast).mean()
        ma_slow_df = prices.ewm(span=self.ma_slow, min_periods=self.ma_slow).mean()
        ma_diff = (ma_fast_df - ma_slow_df) / prices
        ma_norm = ma_diff / (vol / np.sqrt(252.0))
        ma_signal = np.tanh(ma_norm).fillna(0.0)

        combined = 0.70 * tsmom_scores + 0.30 * ma_signal
        return combined.clip(lower=-1.0, upper=1.0)

    def estimate_realized_volatilities(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates annualized rolling realized volatility for each asset."""
        returns = prices.pct_change()
        vol = returns.rolling(window=self.vol_lookback, min_periods=20).std() * np.sqrt(252.0)
        return vol.clip(lower=0.03, upper=0.80)

    def compute_volatility_scaled_weights(
        self,
        convictions: pd.DataFrame,
        volatilities: pd.DataFrame,
    ) -> pd.DataFrame:
        """Scales position size inversely proportional to realized volatility.

        w_i = (target_asset_vol / sigma_i) * Conviction_i
        """
        vol_scale = self.target_asset_vol / volatilities
        vol_scale = vol_scale.clip(upper=self.max_asset_leverage)
        raw_weights = vol_scale * convictions
        return raw_weights.clip(lower=-self.max_asset_leverage, upper=self.max_asset_leverage)

    def solve_risk_parity_weights(
        self,
        cov_matrix: np.ndarray,
        target_risk_contributions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solves for Equal Risk Contribution (ERC) portfolio weights across asset classes using Spinu (2013) convex formulation.

        min_x 0.5 * x^T Sigma x - sum_i b_i ln(x_i)
        w = x / sum(x)
        """
        n_assets = cov_matrix.shape[0]
        if n_assets <= 1:
            return np.ones(n_assets)

        if target_risk_contributions is None:
            b = np.ones(n_assets) / n_assets
        else:
            b = target_risk_contributions / np.sum(target_risk_contributions)

        diag_reg = 1e-6 * np.eye(n_assets)
        reg_cov = cov_matrix + diag_reg

        def objective(x: np.ndarray) -> float:
            if np.any(x <= 1e-12):
                return 1e12
            return float(0.5 * (x @ reg_cov @ x) - np.sum(b * np.log(x)))

        def gradient(x: np.ndarray) -> np.ndarray:
            clipped_x = np.clip(x, 1e-12, None)
            return (reg_cov @ clipped_x) - (b / clipped_x)

        # Initial point based on inverse volatility
        vols = np.sqrt(np.diag(reg_cov))
        vols = np.where(vols > 1e-6, vols, 1.0)
        x0 = 1.0 / vols

        res = minimize(
            objective,
            x0,
            jac=gradient,
            method="BFGS",
            options={"maxiter": 200, "gtol": 1e-7},
        )

        if res.success and np.all(res.x > 0):
            w = res.x / np.sum(res.x)
            return w
        else:
            inv_vol = 1.0 / vols
            return inv_vol / np.sum(inv_vol)

    def allocate_risk_parity_classes(
        self,
        prices: pd.DataFrame,
        vol_scaled_weights: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregates trend signals into 4 core asset classes and balances via Risk Parity."""
        dates = prices.index
        tickers = list(prices.columns)
        returns = prices.pct_change().fillna(0.0)

        classes = sorted(list(set(self.asset_classes.get(t, "Equities") for t in tickers)))
        n_classes = len(classes)

        class_members = {c: [t for t in tickers if self.asset_classes.get(t, "Equities") == c] for c in classes}

        final_weights_df = pd.DataFrame(0.0, index=dates, columns=tickers)
        class_weights_df = pd.DataFrame(0.0, index=dates, columns=classes)

        class_returns_dict = {}
        for c in classes:
            c_tickers = class_members[c]
            if len(c_tickers) > 0:
                class_returns_dict[c] = returns[c_tickers].mean(axis=1)
            else:
                class_returns_dict[c] = pd.Series(0.0, index=dates)
        class_ret_df = pd.DataFrame(class_returns_dict)

        rolling_cov = class_ret_df.rolling(window=self.vol_lookback, min_periods=25).cov() * 252.0

        for t_idx, date in enumerate(dates):
            if t_idx < self.vol_lookback:
                class_w = np.ones(n_classes) / n_classes
            else:
                try:
                    cov_mat = rolling_cov.loc[date].values
                    if np.isnan(cov_mat).any():
                        class_w = np.ones(n_classes) / n_classes
                    else:
                        class_w = self.solve_risk_parity_weights(cov_mat)
                except Exception:
                    class_w = np.ones(n_classes) / n_classes

            class_weights_df.loc[date] = class_w

            for c_idx, c in enumerate(classes):
                c_tickers = class_members[c]
                c_weight_budget = class_w[c_idx]

                if len(c_tickers) == 0:
                    continue

                raw_w = vol_scaled_weights.loc[date, c_tickers].values
                abs_sum = np.sum(np.abs(raw_w))
                if abs_sum > 1e-6:
                    scaled_w = (raw_w / abs_sum) * c_weight_budget * len(c_tickers)
                else:
                    scaled_w = raw_w

                final_weights_df.loc[date, c_tickers] = scaled_w

        return final_weights_df, class_weights_df

    def backtest(
        self,
        prices: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
    ) -> MultiAssetTrendResult:
        """Runs complete multi-asset trend following and risk parity backtest."""
        convictions = self.compute_trend_conviction(prices)
        volatilities = self.estimate_realized_volatilities(prices)
        vol_weights = self.compute_volatility_scaled_weights(convictions, volatilities)

        if self.use_risk_parity:
            raw_weights_df, class_weights_df = self.allocate_risk_parity_classes(prices, vol_weights)
        else:
            raw_weights_df = vol_weights
            classes = sorted(list(set(self.asset_classes.get(t, "Equities") for t in prices.columns)))
            class_weights_df = pd.DataFrame(0.0, index=prices.index, columns=classes)
            for c in classes:
                c_tickers = [t for t in prices.columns if self.asset_classes.get(t, "Equities") == c]
                if len(c_tickers) > 0:
                    class_weights_df[c] = raw_weights_df[c_tickers].abs().sum(axis=1)

        dates = prices.index
        n_days = len(dates)
        tickers = list(prices.columns)
        returns_df = prices.pct_change().fillna(0.0)

        rebal_indices = set(range(0, n_days, self.rebalance_freq))

        daily_weights = np.zeros((n_days, len(tickers)))
        daily_turnover = np.zeros(n_days)
        daily_gross_ret = np.zeros(n_days)
        daily_net_ret = np.zeros(n_days)

        current_w = pd.Series(0.0, index=tickers)
        cost_per_turnover = self.transaction_cost_bps / 10000.0

        for t in range(n_days):
            date = dates[t]

            if t in rebal_indices or t == 0:
                target_w = raw_weights_df.loc[date].copy()

                if t >= self.vol_lookback:
                    hist_ret = returns_df.iloc[t - self.vol_lookback:t].values
                    est_cov = np.cov(hist_ret, rowvar=False) * 252.0
                    port_var = target_w.values @ est_cov @ target_w.values
                    if port_var > 1e-8:
                        port_vol = np.sqrt(port_var)
                        vol_multiplier = self.target_portfolio_vol / port_vol
                        target_w = target_w * np.clip(vol_multiplier, 0.25, 2.5)

                gross_lev = target_w.abs().sum()
                if gross_lev > self.max_gross_leverage:
                    target_w = target_w * (self.max_gross_leverage / gross_lev)

                turnover = (target_w - current_w).abs().sum()
                rebal_cost = turnover * cost_per_turnover
                current_w = target_w
            else:
                turnover = 0.0
                rebal_cost = 0.0

            daily_weights[t, :] = current_w.values
            daily_turnover[t] = turnover

            r_t = returns_df.iloc[t].values
            gross_r = np.sum(current_w.values * r_t)
            net_r = gross_r - rebal_cost

            daily_gross_ret[t] = gross_r
            daily_net_ret[t] = net_r

            if t < n_days - 1:
                evolved_w = current_w.values * (1.0 + r_t)
                current_w = pd.Series(evolved_w, index=tickers)

        weights_df = pd.DataFrame(daily_weights, index=dates, columns=tickers)
        net_series = pd.Series(daily_net_ret, index=dates)
        gross_series = pd.Series(daily_gross_ret, index=dates)
        turnover_series = pd.Series(daily_turnover, index=dates)
        leverage_series = weights_df.abs().sum(axis=1)

        classes = sorted(list(set(self.asset_classes.get(t, "Equities") for t in tickers)))
        class_ret_data = {}
        for c in classes:
            c_tickers = [t for t in tickers if self.asset_classes.get(t, "Equities") == c]
            if len(c_tickers) > 0:
                c_w = weights_df[c_tickers].values
                c_r = returns_df[c_tickers].values
                class_ret_data[c] = np.sum(c_w * c_r, axis=1)
            else:
                class_ret_data[c] = np.zeros(n_days)
        asset_class_returns_df = pd.DataFrame(class_ret_data, index=dates)

        metrics = self._compute_performance_metrics(
            net_returns=net_series,
            turnover=turnover_series,
            leverage=leverage_series,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
        )

        return MultiAssetTrendResult(
            returns=net_series,
            gross_returns=gross_series,
            asset_class_returns=asset_class_returns_df,
            weights=weights_df,
            class_weights=class_weights_df,
            leverage=leverage_series,
            turnover=turnover_series,
            metrics=metrics,
        )

    def _compute_performance_metrics(
        self,
        net_returns: pd.Series,
        turnover: pd.Series,
        leverage: pd.Series,
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

        metrics = {
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Average Leverage": float(leverage.mean()),
            "Annualized Turnover": float(turnover.mean() * ann_factor),
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
