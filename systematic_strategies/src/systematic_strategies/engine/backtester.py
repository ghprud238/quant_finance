"""Vectorized and event-aware backtesting engine for systematic trading strategies.

Accurately simulates:
- Signal-to-weight execution timing (t-1 allocation applied to t returns to eliminate lookahead bias)
- Transaction cost deduction & short borrowing fees
- Equity curve tracking & drawdown series
- Comprehensive risk, return, and trade attribution metrics
"""

from typing import Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .costs import TransactionCostModel


@dataclass
class BacktestResult:
    """Encapsulates complete backtesting performance, equity trajectory, and risk metrics."""

    strategy_name: str
    dates: pd.DatetimeIndex
    gross_returns: pd.Series
    net_returns: pd.Series
    cumulative_returns: pd.Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: Union[pd.Series, pd.DataFrame]
    metrics: Dict[str, Any]
    benchmark_returns: Optional[pd.Series] = None

    def summary_table(self) -> pd.DataFrame:
        """Returns a clean structured pandas DataFrame of all performance metrics."""
        records = []
        for category, items in self._grouped_metrics().items():
            for name, val, desc in items:
                records.append({
                    "Category": category,
                    "Metric": name,
                    "Value": val,
                    "Description": desc,
                })
        return pd.DataFrame(records)

    def _grouped_metrics(self) -> Dict[str, list]:
        m = self.metrics
        return {
            "Performance": [
                ("CAGR (Annual Return)", f"{m.get('cagr', 0.0):+.2%}", "Compound annual growth rate"),
                ("Cumulative Total Return", f"{m.get('total_return', 0.0):+.2%}", "Total wealth growth"),
                ("Daily Mean Return", f"{m.get('daily_mean', 0.0):+.4%}", "Average daily return"),
                ("Win Rate (Hit Ratio)", f"{m.get('win_rate', 0.0):.1%}", "Percentage of positive return days"),
                ("Profit Factor", f"{m.get('profit_factor', 1.0):.2f}", "Gross gains / gross losses"),
                ("Gain / Loss Ratio", f"{m.get('gain_loss_ratio', 1.0):.2f}", "Average winning day / average losing day"),
            ],
            "Risk & Volatility": [
                ("Annualized Volatility", f"{m.get('annualized_volatility', 0.0):.2%}", "Sample standard deviation * sqrt(252)"),
                ("Sharpe Ratio (Rf=2.0%)", f"{m.get('sharpe_ratio', 0.0):.2f}", "Annualized excess return / volatility"),
                ("Sortino Ratio (MAR=0%)", f"{m.get('sortino_ratio', 0.0):.2f}", "Excess return / downside semi-deviation"),
                ("Calmar Ratio", f"{m.get('calmar_ratio', 0.0):.2f}", "CAGR / Absolute Max Drawdown"),
                ("Max Drawdown", f"{m.get('max_drawdown', 0.0):.2%}", "Deepest peak-to-trough decline"),
                ("Max Drawdown Duration", f"{m.get('max_drawdown_duration', 0)} days", "Longest duration spent underwater"),
                ("Historical VaR (95% Daily)", f"{m.get('var_95', 0.0):.2%}", "95% 1-day empirical loss cutoff"),
                ("Expected Shortfall (95% CVaR)", f"{m.get('cvar_95', 0.0):.2%}", "Average loss in worst 5% tail"),
            ],
            "Execution & Costs": [
                ("Annualized Turnover", f"{m.get('annualized_turnover', 0.0):.1%}", "Total absolute weight changes per year"),
                ("Annual Cost Drag", f"{m.get('annual_cost_drag_bps', 0.0):.1f} bps", "Transaction fees and slippage drag"),
                ("Gross Sharpe Ratio", f"{m.get('gross_sharpe_ratio', 0.0):.2f}", "Pre-cost Sharpe ratio"),
                ("Sharpe Drag (Costs)", f"{m.get('sharpe_cost_drag', 0.0):.2f}", "Reduction in Sharpe due to frictions"),
            ],
        }

    def print_summary(self) -> None:
        """Prints a formatted terminal dashboard card."""
        print("┌" + "─" * 78 + "┐")
        title = f"STRATEGY BACKTEST RESULTS: {self.strategy_name.upper()}"
        print(f"│ {title.center(76)} │")
        print("├" + "─" * 78 + "┤")

        for cat_name, items in self._grouped_metrics().items():
            print(f"├─ {cat_name.upper()} " + "─" * (75 - len(cat_name)))
            for name, val, desc in items:
                print(f"│  {name:<32} {val:>12}   {desc:<27} │")
        print("└" + "─" * 78 + "┘")


class BacktestEngine:
    """Institutional systematic strategy backtesting engine."""

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.cost_model = cost_model if cost_model is not None else TransactionCostModel()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.initial_capital = initial_capital

    def run(
        self,
        prices_or_returns: Union[pd.Series, pd.DataFrame],
        weights: Union[pd.Series, pd.DataFrame],
        strategy_name: str = "Systematic Strategy",
        benchmark_returns: Optional[pd.Series] = None,
        is_returns: bool = False,
    ) -> BacktestResult:
        """Executes a backtest given price/return series and target strategy weights.

        Args:
            prices_or_returns: Asset price DataFrame/Series or return DataFrame/Series.
            weights: Target portfolio weights (t-1 allocation active during period t).
            strategy_name: Descriptive label for reporting.
            benchmark_returns: Optional benchmark daily returns for relative analytics.
            is_returns: Set to True if prices_or_returns is already return series.

        Returns:
            BacktestResult dataclass containing full performance and risk accounting.
        """
        # Align dates
        common_index = prices_or_returns.index.intersection(weights.index)
        data = prices_or_returns.loc[common_index]
        w_target = weights.loc[common_index]

        # Calculate asset returns
        if is_returns:
            asset_returns = data.copy()
        else:
            asset_returns = data.pct_change().fillna(0.0)

        # Execution Timing: Shift target weights by 1 period
        # Target weight decided at close of t-1 is held across period t
        w_executed = w_target.shift(1).fillna(0.0)

        # Compute Gross Returns
        if isinstance(asset_returns, pd.DataFrame) and isinstance(w_executed, pd.DataFrame):
            # Element-wise product summed across assets
            # Align columns
            common_cols = [c for c in asset_returns.columns if c in w_executed.columns]
            gross_returns = (asset_returns[common_cols] * w_executed[common_cols]).sum(axis=1)
        else:
            gross_returns = asset_returns * w_executed

        # Deduct Transaction Costs and Borrowing Drag
        net_returns, costs = self.cost_model.apply_costs(gross_returns, w_target)
        turnover = self.cost_model.compute_turnover(w_target)

        # Compute Cumulative Performance & Equity Curve
        cum_returns = (1.0 + net_returns).cumprod() - 1.0
        equity_curve = self.initial_capital * (1.0 + cum_returns)

        # Drawdowns
        hwm = equity_curve.cummax()
        drawdowns = (equity_curve - hwm) / hwm

        # Compute Metrics
        metrics = self._calculate_metrics(
            gross_returns=gross_returns,
            net_returns=net_returns,
            drawdowns=drawdowns,
            turnover=turnover,
            costs=costs,
            benchmark_returns=benchmark_returns,
        )

        return BacktestResult(
            strategy_name=strategy_name,
            dates=common_index,
            gross_returns=gross_returns,
            net_returns=net_returns,
            cumulative_returns=cum_returns,
            equity_curve=equity_curve,
            drawdown_series=drawdowns,
            turnover=turnover,
            costs=costs,
            weights=w_target,
            metrics=metrics,
            benchmark_returns=benchmark_returns,
        )

    def _calculate_metrics(
        self,
        gross_returns: pd.Series,
        net_returns: pd.Series,
        drawdowns: pd.Series,
        turnover: pd.Series,
        costs: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Internal statistics calculator."""
        n_days = len(net_returns)
        if n_days < 2:
            return {}

        total_return = (1.0 + net_returns).prod() - 1.0
        years = n_days / self.periods_per_year
        cagr = (1.0 + total_return) ** (1.0 / max(years, 1e-4)) - 1.0 if total_return > -1.0 else -1.0

        ann_vol = net_returns.std(ddof=1) * np.sqrt(self.periods_per_year)
        gross_vol = gross_returns.std(ddof=1) * np.sqrt(self.periods_per_year)

        # Sharpe
        daily_rf = self.risk_free_rate / self.periods_per_year
        excess_returns = net_returns - daily_rf
        sharpe = (excess_returns.mean() / (net_returns.std(ddof=1) + 1e-8)) * np.sqrt(self.periods_per_year)

        gross_excess = gross_returns - daily_rf
        gross_sharpe = (gross_excess.mean() / (gross_returns.std(ddof=1) + 1e-8)) * np.sqrt(self.periods_per_year)

        # Sortino (Downside semi-deviation)
        downside = net_returns[net_returns < 0.0]
        downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(self.periods_per_year) if len(downside) > 0 else 1e-6
        sortino = (cagr - self.risk_free_rate) / downside_std

        # Drawdowns
        max_dd = float(drawdowns.min())
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

        # Drawdown duration (consecutive days below high water mark)
        underwater = (drawdowns < 0.0).astype(int)
        runs = underwater.groupby((underwater != underwater.shift()).cumsum()).cumsum()
        max_dd_duration = int(runs.max()) if len(runs) > 0 else 0

        # Win rate & Profit factor
        wins = net_returns[net_returns > 0.0]
        losses = net_returns[net_returns < 0.0]
        win_rate = len(wins) / n_days if n_days > 0 else 0.0
        gross_gains = wins.sum()
        gross_losses = abs(losses.sum())
        profit_factor = gross_gains / gross_losses if gross_losses > 1e-8 else (999.0 if gross_gains > 0 else 1.0)
        gain_loss_ratio = (wins.mean() / abs(losses.mean())) if (len(losses) > 0 and abs(losses.mean()) > 1e-8) else 1.0

        # VaR and CVaR
        var_95 = float(np.percentile(net_returns, 5.0))
        var_99 = float(np.percentile(net_returns, 1.0))
        cvar_95 = float(net_returns[net_returns <= var_95].mean()) if len(net_returns[net_returns <= var_95]) > 0 else var_95

        # Turnover & Costs
        ann_turnover = float(turnover.mean() * self.periods_per_year)
        ann_cost_drag = float(costs.mean() * self.periods_per_year * 10_000.0)

        res = {
            "n_periods": n_days,
            "total_return": float(total_return),
            "cagr": float(cagr),
            "daily_mean": float(net_returns.mean()),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "gross_sharpe_ratio": float(gross_sharpe),
            "sharpe_cost_drag": float(gross_sharpe - sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "gain_loss_ratio": float(gain_loss_ratio),
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "annualized_turnover": ann_turnover,
            "annual_cost_drag_bps": ann_cost_drag,
        }

        # Benchmark analytics
        if benchmark_returns is not None:
            bench_aligned = benchmark_returns.reindex(net_returns.index).fillna(0.0)
            cov = np.cov(net_returns, bench_aligned)[0, 1]
            bench_var = np.var(bench_aligned, ddof=1)
            beta = cov / bench_var if bench_var > 1e-8 else 1.0
            bench_cagr = (1.0 + bench_aligned).prod() ** (1.0 / max(years, 1e-4)) - 1.0
            alpha = cagr - (self.risk_free_rate + beta * (bench_cagr - self.risk_free_rate))
            diff = net_returns - bench_aligned
            tracking_err = diff.std(ddof=1) * np.sqrt(self.periods_per_year)
            info_ratio = (diff.mean() * self.periods_per_year) / (tracking_err + 1e-8)

            res.update({
                "benchmark_cagr": float(bench_cagr),
                "beta": float(beta),
                "alpha": float(alpha),
                "tracking_error": float(tracking_err),
                "information_ratio": float(info_ratio),
            })

        return res
