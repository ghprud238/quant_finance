"""
Strategies: Institutional Vectorized Backtesting Engine with Lagged Execution & Cost Friction.
"""

from typing import Union, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    drawdown_series: pd.Series
    turnover: pd.Series
    metrics: Dict[str, float]

    def summary_table(self) -> pd.DataFrame:
        records = [{"Metric": k, "Value": f"{v:.2%}" if "Return" in k or "Vol" in k or "Drawdown" in k or "Rate" in k else f"{v:.2f}"} for k, v in self.metrics.items()]
        return pd.DataFrame(records)


class SystematicBacktestEngine:
    """Vectorized backtest engine enforcing zero lookahead via t-1 lagged weight execution."""

    def __init__(
        self,
        fee_bps: float = 5.0,
        half_spread_bps: float = 2.5,
        quadratic_impact_gamma: float = 1e-4,
        borrow_cost_bps: float = 50.0,
        risk_free_rate: float = 0.02,
    ):
        self.fee_bps = fee_bps
        self.half_spread_bps = half_spread_bps
        self.quadratic_impact_gamma = quadratic_impact_gamma
        self.borrow_cost_bps = borrow_cost_bps
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        target_weights: Union[pd.Series, pd.DataFrame],
        initial_capital: float = 1_000_000.0,
    ) -> BacktestResult:
        """Run backtest with strictly lagged target positions."""
        if isinstance(prices, pd.Series):
            asset_returns = prices.pct_change().fillna(0.0)
            weights = target_weights.shift(1).fillna(0.0)
            weight_diffs = weights.diff().fillna(weights.iloc[0]).abs()
            gross_ret = weights * asset_returns
            turnover = weight_diffs
            borrow_drag = np.maximum(-weights, 0.0) * (self.borrow_cost_bps / 10000.0 / 252.0)
        else:
            asset_returns = prices.pct_change().fillna(0.0)
            weights = target_weights.shift(1).fillna(0.0)
            weight_diffs = weights.diff().fillna(weights.iloc[0]).abs().sum(axis=1)
            gross_ret = (weights * asset_returns).sum(axis=1)
            turnover = weight_diffs
            borrow_drag = (np.maximum(-weights, 0.0).sum(axis=1)) * (self.borrow_cost_bps / 10000.0 / 252.0)

        # Cost model: linear + quadratic impact
        linear_cost = turnover * ((self.fee_bps + self.half_spread_bps) / 10000.0)
        impact_cost = 0.5 * self.quadratic_impact_gamma * (turnover ** 2)
        total_drag = linear_cost + impact_cost + borrow_drag

        net_ret = gross_ret - total_drag
        equity = initial_capital * (1.0 + net_ret).cumprod()

        # Drawdowns
        hwm = equity.cummax()
        dd = (equity - hwm) / hwm
        max_dd = float(dd.min())

        # Metrics
        n_days = len(net_ret)
        cagr = float((equity.iloc[-1] / initial_capital) ** (252.0 / max(n_days, 1)) - 1.0) if equity.iloc[-1] > 0 else -1.0
        ann_vol = float(net_ret.std(ddof=1) * np.sqrt(252.0))
        excess_ret = cagr - self.risk_free_rate
        sharpe = float(excess_ret / ann_vol) if ann_vol > 0 else 0.0

        downside_ret = net_ret[net_ret < 0.0]
        downside_std = float(downside_ret.std(ddof=1) * np.sqrt(252.0)) if len(downside_ret) > 1 else 1e-4
        sortino = float(excess_ret / downside_std)

        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
        win_rate = float(np.mean(net_ret > 0.0))

        metrics = {
            "CAGR": cagr,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_dd,
            "Daily Win Rate": win_rate,
            "Annual Turnover": float(turnover.mean() * 252.0),
        }

        return BacktestResult(
            equity_curve=equity,
            gross_returns=gross_ret,
            net_returns=net_ret,
            drawdown_series=dd,
            turnover=turnover,
            metrics=metrics,
        )
