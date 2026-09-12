"""
End-to-End Quantitative Research Pipeline with Deployment Readiness Scoring.
"""

from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ResearchPipelineTearSheet:
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    annual_turnover: float
    estimated_capacity_usd: float
    metrics_table: pd.DataFrame


@dataclass
class DeploymentReadinessReport:
    readiness_score: int  # 0 to 100
    is_deployable: bool
    checklist_results: Dict[str, bool]
    failure_reasons: List[str]


class QuantResearchValidationPipeline:
    """Automates the institutional 5-stage research workflow."""

    def __init__(self, risk_free_rate: float = 0.02, tx_cost_bps: float = 5.0):
        self.risk_free_rate = risk_free_rate
        self.tx_cost_bps = tx_cost_bps

    def evaluate_strategy(self, price_series: pd.Series, target_weights: pd.Series) -> ResearchPipelineTearSheet:
        # Strictly lagged execution
        w_lagged = target_weights.shift(1).fillna(0.0)
        ret = price_series.pct_change().fillna(0.0)

        turnover = target_weights.diff().abs().fillna(0.0)
        cost_drag = turnover * (self.tx_cost_bps * 1e-4)

        net_ret = (w_lagged * ret) - cost_drag
        cagr = float(np.prod(1.0 + net_ret) ** (252.0 / max(len(net_ret), 1)) - 1.0)
        vol = float(net_ret.std() * np.sqrt(252.0))
        sharpe = float((cagr - self.risk_free_rate) / vol) if vol > 0 else 0.0

        downside_ret = net_ret[net_ret < 0]
        downside_vol = float(downside_ret.std() * np.sqrt(252.0)) if len(downside_ret) > 0 else 1e-4
        sortino = float((cagr - self.risk_free_rate) / downside_vol)

        cum = (1.0 + net_ret).cumprod()
        peaks = cum.cummax()
        dd = (cum - peaks) / peaks
        max_dd = float(dd.min())
        calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

        win_rate = float(np.mean(net_ret > 0))
        ann_turnover = float(turnover.sum() * (252.0 / max(len(turnover), 1)))
        capacity = 50_000_000.0 / max(ann_turnover, 0.5)

        table = pd.DataFrame([
            {"Metric": "CAGR (Annual Return)", "Value": f"{cagr:+.2%}"},
            {"Metric": "Annualized Volatility", "Value": f"{vol:.2%}"},
            {"Metric": "Sharpe Ratio (Rf=2%)", "Value": f"{sharpe:.2f}"},
            {"Metric": "Sortino Ratio", "Value": f"{sortino:.2f}"},
            {"Metric": "Calmar Ratio", "Value": f"{calmar:.2f}"},
            {"Metric": "Maximum Drawdown", "Value": f"{max_dd:.2%}"},
            {"Metric": "Win Rate (%)", "Value": f"{win_rate:.2%}"},
            {"Metric": "Annual Turnover", "Value": f"{ann_turnover:.2f}x"},
            {"Metric": "Estimated Strategy Capacity", "Value": f"${capacity:,.0f}"},
        ])

        return ResearchPipelineTearSheet(
            cagr=cagr,
            volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            annual_turnover=ann_turnover,
            estimated_capacity_usd=capacity,
            metrics_table=table,
        )

    def score_deployment_readiness(self, tear_sheet: ResearchPipelineTearSheet, dsr_p_value: float) -> DeploymentReadinessReport:
        checks = {
            "Sharpe Ratio >= 1.0": tear_sheet.sharpe_ratio >= 1.0,
            "Max Drawdown <= 20%": abs(tear_sheet.max_drawdown) <= 0.20,
            "Deflated Sharpe (DSR) >= 95%": dsr_p_value >= 0.95,
            "Win Rate >= 50%": tear_sheet.win_rate >= 0.50,
            "Capacity >= $10M": tear_sheet.estimated_capacity_usd >= 10_000_000.0,
        }

        score = int(sum(checks.values()) / len(checks) * 100)
        failures = [k for k, v in checks.items() if not v]
        return DeploymentReadinessReport(
            readiness_score=score,
            is_deployable=score >= 80,
            checklist_results=checks,
            failure_reasons=failures,
        )
