"""Portfolio Analysis and Risk Management Engine.

Subpackage for quantitative portfolio risk metrics, drawdown analytics,
Value at Risk (VaR), Conditional VaR (Expected Shortfall), risk-adjusted
performance measures, and interactive risk dashboard.
"""

from quant_foundations.portfolio.risk_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    downside_deviation,
    drawdown_series,
    excess_kurtosis,
    gain_loss_ratio,
    information_ratio,
    jensens_alpha,
    max_drawdown,
    max_drawdown_duration,
    omega_ratio,
    realized_beta,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tail_ratio,
    tracking_error,
    value_at_risk,
    win_rate,
)

from quant_foundations.portfolio.dashboard import PortfolioRiskDashboard

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "value_at_risk",
    "conditional_value_at_risk",
    "realized_beta",
    "jensens_alpha",
    "tracking_error",
    "information_ratio",
    "omega_ratio",
    "tail_ratio",
    "downside_deviation",
    "skewness",
    "excess_kurtosis",
    "win_rate",
    "gain_loss_ratio",
    "PortfolioRiskDashboard",
]
