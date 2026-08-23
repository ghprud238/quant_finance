"""Visualization engine for quant foundations reproducing the infographic dark theme."""

from .plots import (
    DARK_THEME_STYLE,
    plot_equity_price_and_returns,
    plot_risk_dashboard_summary,
    plot_correlation_heatmap,
    plot_factor_exposures,
    plot_market_regime_timeline,
    plot_master_infographic,
)

__all__ = [
    "DARK_THEME_STYLE",
    "plot_equity_price_and_returns",
    "plot_risk_dashboard_summary",
    "plot_correlation_heatmap",
    "plot_factor_exposures",
    "plot_market_regime_timeline",
    "plot_master_infographic",
]
