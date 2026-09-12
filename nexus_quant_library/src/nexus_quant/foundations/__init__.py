"""Nexus Quant Platform Foundations Module."""
from nexus_quant.foundations.returns import (
    simple_returns,
    log_returns,
    cumulative_returns,
    annualized_return,
    rolling_returns,
)
from nexus_quant.foundations.volatility import (
    close_to_close_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
    volatility_cone,
)
from nexus_quant.foundations.regimes import (
    GaussianHMMRegimeDetector,
    TrendVolRegimeFilter,
)

__all__ = [
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "annualized_return",
    "rolling_returns",
    "close_to_close_volatility",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "yang_zhang_volatility",
    "volatility_cone",
    "GaussianHMMRegimeDetector",
    "TrendVolRegimeFilter",
]
