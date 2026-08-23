"""Analyzer module: returns engine, range-based volatility models, and distribution metrics."""

from quant_foundations.analyzer.distribution import (
    fit_distributions,
    jarque_bera_test,
    kurtosis,
    skewness,
)
from quant_foundations.analyzer.returns import (
    annualized_return,
    cumulative_returns,
    log_returns,
    rolling_returns,
    simple_returns,
)
from quant_foundations.analyzer.volatility import (
    close_to_close_volatility,
    garman_klass_volatility,
    parkinson_volatility,
    rogers_satchell_volatility,
    volatility_cone,
    yang_zhang_volatility,
)

__all__ = [
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "rolling_returns",
    "annualized_return",
    "close_to_close_volatility",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "yang_zhang_volatility",
    "volatility_cone",
    "skewness",
    "kurtosis",
    "jarque_bera_test",
    "fit_distributions",
]
