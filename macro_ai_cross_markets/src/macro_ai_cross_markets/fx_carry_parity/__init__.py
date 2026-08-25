"""Cross-Economy FX Carry Trade, Interest Rate Parity & Volatility Surface Engine."""

from .carry_engine import (
    ParityResult,
    FamaRegressionResult,
    MalzVolSurfaceResult,
    FXCarryStrategyResult,
    FXCarryParityEngine,
)

__all__ = [
    "ParityResult",
    "FamaRegressionResult",
    "MalzVolSurfaceResult",
    "FXCarryStrategyResult",
    "FXCarryParityEngine",
]
