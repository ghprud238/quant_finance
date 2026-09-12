"""
Nexus Quant Platform - Derivatives, Option Pricing & Implied State-Price Densities Suite.
"""

from .black_scholes import (
    BlackScholesEngine,
    ImpliedVolatilitySolver,
    OptionGreeks,
    OptionPriceResult,
    SVIModel,
    SVIParameters,
)
from .heston_fft import (
    CarrMadanFFTPricer,
    FangOosterleeCOSPricer,
    HestonCalibrator,
    HestonCharacteristicFunction,
    HestonParameters,
)
from .breeden_litzenberger import (
    BreedenLitzenbergerDensityEstimator,
    DensityExtractionResult,
    RiskNeutralMoments,
)

__all__ = [
    "BlackScholesEngine",
    "OptionGreeks",
    "OptionPriceResult",
    "ImpliedVolatilitySolver",
    "SVIParameters",
    "SVIModel",
    "HestonParameters",
    "HestonCharacteristicFunction",
    "CarrMadanFFTPricer",
    "FangOosterleeCOSPricer",
    "HestonCalibrator",
    "BreedenLitzenbergerDensityEstimator",
    "DensityExtractionResult",
    "RiskNeutralMoments",
]
