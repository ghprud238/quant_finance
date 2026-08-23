"""
Quant Foundations: Factor Exposure Analyzer & Market Regime Detection Suite.
"""

from quant_foundations.factors.model import MultiFactorRegression
from quant_foundations.factors.exposure import FactorExposureReport
from quant_foundations.regimes.hmm_model import GaussianHMMRegimeDetector
from quant_foundations.regimes.heuristic import TrendVolRegimeFilter
from quant_foundations.regimes.gmm_model import GMMRegimeDetector

__version__ = "0.1.0"
__all__ = [
    "MultiFactorRegression",
    "FactorExposureReport",
    "GaussianHMMRegimeDetector",
    "TrendVolRegimeFilter",
    "GMMRegimeDetector",
]
