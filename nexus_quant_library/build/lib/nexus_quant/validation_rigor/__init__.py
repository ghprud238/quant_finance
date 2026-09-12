"""
Statistical Rigor & Backtest Overfitting Detection Suite (Bailey & López de Prado 2014).
"""

from .deflated_sharpe import DeflatedSharpeRatioCalculator, DSRReport, PSRReport
from .validation_pipeline import QuantResearchValidationPipeline, ResearchPipelineTearSheet, DeploymentReadinessReport

__all__ = [
    "DeflatedSharpeRatioCalculator",
    "DSRReport",
    "PSRReport",
    "QuantResearchValidationPipeline",
    "ResearchPipelineTearSheet",
    "DeploymentReadinessReport",
]
