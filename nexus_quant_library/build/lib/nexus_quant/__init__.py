"""
Nexus Quant (nexus-quant)
=========================
Enterprise-Grade Cross-Asset Quantitative Finance, DeFi, High-Frequency Microstructure, 
AI & Climate Risk Library.

Modules:
--------
- foundations: Volatility estimators (Yang-Zhang, Parkinson), Returns, Gaussian HMM Regimes.
- risk: Historical, Cornish-Fisher, Monte Carlo VaR & CVaR, Multi-Crisis Stress Testing, Climate VaR.
- strategies: Backtest Engine, Momentum, Mean Reversion, Online Kalman Filter Stat Arb.
- derivatives: Black-Scholes, Greeks, SVI Surface, Heston FFT/COS, Breeden-Litzenberger RND.
- macro: Nelson-Siegel Curves, Central Bank Hawk/Dove NLP, FX Carry Parity, Carbon ETS.
- microstructure: Level 2 LOB, Almgren-Chriss Execution, VPIN Flow Toxicity, Hawkes Processes, Avellaneda-Stoikov MM.
- ai: Financial Feature Engineering (FFD), GNN Supply Chain, SEC 10-K NLP, Fear & Greed Sentiment, Agentic Swarms.
- defi: Uniswap v3 Concentrated Liquidity, LVR Adverse Selection, Perpetual Funding Basis, Prediction Market Arbitrage.
- portfolio: Markowitz MPT, Wasserstein Distributionally Robust Optimization (DRO), Risk Parity (ERC).
- rigor: Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR), 5-Stage Research Validation Pipeline.
- orchestrator: Unified Multi-Asset Nexus Master Platform Engine.
"""

__version__ = "1.0.0"
__author__ = "Nexus Quant Core Engineering Group"
__license__ = "Apache-2.0"

# Top-level namespace aliases for institutional developer convenience
from nexus_quant import foundations as foundations
from nexus_quant import risk as risk
from nexus_quant import strategies as strategies
from nexus_quant import derivatives as derivatives
from nexus_quant import macro_fixed_income as macro
from nexus_quant import microstructure_execution as microstructure
from nexus_quant import ai_alternative_data as ai
from nexus_quant import defi_prediction_markets as defi
from nexus_quant import portfolio_optimization as portfolio
from nexus_quant import validation_rigor as rigor
from nexus_quant import orchestrator as orchestrator
from nexus_quant.core.data_engine import UnifiedDataEngine as DataEngine
from nexus_quant.orchestrator.master_engine import NexusMasterOrchestrator as Engine

__all__ = [
    "__version__",
    "foundations",
    "risk",
    "strategies",
    "derivatives",
    "macro",
    "microstructure",
    "ai",
    "defi",
    "portfolio",
    "rigor",
    "orchestrator",
    "DataEngine",
    "Engine",
]
