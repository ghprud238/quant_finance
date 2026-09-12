"""
Master Capstone Execution Runner: Executes all 11 quantitative foundation tracks and renders master artifacts.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nexus_quant.ai_alternative_data import (
    FinancialFeatureEngineer,
    MLReturnPredictor,
    SupplyChainGraphAlpha,
    SemanticDriftEngine,
    MultiSourceSentimentEngine,
    MultiAgentHedgeFundSwarm,
)
from nexus_quant.defi_prediction_markets import (
    ConcentratedLiquidityAMM,
    LossVersusRebalancingEngine,
    CashAndCarryBasisTrader,
    PredictionMarketArbitrageEngine,
    BinaryOrderBook,
    OrderBookLevel,
)
from nexus_quant.validation_rigor import (
    DeflatedSharpeRatioCalculator,
    QuantResearchValidationPipeline,
)
from nexus_quant.orchestrator import NexusMasterOrchestrator
from nexus_quant.visualization.plots import (
    plot_cross_asset_portfolio_dashboard,
    plot_ai_alternative_data_dashboard,
    plot_defi_prediction_dashboard,
    plot_validation_rigor_dashboard,
    plot_master_capstone_infographic,
)


def main():
    print("=" * 80)
    print("NEXUS MASTER QUANT PLATFORM — 11-TRACK CAPSTONE INTEGRATION SUITE")
    print("=" * 80)

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Execute Master Orchestrator
    print("\n[1/5] Building Cross-Asset Master Portfolio across all 11 domains...")
    orchestrator = NexusMasterOrchestrator(initial_capital_usd=10_000_000.0)
    port_result = orchestrator.build_cross_asset_portfolio(n_days=1260, seed=42)
    print(port_result.summary())

    # 2. Render Dashboards
    print("\n[2/5] Rendering Cross-Asset Portfolio & Stress Testing Dashboard...")
    plot_cross_asset_portfolio_dashboard(port_result, os.path.join(output_dir, "02_portfolio_risk_dashboard.png"))

    print("\n[3/5] Rendering AI & Alternative Data Engine Dashboard (SEC Drift & GNN)...")
    plot_ai_alternative_data_dashboard(os.path.join(output_dir, "09_advanced_quant_and_garch.png"))

    print("\n[4/5] Rendering DeFi AMM & Prediction Market Mispricing Dashboard...")
    plot_defi_prediction_dashboard(os.path.join(output_dir, "11_defi_prediction_and_rigor.png"))

    print("\n[5/5] Rendering Master 11-Track Capstone Infographic...")
    plot_master_capstone_infographic(port_result, os.path.join(output_dir, "nexus_master_infographic.png"))

    print("\n" + "=" * 80)
    print(f"SUCCESS: All Master Capstone Artifacts Rendered to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
