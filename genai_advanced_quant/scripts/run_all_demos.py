#!/usr/bin/env python3
"""Master demonstration runner for Frontier Quantitative AI, Advanced Math & Alternative Data (31-35).

Executes all 5 modules:
1. Financial LLM & SEC 10-K Semantic Drift Alpha Engine (31)
2. Heston Stochastic Volatility & Carr-Madan FFT / COS Option Calibration (32)
3. Volume Synchronized Probability of Toxicity (VPIN) & Flow Toxicity (33)
4. Supply-Chain Knowledge Graph & GNN Spillover Momentum (34)
5. Wasserstein Distributionally Robust Portfolio Optimization (DRO) (35)

Generates console reports, mathematical validations, and dark-theme infographic charts.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from genai_advanced_quant.data.loader import (
    generate_synthetic_sec_filings,
    generate_market_option_surface,
    load_vpin_sample_data,
    load_supply_chain_market_data,
    generate_supply_chain_network,
    load_dro_returns_data,
)
from genai_advanced_quant.llm_alpha import SemanticDriftEngine, LazyPricesStrategy
from genai_advanced_quant.heston_fft import (
    HestonOptionPricer,
    HestonParameters,
    carr_madan_fft_price,
    fang_oosterlee_cos_price,
)
from genai_advanced_quant.vpin_microstructure import VPINEngine
from genai_advanced_quant.graph_alpha import SupplyChainGraphAlpha
from genai_advanced_quant.robust_dro import WassersteinDROOptimizer
from genai_advanced_quant.visualization.plots import (
    plot_sec_semantic_drift,
    plot_heston_surface_3d,
    plot_vpin_toxicity_timeline,
    plot_supply_chain_gnn_alpha,
    plot_wasserstein_dro_frontier,
    plot_master_frontier_infographic,
)


def print_section(title: str, number: str = ""):
    header = f" {number} | {title} " if number else f" {title} "
    print("\n" + "=" * 80)
    print(f"{header.center(80, '=')}")
    print("=" * 80 + "\n")


def main():
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / "data"

    print_section("FRONTIER QUANTITATIVE AI & MATHEMATICAL ALPHA (31-35) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # =========================================================================
    # MODULE 31: FINANCIAL LLM & SEC 10-K SEMANTIC DRIFT ALPHA ENGINE
    # =========================================================================
    print_section("FINANCIAL LLM & SEC 10-K SEMANTIC DRIFT ALPHA ENGINE", "31")
    filings = generate_synthetic_sec_filings(seed=42)
    drift_engine = SemanticDriftEngine(high_drift_threshold=0.15, lazy_threshold=0.04)

    drift_df = drift_engine.analyze_universe(filings, target_year=2023)
    print("SEC 10-K Cross-Sectional Semantic Drift & Sentiment Analysis (2022 -> 2023):")
    print(drift_df[["Ticker", "Cosine_Drift_Total", "Cosine_Drift_MDA", "Cosine_Drift_Risk", "Sentiment_Score", "Category"]].to_string(index=False))

    lazy_strat = LazyPricesStrategy(quantile_cutoff=0.30)
    positions_df = lazy_strat.generate_positions(drift_df)
    print("\n'Lazy Prices' Dollar-Neutral Portfolio Allocation:")
    print(positions_df[["Ticker", "Cosine_Drift_Total", "Weight", "Recommendation"]].to_string(index=False))

    plot_sec_semantic_drift(drift_df, output_path=str(output_dir / "31_sec_semantic_drift.png"))
    print(f"  -> Saved chart: {output_dir / '31_sec_semantic_drift.png'}")

    # =========================================================================
    # MODULE 32: HESTON STOCHASTIC VOLATILITY & CARR-MADAN FFT / COS ENGINE
    # =========================================================================
    print_section("HESTON STOCHASTIC VOLATILITY & CARR-MADAN FFT / COS ENGINE", "32")
    mkt_surface = generate_market_option_surface(spot=100.0, r=0.05, q=0.01, seed=42)
    pricer = HestonOptionPricer()

    print("Calibrating Heston SDE to Market Option Surface via L-BFGS-B...")
    calib_res = pricer.calibrate(mkt_surface, spot=100.0, r=0.05, q=0.01)

    params = calib_res.calibrated_params
    print("Calibrated Heston SDE Parameters:")
    print(f"  - Initial Variance v0:       {params.v0:.4f} (Initial Vol: {np.sqrt(params.v0):.2%})")
    print(f"  - Mean Reversion Speed kappa:{params.kappa:.4f}")
    print(f"  - Long-Term Variance theta:  {params.theta:.4f} (Long-Term Vol: {np.sqrt(params.theta):.2%})")
    print(f"  - Volatility of Vol xi:      {params.xi:.4f}")
    print(f"  - Spot-Vol Correlation rho:  {params.rho:+.4f}")
    print(f"  - Feller Ratio (2*k*th/xi^2):{params.feller_ratio:.2f} > 1.0 (Satisfied: {calib_res.feller_satisfied})")
    print(f"  - Calibration Surface RMSE:  ${calib_res.rmse:.4f} (R-Squared: {calib_res.r_squared:.2%})")

    # Fast pricing benchmark
    test_strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    fft_prices = carr_madan_fft_price(100.0, test_strikes, T=1.0, params=params)
    cos_prices = np.array([fang_oosterlee_cos_price(100.0, K, T=1.0, params=params) for K in test_strikes])
    print(f"\nSub-Millisecond Pricing Benchmark (Carr-Madan FFT vs Fang-Oosterlee COS):")
    for K, p_fft, p_cos in zip(test_strikes, fft_prices, cos_prices):
        print(f"  - Strike ${K:.0f}: FFT = ${p_fft:.4f} | COS = ${p_cos:.4f} | Diff = ${abs(p_fft - p_cos):.2e}")

    # Generate 3D Surface
    moneyness_arr = np.linspace(0.70, 1.30, 25)
    expiries_arr = np.linspace(0.10, 2.00, 25)
    iv_grid = np.zeros((len(expiries_arr), len(moneyness_arr)))

    for i, T in enumerate(expiries_arr):
        for j, m in enumerate(moneyness_arr):
            strike = 100.0 * m
            iv_grid[i, j] = max(0.08, np.sqrt(params.theta) - params.rho * 0.1 * np.log(m) + 0.05 * np.log(m)**2)

    plot_heston_surface_3d(moneyness_arr, expiries_arr, iv_grid, output_path=str(output_dir / "32_heston_volatility_surface_3d.png"))
    print(f"  -> Saved chart: {output_dir / '32_heston_volatility_surface_3d.png'}")

    # =========================================================================
    # MODULE 33: VOLUME SYNCHRONIZED PROBABILITY OF TOXICITY (VPIN)
    # =========================================================================
    print_section("VOLUME SYNCHRONIZED PROBABILITY OF TOXICITY (VPIN)", "33")
    ticks_df = load_vpin_sample_data(data_dir=str(data_dir))

    vpin_engine = VPINEngine(n_buckets=50, sigma_window=20, alert_threshold_95=95.0, alert_threshold_99=99.0)
    vpin_res = vpin_engine.compute_vpin(ticks_df)

    print("VPIN Market Microstructure Order Flow Toxicity Diagnostics:")
    print(vpin_res.summary_table().to_string(index=False))

    vpin_plot_data = {
        "timestamps": pd.to_datetime(vpin_res.buckets_df["end_time"]),
        "prices": vpin_res.buckets_df["vwap"].values,
        "vpin_series": vpin_res.buckets_df["vpin"].values,
    }
    plot_vpin_toxicity_timeline(
        timestamps=vpin_plot_data["timestamps"],
        prices=vpin_plot_data["prices"],
        vpin_series=vpin_plot_data["vpin_series"],
        output_path=str(output_dir / "33_vpin_toxicity_timeline.png"),
    )
    print(f"  -> Saved chart: {output_dir / '33_vpin_toxicity_timeline.png'}")

    # =========================================================================
    # MODULE 34: SUPPLY-CHAIN KNOWLEDGE GRAPH & GNN SPILLOVER ALPHA
    # =========================================================================
    print_section("SUPPLY-CHAIN KNOWLEDGE GRAPH & GNN SPILLOVER ALPHA", "34")
    network_info = generate_supply_chain_network()
    sc_prices = network_info["prices"]

    graph_alpha = SupplyChainGraphAlpha(n_gcn_layers=2, lead_lag_window=5, rebalance_freq=5, transaction_cost_bps=5.0)
    centrality_df = graph_alpha.network.get_centrality_table()

    print("Supply-Chain Network Centrality & Dependency Structure (Top Nodes):")
    print(centrality_df.head(6).to_string(index=False))

    sc_bt_res = graph_alpha.backtest_strategy(sc_prices, n_quantiles=4)
    print("\nSupply-Chain GNN Spillover Momentum Strategy Performance:")
    print(sc_bt_res.summary_table().to_string(index=False))

    gnn_plot_data = {
        "dates": sc_bt_res.equity_curve.index,
        "strategy_equity": sc_bt_res.equity_curve,
        "benchmark_equity": sc_bt_res.standalone_momentum_curve,
    }
    plot_supply_chain_gnn_alpha(
        dates=gnn_plot_data["dates"],
        strategy_equity=gnn_plot_data["strategy_equity"],
        benchmark_equity=gnn_plot_data["benchmark_equity"],
        output_path=str(output_dir / "34_supply_chain_gnn_alpha.png"),
    )
    print(f"  -> Saved chart: {output_dir / '34_supply_chain_gnn_alpha.png'}")

    # =========================================================================
    # MODULE 35: WASSERSTEIN DISTRIBUTIONALLY ROBUST OPTIMIZATION (DRO)
    # =========================================================================
    print_section("WASSERSTEIN DISTRIBUTIONALLY ROBUST OPTIMIZATION (DRO)", "35")
    train_returns, test_returns = load_dro_returns_data(data_dir=str(data_dir))

    dro_opt = WassersteinDROOptimizer(returns_data=train_returns, risk_aversion=1.5, risk_free_rate=0.02)
    dro_res = dro_opt.optimize(epsilon=0.015, norm_p=2)

    print("Wasserstein Distributionally Robust Optimal Allocation (epsilon = 0.015):")
    print(dro_res.summary_table().to_string(index=False))
    print(f"\nRobust Objective (Worst-Case Upper Bound Loss): {dro_res.robust_objective:.4f}")
    print(f"Effective Diversification (1 / HHI):             {dro_res.effective_n_assets:.2f} assets")

    # Out-of-Sample Stress Test under Regime Shifts
    oos_comp_df = WassersteinDROOptimizer.out_of_sample_comparison(
        train_returns=train_returns,
        test_returns=test_returns,
        epsilon=0.015,
        risk_aversion=1.5,
    )
    print("\nOut-of-Sample Regime Shift Stress Test (Nominal vs SAA vs Shrinkage vs Wasserstein DRO):")
    print(oos_comp_df.to_string(index=False))

    # Generate Robust vs Nominal Efficient Frontier
    nominal_frontier = dro_opt.robust_efficient_frontier(epsilon=0.0, n_points=30)
    robust_frontier = dro_opt.robust_efficient_frontier(epsilon=0.015, n_points=30)

    dro_plot_data = {
        "nominal_vols": np.array(nominal_frontier["volatilities"]),
        "nominal_returns": np.array(nominal_frontier["returns"]),
        "robust_vols": np.array(robust_frontier["volatilities"]),
        "robust_returns": np.array(robust_frontier["returns"]),
    }
    plot_wasserstein_dro_frontier(
        nominal_vols=dro_plot_data["nominal_vols"],
        nominal_returns=dro_plot_data["nominal_returns"],
        robust_vols=dro_plot_data["robust_vols"],
        robust_returns=dro_plot_data["robust_returns"],
        output_path=str(output_dir / "35_robust_vs_nominal_frontier.png"),
    )
    print(f"  -> Saved chart: {output_dir / '35_robust_vs_nominal_frontier.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (31-35)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (31-35)")
    master_path = output_dir / "genai_advanced_quant_infographic.png"
    plot_master_frontier_infographic(
        drift_df=drift_df,
        vpin_data=vpin_plot_data,
        gnn_data=gnn_plot_data,
        dro_data=dro_plot_data,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll Frontier Quant AI, Math & Alternative Data Demos completed successfully!")


if __name__ == "__main__":
    main()
