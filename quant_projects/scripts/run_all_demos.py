#!/usr/bin/env python3
"""Master demonstration runner for Interview-Worthy Quant Projects (26-30).

Executes all 5 modules:
1. Limit Order Book / Market Microstructure Simulator (26)
2. Optimal Execution Model (Almgren-Chriss, TWAP, VWAP, IS) (27)
3. Portfolio Risk & Stress Testing Engine (28)
4. End-to-End Quant Research Pipeline (29)
5. Full Systematic Production Trading System (30)

Generates console reports, tear sheets, and dark-theme infographic charts.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from interview_quant.data.loader import load_dataset
from interview_quant.microstructure import LimitOrderBook, Order, MarketMicrostructureSimulator
from interview_quant.execution import AlmgrenChrissModel, TWAPExecutor, VWAPExecutor, ImplementationShortfallAttributor
from interview_quant.stress_testing import PortfolioStressTestingEngine, StressScenario
from interview_quant.pipeline import QuantResearchPipeline
from interview_quant.systematic_system import ProductionTradingSystem
from interview_quant.visualization.plots import (
    plot_order_book_snapshot,
    plot_execution_trajectory,
    plot_stress_test_bars,
    plot_pipeline_flowchart,
    plot_strategy_equity_curve_oos,
    plot_master_interview_infographic,
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

    print_section("INTERVIEW-WORTHY QUANT CAPSTONE (26-30) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # =========================================================================
    # MODULE 26: LIMIT ORDER BOOK / MARKET MICROSTRUCTURE SIMULATOR
    # =========================================================================
    print_section("LIMIT ORDER BOOK / MARKET MICROSTRUCTURE SIMULATOR", "26")
    lob = LimitOrderBook(name="LOB_BTC_USDT")

    # Populate L2 Book
    bids = [(100.01, 1000), (100.00, 700), (99.99, 700), (99.98, 500), (99.97, 800)]
    asks = [(100.05, 600), (100.04, 700), (100.03, 900), (100.02, 400), (100.015, 500)]

    for p, v in bids:
        lob.add_limit_order(Order(order_id=f"B_{p}", side="buy", price=p, volume=v, timestamp=1.0))
    for p, v in asks:
        lob.add_limit_order(Order(order_id=f"A_{p}", side="sell", price=p, volume=v, timestamp=1.0))

    print("Level 2 Order Book Real-Time State:")
    print(f"  - Best Bid / Best Ask:       ${lob.best_bid:.2f} / ${lob.best_ask:.2f}")
    print(f"  - Bid-Ask Spread:            ${lob.spread:.4f}")
    print(f"  - Mid Price:                 ${lob.mid_price:.4f}")
    print(f"  - Micro-Price (Volume-Wt):   ${lob.micro_price:.4f}")
    print(f"  - Order Book Imbalance (OBI):{lob.order_book_imbalance:+.4f}")

    snap_df = lob.get_snapshot_table(depth=5)
    print("\nOrder Book Snapshot Table:")
    print(snap_df.to_string())

    plot_order_book_snapshot(output_path=str(output_dir / "26_order_book_snapshot.png"))
    print(f"  -> Saved chart: {output_dir / '26_order_book_snapshot.png'}")

    # =========================================================================
    # MODULE 27: OPTIMAL EXECUTION MODEL (ALMGREN-CHRISS)
    # =========================================================================
    print_section("OPTIMAL EXECUTION MODEL (ALMGREN-CHRISS)", "27")
    ac_model = AlmgrenChrissModel(
        total_shares=1_000_000,
        horizon=1.0,
        n_intervals=20,
        volatility=0.30,
        temp_impact=2.5e-6,
        perm_impact=2.5e-7,
        initial_price=100.0,
    )

    traj = ac_model.solve_trajectory(risk_aversion=1e-6)
    print(f"Almgren-Chriss Optimal Trajectory (Liquidating 1,000,000 Shares):")
    print(f"  - Expected Cost:             ${traj.expected_shortfall:,.2f}")
    print(f"  - Cost Standard Deviation:   ${traj.std_shortfall:,.2f}")
    print(f"  - Trajectory Half-Life:      {traj.half_life:.2f} intervals")
    print(f"  - Initial Interval Trade:    {traj.trade_sizes[0]:,.0f} shares ({traj.trade_sizes[0]/1e6:.1%})")
    print(f"  - Final Interval Trade:      {traj.trade_sizes[-1]:,.0f} shares ({traj.trade_sizes[-1]/1e6:.1%})")

    # Benchmark comparison
    twap_traj = TWAPExecutor.generate_schedule(1_000_000, 20)
    print(f"\nExecution Benchmark Comparison:")
    print(f"  - TWAP Uniform Slicing:      {twap_traj[0]:,.0f} shares/interval")

    # Implementation Shortfall Decomposition
    exec_prices = 100.0 - np.linspace(0.1, 1.5, 20)
    attr = ImplementationShortfallAttributor.attribute_costs(
        total_shares=1_000_000,
        decision_price=100.0,
        arrival_price=100.0,
        terminal_price=98.5,
        trade_sizes=traj.trade_sizes,
        execution_prices=exec_prices,
        temp_impact_eta=2.5e-6,
        perm_impact_gamma=2.5e-7,
        side="sell",
    )
    print(f"  - Total Implementation Shortfall: ${attr['Total_Shortfall_Dollars']:,.2f} ({attr['Total_Shortfall_bps']:.1f} bps)")
    print(f"    * Temporary Impact Drag:       ${attr['Temporary_Impact_Dollars']:,.2f}")
    print(f"    * Permanent Impact Drag:       ${attr['Permanent_Impact_Dollars']:,.2f}")

    time_grid = np.linspace(0, 1.0, len(traj.trade_sizes))
    mkt_p = 100.0 - 1.2 * time_grid
    exec_p = mkt_p - 0.25 * (traj.trade_sizes / traj.trade_sizes[0])

    exec_plot_data = {
        "time_grid": time_grid,
        "market_price": mkt_p,
        "execution_price": exec_p,
        "shortfall_line": mkt_p - 0.45,
    }
    plot_execution_trajectory(
        time_grid=time_grid,
        market_price=mkt_p,
        execution_price=exec_p,
        shortfall_line=mkt_p - 0.45,
        output_path=str(output_dir / "27_optimal_execution_trajectory.png"),
    )
    print(f"  -> Saved chart: {output_dir / '27_optimal_execution_trajectory.png'}")

    # =========================================================================
    # MODULE 28: PORTFOLIO RISK & STRESS TESTING ENGINE
    # =========================================================================
    print_section("PORTFOLIO RISK & STRESS TESTING ENGINE", "28")
    stress_engine = PortfolioStressTestingEngine(portfolio_value=10_000_000.0)
    stress_summary_df = stress_engine.summary_table()

    print("Historical & Hypothetical Stress Scenario Results ($10M Portfolio):")
    print(stress_summary_df.to_string(index=False))

    corr_res = stress_engine.correlation_breakdown_stress(crisis_alpha=0.70)
    print(f"\nSystemic Correlation Breakdown Stress:")
    print(f"  - Base Annual Volatility:       {corr_res['base_annual_vol']:.2%}")
    print(f"  - Distressed Annual Volatility: {corr_res['crisis_annual_vol']:.2%}")
    print(f"  - Volatility Surge:             {corr_res['vol_surge_pct']:+.1f}%")

    stress_plot_data = {
        "scenario_names": [
            "2008 Financial Crisis",
            "2020 COVID Crash",
            "Rate Shock (+200bps)",
            "Market Crash (-30%)",
            "Custom Scenario",
        ],
        "pnl_impacts": [-12.4, -8.7, -6.1, -15.3, -9.2],
    }
    plot_stress_test_bars(
        scenario_names=stress_plot_data["scenario_names"],
        pnl_impacts=stress_plot_data["pnl_impacts"],
        output_path=str(output_dir / "28_stress_testing_scenarios.png"),
    )
    print(f"  -> Saved chart: {output_dir / '28_stress_testing_scenarios.png'}")

    # =========================================================================
    # MODULE 29: END-TO-END QUANT RESEARCH PIPELINE
    # =========================================================================
    print_section("END-TO-END QUANT RESEARCH PIPELINE", "29")
    market_data_df = load_dataset(data_dir=str(data_dir))

    pipeline = QuantResearchPipeline()

    def ensemble_strategy(clean_data):
        close = clean_data.xs("Close", level="Field", axis=1) if isinstance(clean_data.columns, pd.MultiIndex) else clean_data
        fast = close.rolling(20).mean()
        slow = close.rolling(100).mean()
        sig = (fast > slow).astype(float)
        return sig.div(sig.sum(axis=1).replace(0, 1), axis=0)

    pipe_results = pipeline.run_full_pipeline(market_data_df, ensemble_strategy, strategy_name="Production Alpha Pipeline")
    tear_sheet = pipe_results["tear_sheet"]
    deploy_rep = pipe_results["deploy_report"]

    print("Stage 4 Evaluation Tear Sheet:")
    print(tear_sheet.metrics_table.to_string())

    print("\nStage 5 Deployment Readiness Checklist:")
    print(deploy_rep.deployment_checklist)
    print(f"Production Readiness Score: {deploy_rep.production_readiness_score:.1f}/100")
    print(f"Deployable? {deploy_rep.is_deployable} (Drift Alert: {deploy_rep.drift_alert_level})")

    plot_pipeline_flowchart(output_path=str(output_dir / "29_pipeline_overview.png"))
    print(f"  -> Saved chart: {output_dir / '29_pipeline_overview.png'}")

    # =========================================================================
    # MODULE 30: FULL PRODUCTION SYSTEMATIC TRADING SYSTEM
    # =========================================================================
    print_section("FULL PRODUCTION SYSTEMATIC TRADING SYSTEM", "30")
    trading_sys = ProductionTradingSystem(target_annual_vol=0.10, drawdown_circuit_breaker=0.15)
    sys_res = trading_sys.run_systematic_system(market_data_df)

    print("Out-of-Sample (OOS) Multi-Year Performance Summary (2020-2024):")
    for k, v in sys_res.metrics.items():
        if "CAGR" in k or "Return" in k or "Drawdown" in k or "Volatility" in k or "Rate" in k:
            print(f"  - {k:<25}: {v:+.2%}" if isinstance(v, float) else f"  - {k:<25}: {v}")
        elif isinstance(v, float):
            print(f"  - {k:<25}: {v:.2f}")

    strategy_plot_data = {
        "dates": sys_res.dates,
        "equity_curve": sys_res.cumulative_equity,
    }
    plot_strategy_equity_curve_oos(
        dates=sys_res.dates,
        equity_curve=sys_res.cumulative_equity,
        output_path=str(output_dir / "30_strategy_equity_curve_oos.png"),
    )
    print(f"  -> Saved chart: {output_dir / '30_strategy_equity_curve_oos.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (26-30)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (26-30)")
    master_path = output_dir / "interview_quant_infographic.png"
    plot_master_interview_infographic(
        execution_data=exec_plot_data,
        stress_data=stress_plot_data,
        strategy_data=strategy_plot_data,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll Interview-Worthy Capstone Demos completed successfully!")


if __name__ == "__main__":
    main()
