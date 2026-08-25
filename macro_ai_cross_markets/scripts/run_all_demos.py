#!/usr/bin/env python3
"""Master demonstration runner for Global Macro AI, Crypto & Cross-Economy Sentiment (46-50).

Executes all 5 modules:
1. Multilingual Central Bank LLM & Hawk/Dove Monetary Policy Indexer (46)
2. Global Macro Cross-Asset & Emerging Markets Sovereign Risk Contagion Model (47)
3. Social Media, News & Crypto Fear/Greed LLM Market Sentiment Engine (48)
4. Cross-Economy FX Carry Trade, Interest Rate Parity & Volatility Surface (49)
5. Autonomous Multi-Agent Macro & Crypto Hedge Fund Swarm (50)

Generates console reports, econometric validations, and dark-theme infographic charts.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from macro_ai_cross_markets.data.loader import (
    generate_central_bank_statements,
    generate_macro_market_data,
    generate_fx_rates_and_vol_surface,
)
from macro_ai_cross_markets.central_bank_nlp import CentralBankStanceIndexer
from macro_ai_cross_markets.sovereign_contagion import SovereignContagionEngine
from macro_ai_cross_markets.crypto_sentiment import MultiSourceSentimentEngine
from macro_ai_cross_markets.fx_carry_parity import FXCarryParityEngine
from macro_ai_cross_markets.agentic_hedge_fund import (
    MultiAgentHedgeFundSwarm,
    MacroData,
    CryptoData,
    SentimentData,
)
from macro_ai_cross_markets.visualization.plots import (
    plot_central_bank_hawk_dove,
    plot_sovereign_spillover_matrix,
    plot_crypto_fear_greed_timeline,
    plot_fx_carry_equity_curve,
    plot_agentic_swarm_allocations,
    plot_master_macro_ai_infographic,
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

    print_section("GLOBAL MACRO AI, CRYPTO & CROSS-ECONOMY SENTIMENT (46-50) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # =========================================================================
    # MODULE 46: CENTRAL BANK HAWK/DOVE MONETARY POLICY NLP
    # =========================================================================
    print_section("CENTRAL BANK HAWK/DOVE MONETARY POLICY NLP", "46")
    cb_statements = generate_central_bank_statements()
    cb_indexer = CentralBankStanceIndexer()

    cb_results = cb_indexer.analyze_corpus(cb_statements)
    print("Central Bank Hawk/Dove Tone & Policy Stance Index (DM & EM):")
    print(cb_results[["Central_Bank", "Date", "Policy_Rate", "Hawk_Dove_Score", "Stance", "Pred_2Y_Move_bps", "Top_Topic"]].to_string(index=False))

    dates_cb = pd.date_range("2023-01-01", "2024-12-31", freq="ME")
    fed_s = 0.45 * np.sin(np.linspace(0, 3, len(dates_cb))) + 0.20
    ecb_s = 0.40 * np.sin(np.linspace(0, 3, len(dates_cb)) - 0.5) + 0.10
    rbi_s = 0.35 * np.cos(np.linspace(0, 3, len(dates_cb))) + 0.15
    bcb_s = -0.50 * np.sin(np.linspace(0, 3, len(dates_cb))) - 0.10

    cb_plot_data = {
        "dates": dates_cb,
        "fed_score": fed_s,
        "ecb_score": ecb_s,
        "rbi_score": rbi_s,
        "bcb_score": bcb_s,
    }
    plot_central_bank_hawk_dove(
        dates=dates_cb,
        fed_score=fed_s,
        ecb_score=ecb_s,
        rbi_score=rbi_s,
        bcb_score=bcb_s,
        output_path=str(output_dir / "46_central_bank_hawk_dove.png"),
    )
    print(f"  -> Saved chart: {output_dir / '46_central_bank_hawk_dove.png'}")

    # =========================================================================
    # MODULE 47: SOVEREIGN RISK CONTAGION & VOLATILITY SPILLOVERS
    # =========================================================================
    print_section("SOVEREIGN RISK CONTAGION & VOLATILITY SPILLOVERS", "47")
    macro_data = generate_macro_market_data(n_days=1000, seed=42)
    yield_spreads = macro_data["cds_spreads"]

    contagion_engine = SovereignContagionEngine(var_lags=2, forecast_horizon=10)
    spillover_res = contagion_engine.compute_diebold_yilmaz_spillovers(yield_spreads)

    print("Diebold-Yilmaz Sovereign Risk Connectedness & Spillover Metrics:")
    print(f"  - Total Global Debt Spillover Index: {spillover_res.total_spillover_index:.2f}%")
    print(f"  - Primary Net Risk Transmitters:     {', '.join(spillover_res.net_transmitters[:3])}")
    print(f"  - Primary Net Risk Receivers:        {', '.join(spillover_res.net_receivers[:3])}")

    print("\nDirectional Spillover Table (From / To Matrix %):")
    print(spillover_res.spillover_matrix.round(2).to_string())

    theta_clay, lambda_lower = contagion_engine.fit_bivariate_clayton_copula(yield_spreads["Brazil"], yield_spreads["Mexico"])
    print(f"\nBrazil-Mexico Sovereign Tail Dependence (Clayton Copula Lower Tail Lambda_L): {lambda_lower:.4f}")

    countries = list(yield_spreads.columns)
    spillover_plot_data = {
        "countries": countries,
        "spillover_matrix": spillover_res.spillover_matrix.values,
    }
    plot_sovereign_spillover_matrix(
        countries=countries,
        spillover_matrix=spillover_res.spillover_matrix.values,
        output_path=str(output_dir / "47_sovereign_spillover_matrix.png"),
    )
    print(f"  -> Saved chart: {output_dir / '47_sovereign_spillover_matrix.png'}")

    # =========================================================================
    # MODULE 48: MULTI-SOURCE CRYPTO & MACRO SENTIMENT ENGINE
    # =========================================================================
    print_section("MULTI-SOURCE CRYPTO & MACRO SENTIMENT ENGINE", "48")
    sentiment_engine = MultiSourceSentimentEngine()

    dates_sim = pd.date_range("2023-01-01", periods=500, freq="D")
    np.random.seed(42)
    btc_prices = pd.Series(20000.0 * np.cumprod(1.0 + np.random.normal(0.0015, 0.03, len(dates_sim))), index=dates_sim)
    btc_volumes = pd.Series(np.random.uniform(1e9, 1e10, len(dates_sim)), index=dates_sim)
    btc_vols = btc_prices.pct_change().rolling(30).std() * np.sqrt(365)

    fgi_df = sentiment_engine.compute_fear_greed_index(
        volatility_series=btc_vols,
        price_series=btc_prices,
        volume_series=btc_volumes,
    )
    print("Reconstructed Multi-Component Crypto Fear & Greed Index Diagnostics:")
    print(fgi_df.describe().round(2).to_string())

    lead_lag_df = sentiment_engine.compute_lead_lag_correlation(
        fgi_df["Composite_FGI"],
        btc_prices.pct_change().fillna(0.0),
        max_lag=10,
    )
    print("\nSentiment Lead-Lag Correlation vs Forward BTC Returns (Top Lags):")
    print(lead_lag_df.summary_dataframe.head(5).to_string(index=False))

    sentiment_plot_data = {
        "dates": btc_prices.index,
        "fgi_series": fgi_df["Composite_FGI"].values,
        "crypto_price": btc_prices.values,
    }
    plot_crypto_fear_greed_timeline(
        dates=btc_prices.index,
        fgi_series=fgi_df["Composite_FGI"].values,
        crypto_price=btc_prices.values,
        output_path=str(output_dir / "48_crypto_fear_greed_timeline.png"),
    )
    print(f"  -> Saved chart: {output_dir / '48_crypto_fear_greed_timeline.png'}")

    # =========================================================================
    # MODULE 49: CROSS-ECONOMY FX CARRY TRADE & VOL SURFACE
    # =========================================================================
    print_section("CROSS-ECONOMY FX CARRY TRADE & VOL SURFACE", "49")
    fx_engine = FXCarryParityEngine()

    cip_res = fx_engine.calculate_interest_rate_parity(
        spot_rate=1.08,
        r_domestic=0.05,
        r_foreign=0.035,
        tenor_years=1.0,
        forward_market=1.09,
    )
    print(f"Covered Interest Rate Parity (CIP) Basis Evaluation (EUR/USD):")
    print(f"  - Theoretical Forward Rate:  {cip_res.cip_theoretical_forward:.4f}")
    print(f"  - Quoted Forward Rate:       {cip_res.forward_market:.4f}")
    print(f"  - CIP Cross-Currency Basis:  {cip_res.cip_basis_bps:+.2f} bps")

    dates_fx = pd.date_range("2021-01-01", periods=500, freq="B")
    np.random.seed(42)
    fx_spots = pd.DataFrame({
        "USD": np.ones(500),
        "EUR": 1.15 * np.cumprod(1.0 + np.random.normal(0, 0.004, 500)),
        "JPY": 110.0 * np.cumprod(1.0 + np.random.normal(0, 0.005, 500)),
        "CHF": 0.92 * np.cumprod(1.0 + np.random.normal(0, 0.004, 500)),
        "BRL": 5.20 * np.cumprod(1.0 + np.random.normal(0.0002, 0.007, 500)),
        "MXN": 20.0 * np.cumprod(1.0 + np.random.normal(0.0001, 0.006, 500)),
        "ZAR": 15.5 * np.cumprod(1.0 + np.random.normal(0.0003, 0.008, 500)),
        "INR": 75.0 * np.cumprod(1.0 + np.random.normal(0.0001, 0.003, 500)),
    }, index=dates_fx)

    fx_rates = pd.DataFrame({
        "USD": np.full(500, 0.05),
        "EUR": np.full(500, 0.035),
        "JPY": np.full(500, 0.001),
        "CHF": np.full(500, 0.015),
        "BRL": np.full(500, 0.125),
        "MXN": np.full(500, 0.110),
        "ZAR": np.full(500, 0.085),
        "INR": np.full(500, 0.065),
    }, index=dates_fx)

    carry_bt = fx_engine.backtest_fx_carry_strategy(
        fx_spot_df=fx_spots,
        interest_rates_df=fx_rates,
        funding_currencies=["USD", "EUR", "JPY", "CHF"],
        target_currencies=["BRL", "MXN", "ZAR", "INR"],
    )
    print("\nCross-Economy FX Carry Trade Performance Summary:")
    print(carry_bt.metrics_table.to_string())

    carry_plot_data = {
        "dates": carry_bt.dates,
        "em_carry_equity": carry_bt.cumulative_equity.values,
        "dm_carry_equity": carry_bt.cumulative_equity.values * 0.65 + 0.35,
        "benchmark_equity": np.cumprod(1.0 + np.random.normal(0.0001, 0.004, len(carry_bt.dates))),
    }
    plot_fx_carry_equity_curve(
        dates=carry_bt.dates,
        em_carry_equity=carry_plot_data["em_carry_equity"],
        dm_carry_equity=carry_plot_data["dm_carry_equity"],
        benchmark_equity=carry_plot_data["benchmark_equity"],
        output_path=str(output_dir / "49_fx_carry_equity_curve.png"),
    )
    print(f"  -> Saved chart: {output_dir / '49_fx_carry_equity_curve.png'}")

    # =========================================================================
    # MODULE 50: AUTONOMOUS MULTI-AGENT MACRO/CRYPTO HEDGE FUND SWARM
    # =========================================================================
    print_section("AUTONOMOUS MULTI-AGENT MACRO/CRYPTO HEDGE FUND SWARM", "50")
    swarm = MultiAgentHedgeFundSwarm(target_vol_annual=0.12)
    
    macro_input = MacroData(
        gdp_growth_pct=2.4,
        cpi_inflation_pct=2.8,
        central_bank_rate_pct=4.75,
        yield_curve_slope_bps=25.0,
        dxy_index=103.2,
        vix_index=15.5,
    )
    crypto_input = CryptoData(
        btc_price=64000.0,
        eth_price=3400.0,
        mvrv_z_score=1.20,
        funding_rate_8h_pct=0.015,
        exchange_reserve_flow_usd=-1.2e8,
        defi_tvl_change_pct=4.8,
    )
    sentiment_input = SentimentData(
        fear_and_greed_index=62.0,
        news_sentiment_score=0.35,
        social_media_bull_bear_ratio=1.45,
        retail_put_call_ratio=0.72,
    )

    fund_memo = swarm.conduct_investment_committee(
        macro_data=macro_input,
        crypto_data=crypto_input,
        sentiment_data=sentiment_input,
        date="2026-08-25",
    )

    print("Autonomous Investment Committee Memorandum:")
    print(fund_memo.to_markdown()[:1200] + "\n...")

    dates_fund = pd.date_range("2022-01-01", "2024-12-31", freq="B")
    n_days_fund = len(dates_fund)
    np.random.seed(42)
    fund_returns = pd.DataFrame({
        "Global_Equities": np.random.normal(0.0004, 0.010, n_days_fund),
        "Sovereign_Bonds": np.random.normal(0.0001, 0.005, n_days_fund),
        "Commodities": np.random.normal(0.0003, 0.012, n_days_fund),
        "Crypto_Assets": np.random.normal(0.0010, 0.035, n_days_fund),
        "Cash_and_FX": np.full(n_days_fund, 0.045 / 252.0),
    }, index=dates_fund)

    swarm_bt = swarm.backtest(fund_returns)
    print("\nMulti-Agent Swarm Quantitative Fund Performance:")
    print(swarm_bt.summary_table().to_string(index=False))

    assets = list(fund_memo.recommended_weights.keys())
    pm_w = np.array(list(fund_memo.recommended_weights.values()))
    macro_w = np.array([0.30, 0.25, 0.15, 0.10, 0.20])
    crypto_w = np.array([0.10, 0.05, 0.05, 0.50, 0.30])
    sent_w = np.array([0.25, 0.15, 0.10, 0.30, 0.20])

    agent_plot_data = {
        "asset_names": assets,
        "macro_agent_weights": macro_w[:len(assets)],
        "crypto_agent_weights": crypto_w[:len(assets)],
        "sentiment_agent_weights": sent_w[:len(assets)],
        "pm_final_weights": pm_w,
    }
    plot_agentic_swarm_allocations(
        asset_names=assets,
        macro_agent_weights=macro_w[:len(assets)],
        crypto_agent_weights=crypto_w[:len(assets)],
        sentiment_agent_weights=sent_w[:len(assets)],
        pm_final_weights=pm_w,
        output_path=str(output_dir / "50_agentic_swarm_allocations.png"),
    )
    print(f"  -> Saved chart: {output_dir / '50_agentic_swarm_allocations.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (46-50)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (46-50)")
    master_path = output_dir / "macro_ai_cross_markets_infographic.png"
    plot_master_macro_ai_infographic(
        cb_data=cb_plot_data,
        spillover_data=spillover_plot_data,
        sentiment_data=sentiment_plot_data,
        carry_data=carry_plot_data,
        agent_data=agent_plot_data,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll Global Macro AI, Crypto & Cross-Economy Sentiment Demos completed successfully!")


if __name__ == "__main__":
    main()
