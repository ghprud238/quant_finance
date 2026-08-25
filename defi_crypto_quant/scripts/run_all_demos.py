#!/usr/bin/env python3
"""Master demonstration runner for DeFi, AMM Liquidity & Crypto Quant (41-45).

Executes all 5 modules:
1. Constant Function Market Makers & Uniswap v2/v3 Concentrated Liquidity (41)
2. Impermanent Loss & Loss-Versus-Rebalancing (LVR) Engine (42)
3. Cross-DEX Flash Loans, Triangular Arbitrage & MEV Searcher (43)
4. Crypto Perpetual Futures, Funding Rate Arbitrage & Basis Trading (44)
5. On-Chain Blockchain Telemetry, MVRV, Exchange Flows & Whale Alpha (45)

Generates console reports, DeFi analytics, and dark-theme infographic charts.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from defi_crypto_quant.data.loader import (
    generate_crypto_market_data,
    generate_uniswap_v3_liquidity_ticks,
    generate_perpetual_funding_data, generate_synthetic_funding_rates,
    generate_onchain_metrics_data,
)
from defi_crypto_quant.uniswap_amm import (
    ConstantProductAMM,
    ConcentratedLiquidityAMM,
    StableswapAMM,
)
from defi_crypto_quant.lvr_impermanent_loss import (
    ImpermanentLossCalculator,
    LossVersusRebalancingEngine,
)
from defi_crypto_quant.mev_arbitrage import (
    CrossDEXArbitrageEngine,
    LiquidityPool,
    PoolType,
    MEVSandwichSimulator,
)
from defi_crypto_quant.perp_funding import (
    PerpetualFundingEngine,
    CashAndCarryBasisTrader,
)
from defi_crypto_quant.onchain_alpha import OnChainAlphaEngine
from defi_crypto_quant.visualization.plots import (
    plot_v3_liquidity_density,
    plot_lvr_vs_fee_revenue,
    plot_mev_sandwich_dynamics,
    plot_perp_funding_basis_equity,
    plot_onchain_mvrv_regimes,
    plot_master_defi_crypto_infographic,
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

    print_section("DEFI, AMM LIQUIDITY & CRYPTO QUANTITATIVE FINANCE (41-45) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # =========================================================================
    # MODULE 41: UNISWAP V2/V3 CONCENTRATED LIQUIDITY AMM
    # =========================================================================
    print_section("UNISWAP V2/V3 CONCENTRATED LIQUIDITY AMM", "41")
    amm_v3 = ConcentratedLiquidityAMM(current_price=3000.0, fee_tier=0.0030)
    pos = amm_v3.mint_position(
        owner="WhaleLP",
        price_lower=2500.0,
        price_upper=3500.0,
        amount_x=10.0,
        amount_y=30_000.0,
    )

    print("Uniswap v3 Concentrated Liquidity Position ($2,500 to $3,500 Range):")
    print(f"  - Active Liquidity (L):       {pos.liquidity:,.2f}")
    print(f"  - Capital Efficiency:         {amm_v3.capital_efficiency_multiplier(2500.0, 3500.0):.2f}x vs v2")
    print(f"  - Locked ETH / Locked USDC:   {pos.amount_x:.4f} ETH / ${pos.amount_y:,.2f} USDC")

    swap_res = amm_v3.swap(amount_in=2.0, token_in="ETH")
    print(f"\nSwap Execution Result (Selling 2.0 ETH into v3 Pool):")
    print(f"  - Received USDC:              ${swap_res.amount_out:,.2f}")
    print(f"  - Execution Price:            ${swap_res.execution_price:,.2f} / ETH")
    print(f"  - Price Impact:               {swap_res.price_impact_pct:+.3f}%")
    print(f"  - Protocol Fee Collected:     ${swap_res.fee_paid:,.2f}")

    prices_arr = np.linspace(1500.0, 4500.0, 100)
    v2_liq = np.full_like(prices_arr, 100.0)
    v3_liq = np.where((prices_arr >= 2500.0) & (prices_arr <= 3500.0), 450.0, 10.0)

    amm_plot_data = {
        "prices": prices_arr,
        "v2_liquidity": v2_liq,
        "v3_liquidity": v3_liq,
        "current_price": 3000.0,
    }
    plot_v3_liquidity_density(
        prices=prices_arr,
        v2_liquidity=v2_liq,
        v3_liquidity=v3_liq,
        current_price=3000.0,
        output_path=str(output_dir / "41_v3_concentrated_liquidity.png"),
    )
    print(f"  -> Saved chart: {output_dir / '41_v3_concentrated_liquidity.png'}")

    # =========================================================================
    # MODULE 42: IMPERMANENT LOSS & LOSS-VERSUS-REBALANCING (LVR)
    # =========================================================================
    print_section("IMPERMANENT LOSS & LOSS-VERSUS-REBALANCING (LVR)", "42")
    mkt_data = generate_crypto_market_data(n_days=180, freq="1h", seed=42)
    eth_prices = mkt_data[("ETH/USDC", "Close")]
    eth_volumes = mkt_data[("ETH/USDC", "Volume")]

    lvr_engine = LossVersusRebalancingEngine(pool_type="v3", fee_rate=0.0030)
    lp_sim = lvr_engine.simulate_lp_performance(
        price_series=eth_prices,
        volume_series=eth_volumes,
        initial_capital_usd=100_000.0,
        price_lower=2500.0,
        price_upper=3500.0,
    )

    print("Uniswap v3 LP Performance & LVR Adverse Selection Diagnostics (180 Days):")
    print(lp_sim.summary_table.to_string(index=False))

    lvr_plot_data = {
        "timestamps": lp_sim.timestamps,
        "cumulative_fees": lp_sim.cumulative_fees_usd,
        "cumulative_lvr": lp_sim.cumulative_lvr_usd,
        "net_lp_pnl": lp_sim.cumulative_net_pnl_usd,
    }
    plot_lvr_vs_fee_revenue(
        timestamps=lp_sim.timestamps,
        cumulative_fees=lp_sim.cumulative_fees_usd,
        cumulative_lvr=lp_sim.cumulative_lvr_usd,
        net_lp_pnl=lp_sim.cumulative_net_pnl_usd,
        output_path=str(output_dir / "42_lvr_vs_fee_revenue.png"),
    )
    print(f"  -> Saved chart: {output_dir / '42_lvr_vs_fee_revenue.png'}")

    # =========================================================================
    # MODULE 43: CROSS-DEX FLASH LOANS & ATOMIC MEV ARBITRAGE
    # =========================================================================
    print_section("CROSS-DEX FLASH LOANS & ATOMIC MEV ARBITRAGE", "43")
    p_uni = LiquidityPool("Uniswap_WETH_USDC", PoolType.UNISWAP_V2, "WETH", "USDC", 2500.0, 7_500_000.0, fee=0.0030)
    p_sushi = LiquidityPool("Sushiswap_WETH_USDC", PoolType.SUSHISWAP, "WETH", "USDC", 1800.0, 5_580_000.0, fee=0.0030)

    mev_engine = CrossDEXArbitrageEngine(eth_price_usd=3000.0)
    arb_res = mev_engine.evaluate_spatial_arbitrage(p_uni, p_sushi, token_borrow="USDC")

    print("Cross-DEX Spatial Flash Loan Arbitrage:")
    print(f"  - Optimal Borrow Size:       ${arb_res.optimal_input:,.2f} USDC")
    print(f"  - Gross Revenue:             ${arb_res.final_output:,.2f} USDC")
    print(f"  - Flash Loan Fee (0.09%):    ${arb_res.flash_loan_fee:,.2f}")
    print(f"  - Gas Cost:                  ${arb_res.gas_cost:,.2f}")
    print(f"  - Net Arbitrage Profit:      ${arb_res.net_profit:+,.2f} ({arb_res.return_on_capital_pct:+.2f}% RoC)")

    sandwich_sim = MEVSandwichSimulator(eth_price_usd=3000.0, priority_fee_gwei=40.0, builder_bribe_pct=0.85)
    sw_res = sandwich_sim.simulate_sandwich(
        pool=p_uni,
        victim_token_in="USDC",
        victim_amount_in=250_000.0,
        victim_max_slippage_pct=0.01,
    )
    print("\nMempool Atomic MEV Sandwich Attack Simulation:")
    print(f"  - Victim Order:              $250,000.00 USDC -> WETH (1.0% Max Slippage)")
    print(f"  - Frontrun Borrow Amount:    ${sw_res.frontrun_amount_in:,.2f} USDC")
    print(f"  - Victim Slippage Absorbed:  {sw_res.victim_slippage_drag_pct:.2f}%")
    print(f"  - Gross MEV Revenue:         ${sw_res.gross_mev_profit:+,.2f}")
    print(f"  - Flashbots Builder Bribe:   ${sw_res.builder_bribe_usd:,.2f} (85%)")
    print(f"  - Net Searcher Profit:       ${sw_res.net_searcher_profit:+,.2f}")

    stages = ["Initial Pool", "Post-Frontrun", "Post-Victim", "Post-Backrun"]
    pool_prices = [3000.0, 3038.5, 3068.2, 3000.4]

    mev_plot_data = {
        "stages": stages,
        "pool_prices": pool_prices,
    }
    plot_mev_sandwich_dynamics(
        stages=stages,
        pool_prices=pool_prices,
        output_path=str(output_dir / "43_mev_sandwich_dynamics.png"),
    )
    print(f"  -> Saved chart: {output_dir / '43_mev_sandwich_dynamics.png'}")

    # =========================================================================
    # MODULE 44: PERPETUAL FUTURES & CASH-AND-CARRY BASIS TRADING
    # =========================================================================
    print_section("PERPETUAL FUTURES & CASH-AND-CARRY BASIS TRADING", "44")
    funding_df = generate_synthetic_funding_rates(n_periods=1095, base_rate_annual=0.14, seed=42)

    basis_trader = CashAndCarryBasisTrader(initial_capital_usd=1_000_000.0, spot_allocation_pct=0.50, staking_yield_apy=0.035)
    basis_res = basis_trader.backtest(funding_df)

    print("Cash-and-Carry Delta-Neutral Strategy Performance ($1M Portfolio):")
    print(basis_res.summary_table().to_string(index=False))

    perp_plot_data = {
        "dates": basis_res.equity_curve.index,
        "basis_equity": basis_res.equity_curve.values,
        "spot_price": funding_df["Spot_Price"].values,
    }
    plot_perp_funding_basis_equity(
        dates=basis_res.equity_curve.index,
        basis_equity=basis_res.equity_curve.values,
        spot_price=funding_df["Spot_Price"].values,
        output_path=str(output_dir / "44_perp_funding_basis_equity.png"),
    )
    print(f"  -> Saved chart: {output_dir / '44_perp_funding_basis_equity.png'}")

    # =========================================================================
    # MODULE 45: ON-CHAIN BLOCKCHAIN TELEMETRY & WHALE ALPHA
    # =========================================================================
    print_section("ON-CHAIN BLOCKCHAIN TELEMETRY & WHALE ALPHA", "45")
    onchain_engine = OnChainAlphaEngine()
    onchain_df = onchain_engine.generate_synthetic_onchain_data(n_days=1500, initial_price=10000.0, seed=42)

    mvrv, mvrv_z = onchain_engine.calculate_mvrv(onchain_df["Market_Cap"], onchain_df["Realized_Cap"])
    net_flow, efi = onchain_engine.calculate_exchange_flow_imbalance(onchain_df["Exchange_Inflows"], onchain_df["Exchange_Outflows"])
    whale_idx = onchain_engine.calculate_whale_accumulation_index(onchain_df["Whale_Balance"])
    addr_vel = onchain_engine.calculate_address_velocity(onchain_df["Active_Addresses"])

    onchain_bt = onchain_engine.backtest_strategy(onchain_df, initial_capital=100_000.0, transaction_cost_bps=10.0)
    print("Systematic On-Chain Quantitative Strategy Performance:")
    print(onchain_bt.summary_table().to_string(index=False))

    onchain_plot_data = {
        "dates": onchain_df.index,
        "spot_price": onchain_df["Price"].values,
        "mvrv_z_score": mvrv_z.values,
    }
    plot_onchain_mvrv_regimes(
        dates=onchain_df.index,
        spot_price=onchain_df["Price"].values,
        mvrv_z_score=mvrv_z.values,
        output_path=str(output_dir / "45_onchain_mvrv_regimes.png"),
    )
    print(f"  -> Saved chart: {output_dir / '45_onchain_mvrv_regimes.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (41-45)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (41-45)")
    master_path = output_dir / "defi_crypto_quant_infographic.png"
    plot_master_defi_crypto_infographic(
        amm_data=amm_plot_data,
        lvr_data=lvr_plot_data,
        mev_data=mev_plot_data,
        perp_data=perp_plot_data,
        onchain_data=onchain_plot_data,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll DeFi, AMM Liquidity & Crypto Quantitative Finance Demos completed successfully!")


if __name__ == "__main__":
    main()
