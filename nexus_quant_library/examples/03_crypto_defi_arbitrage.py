"""
Example 03: Crypto, DeFi & Prediction Markets Desk Workflow
===========================================================
"""
import pandas as pd
import numpy as np
import nexus_quant as nq

# 1. Uniswap v3 Concentrated Liquidity AMM
amm_v3 = nq.defi.uniswap_v3.ConcentratedLiquidityAMM(current_price=3000.0, fee_tier=0.0030)
pos = amm_v3.mint_position(
    owner="AlphaLP",
    price_lower=2500.0,
    price_upper=3500.0,
    amount_x=10.0,
    amount_y=30000.0,
)
print("Uniswap v3 Concentrated Liquidity Position:")
print(f"  - Virtual Liquidity L: {pos.liquidity:,.2f}")
print(f"  - Capital Efficiency:  {amm_v3.capital_efficiency_multiplier(2500.0, 3500.0):.2f}x vs v2 full-range")

# Execute swap
swap_res = amm_v3.swap(amount_in=2.0, token_in="ETH")
print(f"  - Swapped 2.0 ETH -> ${swap_res.amount_out:,.2f} USDC (Exec Price: ${swap_res.execution_price:.2f})")

# 2. Loss-Versus-Rebalancing (LVR) Adverse Selection Simulation
np.random.seed(42)
t = np.linspace(0, 1, 24 * 180)
price_sim = 3000.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, len(t))))
vol_sim = np.random.uniform(500000, 2000000, len(t))
p_series = pd.Series(price_sim)
v_series = pd.Series(vol_sim)

lvr_engine = nq.defi.lvr_model.LossVersusRebalancingEngine(pool_type="v3", fee_rate=0.0030)
lvr_res = lvr_engine.simulate_lp_performance(p_series, v_series, initial_capital_usd=100000.0)
print(f"\nLVR Adverse Selection Results:")
print(f"  - Fee Revenue Earned:       ${lvr_res.total_fee_revenue_usd:,.2f}")
print(f"  - LVR Adverse Selection:    ${lvr_res.total_lvr_usd:,.2f}")
print(f"  - Net LP PnL (Fees - LVR):  ${lvr_res.net_lp_profit_usd:+,.2f}")
print(f"  - Breakeven Volatility:     {lvr_res.breakeven_volatility_annual:.2%}")

# 3. Prediction Market Arbitrage & Kelly Staking
pm_engine = nq.defi.prediction_arbitrage.PredictionMarketArbitrageEngine()
book_poly = nq.defi.prediction_arbitrage.BinaryOrderBook(
    venue="Polymarket", contract_id="FED_RATE",
    yes_bids=[nq.defi.prediction_arbitrage.OrderBookLevel(0.44, 5000)],
    yes_asks=[nq.defi.prediction_arbitrage.OrderBookLevel(0.45, 5000)],
    no_bids=[nq.defi.prediction_arbitrage.OrderBookLevel(0.47, 5000)],
    no_asks=[nq.defi.prediction_arbitrage.OrderBookLevel(0.48, 5000)],
)
intra_arb = pm_engine.check_intra_venue_arbitrage(book_poly, target_size=2000.0)
if intra_arb and intra_arb.is_executable:
    print("\nPrediction Market Intra-Venue Arbitrage:")
    print(f"  - Combined Ask Price: ${intra_arb.effective_combined_price:.3f} (< $1.00 guaranteed payout)")
    print(f"  - Net Profit:         ${intra_arb.net_profit_usd:+,.2f} (ROI: {intra_arb.net_roi_pct:+.2f}%)")
    kelly = pm_engine.calculate_kelly_fraction(0.999, intra_arb.effective_combined_price, bankroll_usd=100000.0)
    print(f"  - Fractional Kelly Stake: ${kelly.recommended_stake_usd:,.2f} ({kelly.recommended_fraction:.1%} of bankroll)")
