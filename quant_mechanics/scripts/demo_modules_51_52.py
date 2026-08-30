"""Demo runner for Module 51 (Prediction Markets) and Module 52 (Hawkes Process)."""

import numpy as np
import pandas as pd
from quant_mechanics.data.loader import (
    generate_prediction_market_order_books,
    generate_macro_news_and_trades,
)
from quant_mechanics.prediction_markets.mispricing import PredictionMarketArbitrageEngine
from quant_mechanics.hawkes_diffusion.hawkes_model import MultivariateHawkesEngine


def run_demo():
    print("=" * 80)
    print("===== MODULE 51: PREDICTION MARKET ARBITRAGE & KELLY EXECUTION ENGINE =====")
    print("=" * 80)

    books_dict = generate_prediction_market_order_books(seed=42)
    engine_arb = PredictionMarketArbitrageEngine(default_fractional_kelly=0.50)

    for contract_id, venues in books_dict.items():
        poly_book = venues["Polymarket"]
        kalshi_book = venues["Kalshi"]
        print(f"\n[+] Contract: {contract_id} ('{poly_book.event_title}')")
        print(f"    Polymarket Best Ask: Yes @ ${poly_book.yes_asks[0].price:.2f} | No @ ${poly_book.no_asks[0].price:.2f}")
        print(f"    Kalshi     Best Ask: Yes @ ${kalshi_book.yes_asks[0].price:.2f} | No @ ${kalshi_book.no_asks[0].price:.2f}")

        # Check Cross-Venue Arbitrage
        cross_arbs = engine_arb.check_cross_venue_arbitrage(poly_book, kalshi_book, target_size=2500.0)
        if cross_arbs:
            for arb in cross_arbs:
                print(f"    >>> ARBITRAGE FOUND ({arb.arb_type.value}):")
                print(f"        Long YES on {arb.leg_yes_venue} (${arb.leg_yes_price:.2f}) + Long NO on {arb.leg_no_venue} (${arb.leg_no_price:.2f})")
                print(f"        Gross Combined Price: ${arb.gross_combined_price:.2f} (Gross Edge: {arb.gross_edge_pct:.1f}%)")
                print(f"        Depth-Weighted Exec Price: ${arb.effective_combined_price:.4f}")
                print(f"        Max Executable Size: {arb.max_executable_size:,.0f} contracts")
                print(f"        Net Profit: ${arb.net_profit_usd:+,.2f} | Net ROI: {arb.net_roi_pct:+.2f}%")
                
                # Kelly Sizing
                kelly = engine_arb.calculate_kelly_fraction(
                    true_win_prob=0.999,  # Arbitrage guaranteed payout
                    market_price=arb.effective_combined_price,
                    bankroll_usd=100_000.0,
                )
                print(f"        Kelly Recommendation: {kelly.recommended_fraction:.1%} of bankroll (${kelly.recommended_stake_usd:,.2f})")
                
                # Latency Decay
                edge_0 = arb.gross_edge_pct
                edge_1s = engine_arb.calculate_latency_alpha_decay(edge_0, elapsed_sec=1.0, half_life_sec=2.0)
                edge_3s = engine_arb.calculate_latency_alpha_decay(edge_0, elapsed_sec=3.0, half_life_sec=2.0)
                print(f"        Latency Alpha Decay: 0s: {edge_0:.1f}% -> 1s: {edge_1s:.1f}% -> 3s: {edge_3s:.1f}%")
        else:
            print("    No cross-venue arbitrage detected (Markets Efficient).")

    print("\n" + "=" * 80)
    print("===== MODULE 52: INFORMATION DIFFUSION & MULTIVARIATE HAWKES PROCESS =====")
    print("=" * 80)

    print("[+] Simulating high-frequency trade prints and scheduled macro announcements...")
    data_hawkes = generate_macro_news_and_trades(n_days=3, seed=42)
    news_times = data_hawkes["macro_news_times"]
    trade_times = data_hawkes["price_jump_times"]
    horizon = data_hawkes["horizon"]

    print(f"    Macro News Events: {len(news_times)} | Price Jump Events: {len(trade_times)}")
    print(f"    Observation Horizon: {horizon:,.0f} seconds ({horizon/86400:.1f} days)")

    engine_hawkes = MultivariateHawkesEngine(dimension=2)
    print("\n[+] Fitting 2-Variate Mutually Exciting Hawkes Point Process via MLE...")
    fit_res = engine_hawkes.fit([news_times, trade_times], horizon=horizon, max_iter=100)

    print(f"    Fitted Baseline Intensities (mu):")
    print(f"      - Macro News (mu_0):       {fit_res.mu[0]:.6f} events/sec")
    print(f"      - Price Jumps (mu_1):      {fit_res.mu[1]:.6f} events/sec")
    
    print(f"    Fitted Excitation Matrix (alpha):")
    print(f"      - News -> News (alpha_00):     {fit_res.alpha[0, 0]:.4f}")
    print(f"      - News -> Jumps (alpha_10):    {fit_res.alpha[1, 0]:.4f}")
    print(f"      - Jumps -> News (alpha_01):    {fit_res.alpha[0, 1]:.4f}")
    print(f"      - Jumps -> Jumps (alpha_11):   {fit_res.alpha[1, 1]:.4f}")

    print(f"    Fitted Decay Rates (beta):")
    print(f"      - News -> Jumps (beta_10):     {fit_res.beta[1, 0]:.4f} (Half-life: {fit_res.half_lives[1, 0]:.2f} sec)")
    print(f"      - Jumps -> Jumps (beta_11):    {fit_res.beta[1, 1]:.4f} (Half-life: {fit_res.half_lives[1, 1]:.2f} sec)")

    print(f"    Branching Matrix (A = alpha / beta):")
    print(f"      [[{fit_res.branching_matrix[0,0]:.4f}, {fit_res.branching_matrix[0,1]:.4f}],")
    print(f"       [{fit_res.branching_matrix[1,0]:.4f}, {fit_res.branching_matrix[1,1]:.4f}]]")

    print(f"\n    Econometric Diagnostics:")
    print(f"      - Spectral Radius:             {fit_res.spectral_radius:.4f} (< 1.0 -> Subcritical / Stable)")
    print(f"      - Endogeneity Ratio:           {fit_res.endogeneity_ratio:.2%}")
    print(f"      - Information Absorption t_1/2:{engine_hawkes.information_absorption_half_life(0, 1):.2f} seconds")
    print(f"      - Log-Likelihood:              {fit_res.log_likelihood:,.2f}")
    print(f"      - AIC / BIC:                   {fit_res.aic:,.2f} / {fit_res.bic:,.2f}")

    print("\n[+] Evaluating Papangelou Residual Diagnostics (Exp(1) Kolmogorov-Smirnov Test)...")
    diag = engine_hawkes.evaluate_residual_diagnostics([news_times, trade_times])
    print(f"    - Macro News KS Stat: {diag.ks_statistics[0]:.4f} (p-value: {diag.ks_p_values[0]:.4f}, Valid: {diag.is_valid_exp1[0]})")
    print(f"    - Price Jumps KS Stat:{diag.ks_statistics[1]:.4f} (p-value: {diag.ks_p_values[1]:.4f}, Valid: {diag.is_valid_exp1[1]})")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
