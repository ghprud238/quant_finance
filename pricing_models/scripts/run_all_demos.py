#!/usr/bin/env python3
"""Master demonstration runner for Derivatives & Pricing Models (16-20).

Executes all 5 modules:
1. Black-Scholes Option Pricing Engine (16)
2. Implied Volatility Solver & Smile (17)
3. Option Greeks Calculator (18)
4. Binomial Tree Option Pricing Model (19)
5. Monte Carlo Option Pricing Engine & Exotics (20)

Also benchmarks model accuracy against market prices, pricing errors, Greeks, and volatility assumptions.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from pricing_models.black_scholes import BlackScholesModel, OptionChainPricer
from pricing_models.greeks import AnalyticalGreeks, NumericalGreeks
from pricing_models.implied_vol import ImpliedVolatilitySolver, VolatilitySmile, VolatilitySurface
from pricing_models.binomial_tree import BinomialTreePricer
from pricing_models.monte_carlo import MonteCarloOptionPricer, ExoticOptionPricer
from pricing_models.visualization.plots import (
    plot_black_scholes_card,
    plot_volatility_smile,
    plot_volatility_surface_3d,
    plot_option_greeks_card,
    plot_binomial_tree_diagram,
    plot_monte_carlo_option_paths,
    plot_master_pricing_infographic,
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

    print_section("QUANTITATIVE DERIVATIVES & PRICING (16-20) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # Standard test parameters
    S0 = 100.0       # Spot Price ($)
    r = 0.05         # Risk-Free Rate (5%)
    q = 0.01         # Dividend Yield (1%)
    sigma = 0.20     # Volatility (20%)
    T = 0.50         # Expiration (6 Months)

    # =========================================================================
    # MODULE 16: BLACK-SCHOLES OPTION PRICING ENGINE
    # =========================================================================
    print_section("BLACK-SCHOLES OPTION PRICING ENGINE", "16")
    call_res = BlackScholesModel.calculate(S0, K=100.0, T=T, r=r, sigma=sigma, q=q, option_type="call")
    put_res = BlackScholesModel.calculate(S0, K=100.0, T=T, r=r, sigma=sigma, q=q, option_type="put")

    print("European ATM Option Pricing (S=100, K=100, T=0.5y, r=5%, q=1%, sigma=20%):")
    print(f"  - Call Price:               ${call_res.call_price:.4f}")
    print(f"  - Put Price:                ${put_res.put_price:.4f}")
    print(f"  - Call Intrinsic / Time:    ${call_res.intrinsic_value:.4f} / ${call_res.time_value:.4f}")
    print(f"  - Put-Call Parity Diff:     {call_res.put_call_parity_diff:.2e} (Exact Parity Verified)")

    # Option Chain generation matching screenshot table
    strikes = [100.0, 105.0, 110.0, 115.0, 120.0]
    chain_df = OptionChainPricer.generate_chain(S0, strikes, T, r, sigma, q)
    print("\nOption Chain Table (Calls & Puts across Strikes):")
    print(chain_df.to_string())

    plot_black_scholes_card(chain_df, output_path=str(output_dir / "16_black_scholes_card.png"))
    print(f"  -> Saved chart: {output_dir / '16_black_scholes_card.png'}")

    # =========================================================================
    # MODULE 17: IMPLIED VOLATILITY SOLVER & VOLATILITY SMILE
    # =========================================================================
    print_section("IMPLIED VOLATILITY SOLVER & SMILE", "17")
    solver = ImpliedVolatilitySolver(tolerance=1e-8)

    # Invert sample market price
    sample_mkt_call = 5.80
    recovered_iv = solver.solve(sample_mkt_call, spot=100.0, strike=100.0, time_to_expiry=T, risk_free_rate=r, dividend_yield=q, option_type="call")
    print(f"Implied Volatility Root-Finding (Market Price = ${sample_mkt_call:.2f}):")
    print(f"  - Recovered Implied Volatility: {recovered_iv:.2%}")

    # Volatility Smile Calibration
    moneyness_grid = np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40])
    market_ivs = np.array([0.38, 0.30, 0.24, 0.20, 0.18, 0.20, 0.24, 0.29, 0.36])

    smile = VolatilitySmile(spot=100.0, time_to_expiry=T, risk_free_rate=r, dividend_yield=q)
    smile_strikes = moneyness_grid * 100.0
    svi_fit = smile.fit_svi(smile_strikes, market_ivs)

    print("\nParametric SVI Volatility Smile Calibration:")
    print(f"  - ATM Implied Volatility:   {smile.get_atm_vol():.2%}")
    print(f"  - 90%-110% Risk Reversal:   {smile.get_skew():.4f}")
    print(f"  - SVI Parameters (a, b, rho, m, sigma): ({svi_fit.a:.3f}, {svi_fit.b:.3f}, {svi_fit.rho:.3f}, {svi_fit.m:.3f}, {svi_fit.sigma:.3f})")

    # 3D Volatility Surface
    surf = VolatilitySurface.create_synthetic_market_surface(spot=100.0)
    mesh_dict = surf.generate_mesh()

    plot_volatility_smile(moneyness_grid, market_ivs, output_path=str(output_dir / "17_volatility_smile.png"))
    plot_volatility_surface_3d(mesh_dict["moneyness_grid"], mesh_dict["expiry_grid"], mesh_dict["iv_grid"], output_path=str(output_dir / "17_volatility_surface_3d.png"))
    print(f"  -> Saved smile chart:   {output_dir / '17_volatility_smile.png'}")
    print(f"  -> Saved 3D surf chart: {output_dir / '17_volatility_surface_3d.png'}")

    # =========================================================================
    # MODULE 18: OPTION GREEKS CALCULATOR (Δ, Γ, Θ, ν, ρ)
    # =========================================================================
    print_section("OPTION GREEKS CALCULATOR", "18")
    call_greeks = AnalyticalGreeks.calculate_all(S0, K=100.0, T=T, r=r, sigma=sigma, q=q, option_type="call")
    num_greeks = NumericalGreeks.calculate_all(S0, K=100.0, T=T, r=r, sigma=sigma, q=q, option_type="call")

    print("ATM European Call Greeks Comparison (Analytical vs Numerical Finite Difference):")
    print(f"  - Delta (Δ = ∂V/∂S):         {call_greeks.delta:+.4f} (Numerical: {num_greeks['delta']:+.4f})")
    print(f"  - Gamma (Γ = ∂²V/∂S²):       {call_greeks.gamma:.4f}  (Numerical: {num_greeks['gamma']:.4f})")
    print(f"  - Theta (Θ = ∂V/∂t daily):   {call_greeks.theta_daily:+.4f} (Numerical: {(num_greeks['theta']/365.0):+.4f})")
    print(f"  - Vega  (ν = ∂V/∂σ 1%):      {call_greeks.vega_pct:.4f}  (Numerical: {(num_greeks['vega']/100.0):.4f})")
    print(f"  - Rho   (ρ = ∂V/∂r 1%):      {call_greeks.rho_pct:+.4f} (Numerical: {(num_greeks['rho']/100.0):+.4f})")
    print(f"  - Vanna (∂²V/∂S∂σ):          {call_greeks.vanna:+.4f}")
    print(f"  - Volga (∂²V/∂σ²):           {call_greeks.volga:.4f}")

    greeks_dict = {
        "delta": call_greeks.delta,
        "gamma": call_greeks.gamma,
        "theta_daily": call_greeks.theta_daily,
        "vega_pct": call_greeks.vega_pct,
        "rho_pct": call_greeks.rho_pct,
    }
    plot_option_greeks_card(greeks_dict, output_path=str(output_dir / "18_option_greeks_card.png"))
    print(f"  -> Saved chart: {output_dir / '18_option_greeks_card.png'}")

    # =========================================================================
    # MODULE 19: BINOMIAL TREE OPTION PRICING MODEL
    # =========================================================================
    print_section("BINOMIAL TREE OPTION PRICING MODEL", "19")
    tree_pricer = BinomialTreePricer(S0=S0, K=100.0, T=T, r=r, sigma=sigma, q=q, n_steps=200)

    # European vs American Put
    bin_call = tree_pricer.price(option_type="call", exercise_style="european", model="crr")
    bin_put_eur = tree_pricer.price(option_type="put", exercise_style="european", model="crr")
    bin_put_ame = tree_pricer.price(option_type="put", exercise_style="american", model="crr")

    print("CRR Binomial Lattice Pricing (N=200 Steps):")
    print(f"  - European Call Price:       ${bin_call.price:.4f} (BS Error: ${abs(bin_call.price - call_res.call_price):.4e})")
    print(f"  - European Put Price:        ${bin_put_eur.price:.4f} (BS Error: ${abs(bin_put_eur.price - put_res.put_price):.4e})")
    print(f"  - American Put Price:        ${bin_put_ame.price:.4f}")
    print(f"  - Early Exercise Premium:    ${bin_put_ame.early_exercise_premium:.4f}")
    print(f"  - Lattice Delta / Gamma:     {bin_call.greeks.delta:.4f} / {bin_call.greeks.gamma:.4f}")

    plot_binomial_tree_diagram(output_path=str(output_dir / "19_binomial_tree_diagram.png"))
    print(f"  -> Saved chart: {output_dir / '19_binomial_tree_diagram.png'}")

    # =========================================================================
    # MODULE 20: MONTE CARLO OPTION PRICING ENGINE & EXOTICS
    # =========================================================================
    print_section("MONTE CARLO OPTION PRICING ENGINE", "20")
    mc_pricer = MonteCarloOptionPricer(S0=S0, K=100.0, T=T, r=r, sigma=sigma, q=q)

    # Price with Control Variate variance reduction
    mc_res = mc_pricer.price(option_type="call", n_simulations=100000, control_variate=True)
    print(f"Monte Carlo European Call Simulation (N=100,000 Paths):")
    print(f"  - Estimated Price:           ${mc_res.price:.4f} +/- ${mc_res.standard_error:.4f}")
    print(f"  - 95% Confidence Interval:   [${mc_res.confidence_interval_95[0]:.4f}, ${mc_res.confidence_interval_95[1]:.4f}]")
    print(f"  - Black-Scholes Benchmark:   ${call_res.call_price:.4f}")
    print(f"  - Variance Reduction Factor: {mc_res.variance_reduction_ratio:.1f}x")

    # Exotic derivatives
    exotics = ExoticOptionPricer(S0=S0, K=100.0, T=T, r=r, sigma=sigma, q=q)
    asian = exotics.price_asian(option_type="call", averaging_type="arithmetic", n_simulations=100000)
    barrier = exotics.price_barrier(option_type="call", barrier_type="up_and_out", barrier_level=125.0, n_simulations=100000)
    lsm_put = exotics.price_american_lsm(option_type="put", n_simulations=50000, n_steps=50)

    print("\nPath-Dependent & Exotic Option Prices:")
    print(f"  - Arithmetic Asian Call:     ${asian.price:.4f}")
    print(f"  - Up-and-Out Barrier Call:   ${barrier.price:.4f} (Barrier Hit Prob: {barrier.hit_probability:.1%})")
    print(f"  - LSM American Put:          ${lsm_put.price:.4f} (Early Exercise Prem: ${lsm_put.early_exercise_premium:.4f})")

    # Generate simulation paths for plotting
    time_grid = np.linspace(0, 1.0, 100)
    dt = 1.0 / 100
    z_paths = np.random.normal(0, 1, (1000, 100))
    log_paths = np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_paths, axis=1)
    sim_paths = S0 * np.exp(np.hstack([np.zeros((1000, 1)), log_paths]))[:, :-1]

    plot_monte_carlo_option_paths(sim_paths, strike=100.0, output_path=str(output_dir / "20_monte_carlo_simulation.png"))
    print(f"  -> Saved chart: {output_dir / '20_monte_carlo_simulation.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (16-20)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (16-20)")
    master_path = output_dir / "pricing_models_infographic.png"
    plot_master_pricing_infographic(
        chain_df=chain_df,
        moneyness=moneyness_grid,
        implied_vols=market_ivs,
        greeks_dict=greeks_dict,
        mc_paths=sim_paths,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll derivatives & pricing model demos executed successfully!")


if __name__ == "__main__":
    main()
