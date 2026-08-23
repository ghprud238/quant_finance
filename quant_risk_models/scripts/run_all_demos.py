#!/usr/bin/env python3
"""Master demonstration runner for Quantitative Risk & Portfolio Projects (06-10).

Executes all 5 modules:
1. Historical VaR Calculator (06)
2. Parametric VaR Model (07)
3. Monte Carlo VaR Engine (08)
4. Expected Shortfall / CVaR Model (09)
5. Portfolio Optimization using Mean-Variance Analysis (10)

Generates console reports, risk dashboards, and dark-theme charts.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from quant_risk_models.data import load_portfolio_data
from quant_risk_models.var.historical import HistoricalVaRCalculator
from quant_risk_models.var.parametric import ParametricVaRModel
from quant_risk_models.var.monte_carlo import MonteCarloVaREngine
from quant_risk_models.cvar.expected_shortfall import ExpectedShortfallModel
from quant_risk_models.portfolio.risk_metrics import PortfolioRiskReport
from quant_risk_models.optimization.mean_variance import MeanVarianceOptimizer
from quant_risk_models.visualization import (
    plot_distribution_and_var,
    plot_risk_metrics_card,
    plot_monte_carlo_simulation,
    plot_efficient_frontier,
    plot_master_risk_infographic,
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

    print_section("QUANTITATIVE RISK & PORTFOLIO MODELS (06-10) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # =========================================================================
    # STEP 0: DATA LOADING
    # =========================================================================
    print("[+] Loading Portfolio and Asset Returns...")
    prices_df, asset_returns, port_returns, weights = load_portfolio_data(data_dir=str(data_dir))
    dates = port_returns.index

    print(f"    Loaded {asset_returns.shape[1]} assets over {len(dates)} trading days ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}).")
    print(f"    Asset Universe: {list(asset_returns.columns)}")
    print(f"    Portfolio Allocation: {weights}")

    # =========================================================================
    # PROJECT 06: HISTORICAL VaR CALCULATOR
    # =========================================================================
    print_section("HISTORICAL VaR CALCULATOR", "06")
    hist_var_calc = HistoricalVaRCalculator(port_returns)
    h_var_95 = hist_var_calc.compute_var(confidence_level=0.95)
    h_var_99 = hist_var_calc.compute_var(confidence_level=0.99)
    h_cvar_95 = hist_var_calc.compute_cvar(confidence_level=0.95)
    h_cvar_99 = hist_var_calc.compute_cvar(confidence_level=0.99)

    pt_est, boot_low_95, boot_high_95 = hist_var_calc.bootstrap_confidence_interval(confidence_level=0.95, n_bootstraps=2000)

    print("Historical Value at Risk (Empirical Percentile):")
    print(f"  - 95% 1-Day VaR:           {-h_var_95:.2%} (95% CI: [{-boot_high_95:.2%}, {-boot_low_95:.2%}])")
    print(f"  - 99% 1-Day VaR:           {-h_var_99:.2%}")
    print(f"  - 95% Expected Shortfall:  {-h_cvar_95:.2%}")
    print(f"  - 99% Expected Shortfall:  {-h_cvar_99:.2%}")

    # =========================================================================
    # PROJECT 07: PARAMETRIC VaR MODEL
    # =========================================================================
    print_section("PARAMETRIC VaR MODEL", "07")
    param_model = ParametricVaRModel(port_returns)
    gauss_var_95 = param_model.gaussian_var(confidence_level=0.95)
    gauss_var_99 = param_model.gaussian_var(confidence_level=0.99)
    t_var_95 = param_model.student_t_var(confidence_level=0.95)
    cf_var_95 = param_model.cornish_fisher_var(confidence_level=0.95)

    print("Parametric Value at Risk Estimates:")
    print(f"  - Delta-Normal (Gaussian) 95% VaR: {-gauss_var_95:.2%}")
    print(f"  - Delta-Normal (Gaussian) 99% VaR: {-gauss_var_99:.2%}")
    print(f"  - Student's t-Distributed 95% VaR: {-t_var_95:.2%}")
    print(f"  - Cornish-Fisher Modified 95% VaR: {-cf_var_95:.2%} (adjusted for Skew & Kurtosis)")

    # =========================================================================
    # PROJECT 08: MONTE CARLO VaR ENGINE
    # =========================================================================
    print_section("MONTE CARLO VaR ENGINE", "08")
    print("Running 100,000 Monte Carlo path simulations (T=252 days)...")
    mc_engine = MonteCarloVaREngine(returns=port_returns)
    mc_paths = mc_engine.simulate_gbm(n_simulations=100000, horizon=252, n_steps=252, initial_value=1.0, random_state=42)

    terminal_returns = mc_paths[:, -1] / 1.0 - 1.0
    mc_var_95 = mc_engine.compute_var(simulated_returns=terminal_returns, confidence_level=0.95)
    mc_var_99 = mc_engine.compute_var(simulated_returns=terminal_returns, confidence_level=0.99)
    mc_cvar_95 = mc_engine.compute_cvar(simulated_returns=terminal_returns, confidence_level=0.95)

    terminal_values = mc_paths[:, -1]
    terminal_var_val = np.percentile(terminal_values, 5.0)

    print(f"Monte Carlo Simulation Results (N=100,000, 1-Year Horizon):")
    print(f"  - Terminal 95% VaR Cutoff (Wealth Value): {terminal_var_val:.3f} (Loss: {1.0 - terminal_var_val:.1%})")
    print(f"  - 1-Year 95% Simulated VaR:              {-mc_var_95:.2%}")
    print(f"  - 1-Year 99% Simulated VaR:              {-mc_var_99:.2%}")
    print(f"  - 1-Year 95% Simulated Expected Shortfall: {-mc_cvar_95:.2%}")

    plot_monte_carlo_simulation(
        paths=mc_paths,
        confidence_level=0.95,
        n_simulations=100000,
        output_path=str(output_dir / "08_monte_carlo_var_simulation.png"),
    )
    print(f"  -> Saved chart: {output_dir / '08_monte_carlo_var_simulation.png'}")

    # =========================================================================
    # PROJECT 09: EXPECTED SHORTFALL / CVaR MODEL & RISK DASHBOARD
    # =========================================================================
    print_section("EXPECTED SHORTFALL / CVaR MODEL", "09")
    es_model = ExpectedShortfallModel(confidence_level=0.95)
    es_hist_95 = es_model.historical_es(port_returns, 0.95)
    es_gauss_95 = es_model.parametric_gaussian_es(port_returns, 0.95)
    es_t_95 = es_model.parametric_student_t_es(port_returns, 0.95)

    print("Expected Shortfall (Conditional Value at Risk) Comparison:")
    print(f"  - Historical 95% Expected Shortfall:   {-es_hist_95:.2%}")
    print(f"  - Parametric Gaussian 95% ES:         {-es_gauss_95:.2%}")
    print(f"  - Parametric Student-t 95% ES:        {-es_t_95:.2%}")

    # Kupiec POF Backtest
    backtest_rep = es_model.backtest_var(port_returns, var_forecasts=h_var_95, confidence_level=0.95)
    print("\nVaR Exception Backtesting Report (Kupiec POF & Christoffersen Independence):")
    print(backtest_rep.summary().to_string())

    # Generate complete Portfolio Risk Report matching screenshot table
    risk_report = PortfolioRiskReport(port_returns, portfolio_name="Example Portfolio", risk_free_rate=0.02)
    risk_report.print_terminal_card()
    raw_dict = risk_report.to_dict()

    metrics_dict = {
        "annualized_return": raw_dict["annualized_return"],
        "annualized_volatility": raw_dict["annualized_volatility"],
        "sharpe_ratio": raw_dict["sharpe_ratio"],
        "sortino_ratio": raw_dict["sortino_ratio"],
        "var_95": -raw_dict["var_95_historical"],
        "var_99": -raw_dict["var_99_historical"],
        "cvar_95": -raw_dict["cvar_95_historical"],
        "max_drawdown": raw_dict["max_drawdown"],
    }

    plot_distribution_and_var(
        returns=port_returns,
        var_95=metrics_dict["var_95"],
        var_99=metrics_dict["var_99"],
        es_95=metrics_dict["cvar_95"],
        output_path=str(output_dir / "06_distribution_and_var.png"),
    )
    plot_risk_metrics_card(metrics_dict, output_path=str(output_dir / "09_expected_shortfall_cvar.png"))
    print(f"  -> Saved distribution chart: {output_dir / '06_distribution_and_var.png'}")
    print(f"  -> Saved risk card:         {output_dir / '09_expected_shortfall_cvar.png'}")

    # =========================================================================
    # PROJECT 10: PORTFOLIO OPTIMIZATION USING MEAN-VARIANCE ANALYSIS
    # =========================================================================
    print_section("PORTFOLIO OPTIMIZATION (MEAN-VARIANCE ANALYSIS)", "10")
    expected_returns = asset_returns.mean() * 252
    cov_matrix = asset_returns.cov() * 252

    optimizer = MeanVarianceOptimizer(expected_returns=expected_returns, cov_matrix=cov_matrix, risk_free_rate=0.02)

    # 1. Minimum Volatility Portfolio
    min_vol_res = optimizer.min_volatility()
    print("Global Minimum Volatility Portfolio:")
    print(f"  - Annualized Return:     {min_vol_res.expected_return:.2%}")
    print(f"  - Annualized Volatility: {min_vol_res.volatility:.2%}")
    print(f"  - Sharpe Ratio:          {min_vol_res.sharpe_ratio:.2f}")

    # 2. Maximum Sharpe Ratio / Tangency Portfolio
    opt_sharpe_res = optimizer.max_sharpe_ratio()
    print("\nOptimal Tangency Portfolio (Maximum Sharpe Ratio):")
    print(f"  - Annualized Return:     {opt_sharpe_res.expected_return:.2%}")
    print(f"  - Annualized Volatility: {opt_sharpe_res.volatility:.2%}")
    print(f"  - Sharpe Ratio:          {opt_sharpe_res.sharpe_ratio:.2f}")
    print("  - Asset Weights:")
    for ticker, w in opt_sharpe_res.weights.items():
        if w > 0.001:
            print(f"    * {ticker}: {w:.1%}")

    # 3. Simulate 5,000 Random Portfolios
    print("\nSimulating 5,000 random portfolios for risk-return cloud...")
    sim_res = optimizer.simulate_random_portfolios(n_portfolios=5000, seed=42)

    # 4. Generate Efficient Frontier Curve (60 points)
    print("Calculating 60 points along the Efficient Frontier...")
    frontier_res = optimizer.efficient_frontier(n_points=60)

    plot_efficient_frontier(
        sim_vols=sim_res.volatilities,
        sim_returns=sim_res.returns,
        frontier_vols=frontier_res.volatilities,
        frontier_returns=frontier_res.returns,
        optimal_vol=opt_sharpe_res.volatility,
        optimal_return=opt_sharpe_res.expected_return,
        min_vol=min_vol_res.volatility,
        min_vol_return=min_vol_res.expected_return,
        output_path=str(output_dir / "10_efficient_frontier_optimization.png"),
    )
    print(f"  -> Saved chart: {output_dir / '10_efficient_frontier_optimization.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (06-10)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (06-10)")
    master_path = output_dir / "quant_risk_models_infographic.png"
    plot_master_risk_infographic(
        returns=port_returns,
        metrics_dict=metrics_dict,
        mc_paths=mc_paths,
        sim_vols=sim_res.volatilities,
        sim_returns=sim_res.returns,
        frontier_vols=frontier_res.volatilities,
        frontier_returns=frontier_res.returns,
        optimal_vol=opt_sharpe_res.volatility,
        optimal_return=opt_sharpe_res.expected_return,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll risk demos executed successfully!")


if __name__ == "__main__":
    main()
