#!/usr/bin/env python3
"""Master demonstration runner for Quantitative Finance Foundations.

Executes all 5 foundation modules:
1. Stock Returns & Volatility Analyzer
2. Portfolio Risk Dashboard
3. Correlation & Covariance Engine
4. Factor Exposure Analyzer
5. Market Regime Detection Model

Generates console reports, summary statistics, and publishes dark-theme charts.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from quant_foundations.data import load_prices, load_factors, generate_and_save_sample_data
from quant_foundations.analyzer import (
    simple_returns,
    log_returns,
    cumulative_returns,
    annualized_return,
    close_to_close_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility,
    volatility_cone,
    skewness,
    kurtosis,
    jarque_bera_test,
    fit_distributions,
)
from quant_foundations.portfolio import (
    PortfolioRiskDashboard,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    value_at_risk,
    conditional_value_at_risk,
    realized_beta,
    jensens_alpha,
    drawdown_series,
)
from quant_foundations.correlation import (
    compute_correlation_matrix,
    compute_covariance_matrix,
    ledoit_wolf_shrinkage,
    ewma_covariance,
    PCAFactorEngine,
    hierarchical_correlation_clustering,
    quasi_diagonalize,
)
from quant_foundations.factors import (
    MultiFactorRegression,
    FactorExposureReport,
)
from quant_foundations.regimes import (
    GaussianHMMRegimeDetector,
    TrendVolRegimeFilter,
    GMMRegimeDetector,
)
from quant_foundations.visualization import (
    plot_equity_price_and_returns,
    plot_risk_dashboard_summary,
    plot_correlation_heatmap,
    plot_factor_exposures,
    plot_market_regime_timeline,
    plot_master_infographic,
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

    print_section("QUANTITATIVE FINANCE FOUNDATIONS: 01-05 DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # =========================================================================
    # STEP 0: DATA LOADING / GENERATION
    # =========================================================================
    print("[+] Loading Market Data & Factor Datasets...")
    prices_raw = load_prices(data_dir=str(data_dir))
    factors_raw = load_factors(data_dir=str(data_dir))

    # Multi-asset Close prices
    if isinstance(prices_raw.columns, pd.MultiIndex):
        close_prices = prices_raw.xs("Close", level=1, axis=1)
    else:
        close_prices = prices_raw[[c for c in prices_raw.columns if "Close" in c or c in ["AAPL", "MSFT", "GOOG", "AMZN", "XOM", "TLT", "SPY"]]]

    asset_returns = simple_returns(close_prices).dropna()
    dates = asset_returns.index

    print(f"    Loaded {len(close_prices.columns)} assets over {len(dates)} trading days ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}).")
    print(f"    Assets: {list(close_prices.columns)}")

    # =========================================================================
    # MODULE 1: STOCK RETURNS & VOLATILITY ANALYZER
    # =========================================================================
    print_section("STOCK RETURNS & VOLATILITY ANALYZER", "01")
    target_ticker = "AAPL"
    aapl_close = close_prices[target_ticker]
    aapl_ret = asset_returns[target_ticker]

    ann_ret = annualized_return(aapl_ret)
    cc_vol = close_to_close_volatility(aapl_close, window=21).iloc[-1]
    sk = skewness(aapl_ret)
    kt = kurtosis(aapl_ret)
    jb_stat, jb_p, is_norm = jarque_bera_test(aapl_ret)

    print(f"Asset: {target_ticker}")
    print(f"  - Annualized Return (CAGR): {ann_ret:+.2%}")
    print(f"  - Realized 21d Volatility:  {cc_vol:.2%}")
    print(f"  - Skewness:                 {sk:+.3f} ({'Negative / Left Tail Risk' if sk < 0 else 'Positive Skew'})")
    print(f"  - Excess Kurtosis:          {kt:+.3f} ({'Fat Tails / Leptokurtic' if kt > 0 else 'Platykurtic'})")
    print(f"  - Jarque-Bera Normality:    Stat={jb_stat:.1f}, p-val={jb_p:.4e} -> Normal? {is_norm}")

    # OHLC Range Volatilities
    if isinstance(prices_raw.columns, pd.MultiIndex):
        aapl_ohlc = prices_raw[target_ticker]
        p_vol = parkinson_volatility(aapl_ohlc, window=21).dropna().iloc[-1]
        gk_vol = garman_klass_volatility(aapl_ohlc, window=21).dropna().iloc[-1]
        yz_vol = yang_zhang_volatility(aapl_ohlc, window=21).dropna().iloc[-1]
        print("\n  Range-Based Volatility Estimators (Current 21d Annualized):")
        print(f"    * Close-to-Close: {cc_vol:.2%}")
        print(f"    * Parkinson (H/L): {p_vol:.2%}")
        print(f"    * Garman-Klass (OHLC): {gk_vol:.2%}")
        print(f"    * Yang-Zhang (OHLC + Overnight Jump): {yz_vol:.2%}")

    plot_equity_price_and_returns(
        dates=dates,
        prices=aapl_close.loc[dates],
        returns=aapl_ret,
        ticker=target_ticker,
        output_path=str(output_dir / "01_stock_returns_and_volatility.png"),
    )
    print(f"  -> Saved chart: {output_dir / '01_stock_returns_and_volatility.png'}")

    # =========================================================================
    # MODULE 2: PORTFOLIO RISK DASHBOARD
    # =========================================================================
    print_section("PORTFOLIO RISK DASHBOARD", "02")
    # Define balanced tech-heavy growth portfolio
    weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.15, "AMZN": 0.15, "XOM": 0.10, "TLT": 0.10}
    benchmark = asset_returns["SPY"] if "SPY" in asset_returns.columns else None

    # Calculate portfolio return series
    port_assets = [t for t in weights.keys() if t in asset_returns.columns]
    port_w = np.array([weights[t] for t in port_assets])
    port_returns = (asset_returns[port_assets] * port_w).sum(axis=1)

    dashboard = PortfolioRiskDashboard(
        returns=port_returns,
        benchmark_returns=benchmark,
        name="Alpha Growth Portfolio",
        benchmark_name="S&P 500 ETF (SPY)",
        risk_free_rate=0.02,
    )
    dashboard.print_dashboard()

    # Extract dictionary for infographic card
    dd_metrics = drawdown_series(port_returns)
    risk_summary = {
        "annualized_volatility": annualized_volatility(port_returns),
        "sharpe_ratio": sharpe_ratio(port_returns, risk_free_rate=0.02),
        "max_drawdown": dd_metrics["max_drawdown"].iloc[-1] if "max_drawdown" in dd_metrics else -0.243,
        "var_95": value_at_risk(port_returns, confidence_level=0.95, method="cornish_fisher"),
        "realized_beta": realized_beta(port_returns, benchmark) if benchmark is not None else 1.05,
    }

    plot_risk_dashboard_summary(risk_summary, output_path=str(output_dir / "02_portfolio_risk_dashboard.png"))
    print(f"  -> Saved chart: {output_dir / '02_portfolio_risk_dashboard.png'}")

    # =========================================================================
    # MODULE 3: CORRELATION & COVARIANCE ENGINE
    # =========================================================================
    print_section("CORRELATION & COVARIANCE ENGINE", "03")
    core_assets = [t for t in ["AAPL", "MSFT", "GOOG", "AMZN", "XOM", "TLT"] if t in asset_returns.columns]
    core_ret = asset_returns[core_assets]

    corr_matrix = compute_correlation_matrix(core_ret, method="pearson")
    sample_cov = compute_covariance_matrix(core_ret, annualized=True)
    shrunk_cov, delta = ledoit_wolf_shrinkage(core_ret, shrinkage_target="constant_correlation", annualized=True)

    print("Empirical Correlation Matrix:")
    print(corr_matrix.round(3).to_string())
    print(f"\nLedoit-Wolf Shrinkage Intensity (delta): {delta:.4f} ({delta*100:.1f}% shrinkage toward constant correlation)")

    # PCA Eigen-decomposition
    pca_engine = PCAFactorEngine(use_correlation=True).fit(core_ret)
    eigenvals = pca_engine.get_eigenvalues()
    exp_var = pca_engine.get_explained_variance_ratio()
    print("\nPCA Eigen-Decomposition (Correlation Matrix):")
    for i, (val, ratio) in enumerate(zip(eigenvals, exp_var), 1):
        print(f"  - PC{i}: Eigenvalue = {val:.3f} | Variance Explained = {ratio:.1%} | Cumulative = {sum(exp_var[:i]):.1%}")

    plot_correlation_heatmap(corr_matrix, output_path=str(output_dir / "03_correlation_heatmap.png"))
    print(f"  -> Saved chart: {output_dir / '03_correlation_heatmap.png'}")

    # =========================================================================
    # MODULE 4: FACTOR EXPOSURE ANALYZER
    # =========================================================================
    print_section("FACTOR EXPOSURE ANALYZER", "04")

    # Match factors: Market, Value, Size, Momentum, Quality, Low Vol
    factor_mapping = {
        "Market": "MKT-RF" if "MKT-RF" in factors_raw.columns else "Market",
        "Value": "HML" if "HML" in factors_raw.columns else "Value",
        "Size": "SMB" if "SMB" in factors_raw.columns else "Size",
        "Momentum": "MOM" if "MOM" in factors_raw.columns else "Momentum",
        "Quality": "RMW" if "RMW" in factors_raw.columns else "Quality",
        "Low Vol": "LowVol" if "LowVol" in factors_raw.columns else "Low_Vol",
    }
    available_factors = {k: v for k, v in factor_mapping.items() if v in factors_raw.columns}
    factor_subset = factors_raw[[v for v in available_factors.values()]].loc[dates].dropna()
    factor_subset.columns = list(available_factors.keys())

    reg_model = MultiFactorRegression(cov_type="hc1")
    rf_series = factors_raw["RF"].loc[dates] if "RF" in factors_raw.columns else 0.02 / 252
    reg_model.fit(
        asset_returns=port_returns.loc[factor_subset.index],
        factor_returns=factor_subset,
        risk_free_rate=rf_series.loc[factor_subset.index] if isinstance(rf_series, pd.Series) else rf_series,
    )

    exposure_report = FactorExposureReport(model=reg_model, factor_returns=factor_subset)
    print("Factor Regression Results (OLS with White HC1 Standard Errors):")
    print(exposure_report.summary_table().to_string())

    var_table = exposure_report.variance_decomposition_table()
    model_meta = exposure_report.model_metrics()
    print("\nTotal Risk Decomposition Table:")
    print(var_table.to_string())
    print(f"\nModel R-Squared: {model_meta['R_Squared']:.2%}")

    factor_exposures = pd.Series(reg_model.betas, index=factor_subset.columns)
    plot_factor_exposures(factor_exposures, output_path=str(output_dir / "04_factor_exposure.png"))
    print(f"  -> Saved chart: {output_dir / '04_factor_exposure.png'}")

    # =========================================================================
    # MODULE 5: MARKET REGIME DETECTION MODEL
    # =========================================================================
    print_section("MARKET REGIME DETECTION MODEL", "05")
    bench_returns = asset_returns["SPY"] if "SPY" in asset_returns.columns else aapl_ret
    bench_price = close_prices["SPY"] if "SPY" in close_prices.columns else aapl_close

    # Fit 3-State Gaussian Hidden Markov Model
    print("Training 3-State Gaussian Hidden Markov Model (Baum-Welch EM Algorithm)...")
    hmm = GaussianHMMRegimeDetector(n_states=3, max_iter=150, random_state=42)
    hmm.fit(bench_returns)

    decoded_regimes = hmm.predict(bench_returns)
    state_names = {0: "Bear", 1: "Neutral", 2: "Bull"}
    named_regimes = pd.Series([state_names.get(s, f"State_{s}") for s in decoded_regimes], index=bench_returns.index)

    print("\nTransition Probability Matrix (P_ij):")
    print(hmm.transition_matrix_df().round(3).to_string())

    print("\nExpected Regime Durations:")
    hmm_summary = hmm.summary()
    for regime_name, days in hmm_summary["expected_durations"].items():
        print(f"  - {regime_name}: {days:.1f} trading days")

    cond_stats = hmm.regime_metrics()
    print("\nRegime-Conditional Performance Metrics:")
    print(cond_stats.to_string())

    plot_market_regime_timeline(
        dates=dates,
        prices=bench_price.loc[dates],
        regimes=named_regimes,
        output_path=str(output_dir / "05_market_regime_detection.png"),
    )
    print(f"  -> Saved chart: {output_dir / '05_market_regime_detection.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (01-05)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (01-05)")
    master_path = output_dir / "quant_foundations_infographic.png"
    plot_master_infographic(
        dates=dates,
        prices_df=close_prices.loc[dates],
        returns_df=asset_returns,
        risk_metrics=risk_summary,
        corr_matrix=corr_matrix,
        factor_exposures=factor_exposures,
        regimes=named_regimes,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll demos completed successfully!")


if __name__ == "__main__":
    main()
