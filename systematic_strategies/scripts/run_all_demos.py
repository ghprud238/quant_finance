#!/usr/bin/env python3
"""Master demonstration runner for Systematic Trading Strategies (11-15).

Executes all 5 modules:
1. Moving Average Mean-Reversion Strategy (11)
2. Momentum Trading Strategy (12)
3. Pairs Trading / Statistical Arbitrage (13)
4. Factor-Based Long/Short Strategy (14)
5. Multi-Asset Trend-Following System (15)

Also demonstrates the systematic workflow:
SIGNAL -> POSITION SIZING -> TRANSACTION COSTS -> BACKTEST -> RISK METRICS -> OUT-OF-SAMPLE VALIDATION
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from systematic_strategies.data.loader import load_equities, load_pairs, load_macro, load_cross_sectional
from systematic_strategies.engine import (
    BacktestEngine,
    TransactionCostModel,
    PositionSizer,
    WalkForwardValidator,
)
from systematic_strategies.strategies.mean_reversion import MovingAverageMeanReversionStrategy
from systematic_strategies.strategies.momentum import MomentumTradingStrategy
from systematic_strategies.strategies.pairs_trading import PairsTradingStrategy
from systematic_strategies.strategies.factor_long_short import FactorLongShortStrategy
from systematic_strategies.strategies.multi_asset_trend import MultiAssetTrendStrategy
from systematic_strategies.visualization import (
    plot_mean_reversion,
    plot_momentum,
    plot_pairs_spread,
    plot_factor_exposure_heatmap,
    plot_multi_asset_trend,
    plot_master_systematic_infographic,
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

    print_section("QUANTITATIVE SYSTEMATIC STRATEGIES (11-15) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    cost_model = TransactionCostModel(fee_bps=5.0, half_spread_bps=2.5)
    engine = BacktestEngine(cost_model=cost_model, risk_free_rate=0.02)

    # =========================================================================
    # MODULE 11: MOVING AVERAGE MEAN-REVERSION STRATEGY
    # =========================================================================
    print_section("MOVING AVERAGE MEAN-REVERSION STRATEGY", "11")
    equities_df = load_equities(data_dir=str(data_dir))
    aapl_price = equities_df["AAPL"]
    dates = aapl_price.index

    mr_strat = MovingAverageMeanReversionStrategy(lookback_window=20, num_std=2.0, z_entry=1.8, z_exit=0.3)
    mr_res = mr_strat.generate_signals(aapl_price)

    bt_mr = engine.run(aapl_price, mr_res.position, strategy_name="MA Mean Reversion (AAPL)")
    bt_mr.print_summary()

    mr_plot_data = {
        "dates": dates,
        "prices": aapl_price,
        "ma": mr_res.ma,
        "upper_band": mr_res.upper_band,
        "lower_band": mr_res.lower_band,
        "buy_signals": mr_res.entries_long,
        "sell_signals": mr_res.entries_short,
    }
    plot_mean_reversion(
        dates=dates,
        prices=aapl_price,
        ma=mr_res.ma,
        upper_band=mr_res.upper_band,
        lower_band=mr_res.lower_band,
        buy_signals=mr_res.entries_long,
        sell_signals=mr_res.entries_short,
        output_path=str(output_dir / "11_mean_reversion_strategy.png"),
    )
    print(f"  -> Saved chart: {output_dir / '11_mean_reversion_strategy.png'}")

    # =========================================================================
    # MODULE 12: MOMENTUM TRADING STRATEGY
    # =========================================================================
    print_section("MOMENTUM TRADING STRATEGY", "12")
    qqq_price = equities_df["QQQ"]

    mom_strat = MomentumTradingStrategy(mode="composite", fast_window=20, slow_window=50, tsmom_lookback=252)
    mom_res = mom_strat.generate_signals(qqq_price)

    # Apply volatility-targeted sizing
    sized_mom_weights = PositionSizer.volatility_targeting(
        signals=mom_res.position,
        returns=qqq_price.pct_change(),
        target_vol=0.15,
        max_leverage=1.5,
    )
    bt_mom = engine.run(qqq_price, sized_mom_weights, strategy_name="Momentum Vol-Targeted (QQQ)")
    bt_mom.print_summary()

    mom_plot_data = {
        "dates": dates,
        "prices": qqq_price,
        "trend_ma": mom_res.slow_ma,
        "entries": mom_res.entries_long,
        "exits": mom_res.entries_short | (mom_res.position == 0),
    }
    plot_momentum(
        dates=dates,
        prices=qqq_price,
        trend_ma=mom_res.slow_ma,
        entries=mom_res.entries_long,
        exits=mom_res.entries_short,
        output_path=str(output_dir / "12_momentum_strategy.png"),
    )
    print(f"  -> Saved chart: {output_dir / '12_momentum_strategy.png'}")

    # =========================================================================
    # MODULE 13: PAIRS TRADING / STATISTICAL ARBITRAGE
    # =========================================================================
    print_section("PAIRS TRADING / STATISTICAL ARBITRAGE", "13")
    pairs_df = load_pairs(data_dir=str(data_dir))
    ko_price = pairs_df["KO"]
    pep_price = pairs_df["PEP"]

    pairs_strat = PairsTradingStrategy(lookback_window=60, hedge_method="ols", z_entry=2.0, z_exit=0.5)
    pairs_res = pairs_strat.generate_signals(ko_price, pep_price)

    print("Engle-Granger Cointegration & Ornstein-Uhlenbeck Analysis (KO vs PEP):")
    print(f"  - ADF Test Statistic: {pairs_res.coint_test.adf_statistic:.3f} (p-value: {pairs_res.coint_test.p_value:.4f})")
    print(f"  - Cointegrated?       {pairs_res.coint_test.is_cointegrated}")
    print(f"  - Mean Hedge Ratio:   {pairs_res.coint_test.hedge_ratio_static:.3f}")
    print(f"  - OU Half-Life:       {pairs_res.ou_params.half_life_days:.1f} trading days")
    print(f"  - Mean Reversion Speed theta: {pairs_res.ou_params.reversion_speed_theta:.4f}")

    # Backtest pairs portfolio
    pairs_asset_prices = pd.DataFrame({"KO": ko_price, "PEP": pep_price})
    pairs_dollar_weights = pd.DataFrame({"KO": pairs_res.dollar_weight1, "PEP": pairs_res.dollar_weight2})
    bt_pairs = engine.run(pairs_asset_prices, pairs_dollar_weights, strategy_name="Pairs Stat Arb (KO/PEP)")
    bt_pairs.print_summary()

    pairs_plot_data = {
        "dates": dates,
        "spread": pairs_res.spread,
        "z_score": pairs_res.z_score,
        "long_signals": pairs_res.entries_long_spread,
        "short_signals": pairs_res.entries_short_spread,
    }
    plot_pairs_spread(
        dates=dates,
        spread=pairs_res.spread,
        z_score=pairs_res.z_score,
        long_spread_signals=pairs_res.entries_long_spread,
        short_spread_signals=pairs_res.entries_short_spread,
        output_path=str(output_dir / "13_pairs_trading_spread.png"),
    )
    print(f"  -> Saved chart: {output_dir / '13_pairs_trading_spread.png'}")

    # =========================================================================
    # MODULE 14: FACTOR-BASED LONG/SHORT STRATEGY
    # =========================================================================
    print_section("FACTOR-BASED LONG/SHORT STRATEGY", "14")
    cs_prices, cs_factors = load_cross_sectional(data_dir=str(data_dir))
    factor_dict = {col.lower(): cs_factors[col].unstack(level="Ticker") for col in cs_factors.columns}

    factor_strat = FactorLongShortStrategy(
        n_quantiles=5,
        dollar_neutral=True,
        rebalance_freq=21,
        turnover_smoothing=0.8,
        transaction_cost_bps=5.0,
    )
    factor_res = factor_strat.backtest(prices=cs_prices, factor_data=factor_dict)
    print("Cross-Sectional Factor Long/Short Performance (Top vs Bottom Quintile):")
    print(factor_res.summary_table().to_string())

    # Snapshot of factor exposures for heatmap
    latest_factors = cs_factors.xs(cs_factors.index.levels[0][-1], level="Date").T
    plot_factor_exposure_heatmap(
        factor_matrix=latest_factors,
        output_path=str(output_dir / "14_factor_exposure_heatmap.png"),
    )
    print(f"  -> Saved chart: {output_dir / '14_factor_exposure_heatmap.png'}")

    # =========================================================================
    # MODULE 15: MULTI-ASSET TREND-FOLLOWING SYSTEM
    # =========================================================================
    print_section("MULTI-ASSET TREND-FOLLOWING SYSTEM", "15")
    macro_df = load_macro(data_dir=str(data_dir))

    trend_strat = MultiAssetTrendStrategy(
        lookback_horizons=[21, 63, 126, 252],
        target_asset_vol=0.10,
        target_portfolio_vol=0.10,
        use_risk_parity=True,
        rebalance_freq=5,
    )
    trend_res = trend_strat.backtest(macro_df)
    print("Multi-Asset Trend-Following & Risk Parity Summary:")
    print(trend_res.summary_table().to_string())

    macro_ret = trend_res.asset_class_returns.fillna(0.0)
    macro_curves = {
        "dates": macro_ret.index,
        "equities": (1.0 + macro_ret["Equities"]).cumprod(),
        "bonds": (1.0 + macro_ret["Bonds"]).cumprod(),
        "fx": (1.0 + macro_ret["Currencies"]).cumprod(),
        "commodities": (1.0 + macro_ret["Commodities"]).cumprod(),
    }
    plot_multi_asset_trend(
        dates=macro_curves["dates"],
        equity_curve_eq=macro_curves["equities"],
        equity_curve_bonds=macro_curves["bonds"],
        equity_curve_fx=macro_curves["fx"],
        equity_curve_comm=macro_curves["commodities"],
        output_path=str(output_dir / "15_multi_asset_trend.png"),
    )
    print(f"  -> Saved chart: {output_dir / '15_multi_asset_trend.png'}")

    # =========================================================================
    # WORKFLOW VALIDATION: IN-SAMPLE VS OUT-OF-SAMPLE TESTING
    # =========================================================================
    print_section("WORKFLOW VALIDATION: IS VS OOS OVERFITTING DIAGNOSTICS")
    validator = WalkForwardValidator(cost_model=cost_model)
    is_res, oos_res, degradation = validator.simple_train_test_split(qqq_price, sized_mom_weights, train_ratio=0.70)
    
    print("In-Sample (Train) Performance:")
    print(f"  - CAGR:          {is_res.metrics['cagr']:+.2%}")
    print(f"  - Sharpe Ratio:  {is_res.metrics['sharpe_ratio']:.2f}")
    print(f"  - Max Drawdown:  {is_res.metrics['max_drawdown']:.2%}")

    print("\nOut-of-Sample (Test) Performance:")
    print(f"  - CAGR:          {oos_res.metrics['cagr']:+.2%}")
    print(f"  - Sharpe Ratio:  {oos_res.metrics['sharpe_ratio']:.2f}")
    print(f"  - Max Drawdown:  {oos_res.metrics['max_drawdown']:.2%}")

    print(f"\nSharpe Ratio Degradation (IS -> OOS): {degradation:.2%}")
    if degradation < 0.30:
        print("Assessment: ROBUST MODEL (Low Overfitting Risk - OOS Performance Persists)")
    elif degradation < 0.60:
        print("Assessment: MODERATE DEGRADATION (Monitor Turnover & Parameter Sensitivity)")
    else:
        print("Assessment: OVERFIT MODEL (Significant In-Sample Curve-Fitting Detected)")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (11-15)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (11-15)")
    master_path = output_dir / "systematic_strategies_infographic.png"
    plot_master_systematic_infographic(
        mr_data=mr_plot_data,
        mom_data=mom_plot_data,
        pairs_data=pairs_plot_data,
        factor_matrix=latest_factors,
        macro_curves=macro_curves,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll systematic strategy demos executed successfully!")


if __name__ == "__main__":
    main()
