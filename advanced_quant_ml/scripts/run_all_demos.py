#!/usr/bin/env python3
"""Master demonstration runner for Advanced Quant & Machine Learning (21-25).

Executes all 5 modules:
1. Volatility Forecasting with GARCH (21)
2. Yield Curve Modeling & Bootstrapping (22)
3. Kalman Filter for Dynamic Pairs Trading (23)
4. Machine Learning Return Predictor (24)
5. Alternative Data Alpha Model (25)

Generates console reports, econometric summaries, and dark-theme infographic charts.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from advanced_quant_ml.data.loader import (
    load_equity_returns,
    load_yield_curve_data,
    load_alternative_data,
    YIELD_CURVE_MATURITIES,
    YIELD_CURVE_TENORS,
)
from advanced_quant_ml.garch import GARCHModel
from advanced_quant_ml.yield_curve import (
    NelsonSiegelModel,
    NelsonSiegelSvenssonModel,
    YieldCurveBootstrapper,
    YieldCurvePCA,
)
from advanced_quant_ml.kalman import KalmanFilterPairs, KalmanPairsStrategy
from advanced_quant_ml.ml_predictor import FinancialFeatureEngineer, MLReturnPredictor
from advanced_quant_ml.alternative_data import AlternativeDataAlphaModel
from advanced_quant_ml.visualization.plots import (
    plot_garch_forecast,
    plot_yield_curve,
    plot_kalman_hedge_ratio,
    plot_ml_predicted_returns,
    plot_alternative_data_signal,
    plot_master_advanced_quant_infographic,
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

    print_section("QUANTITATIVE ADVANCED QUANT & ML (21-25) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # =========================================================================
    # MODULE 21: VOLATILITY FORECASTING WITH GARCH(1,1)
    # =========================================================================
    print_section("VOLATILITY FORECASTING WITH GARCH", "21")
    spy_returns = load_equity_returns(data_dir=str(data_dir), ticker="SPY")
    dates = spy_returns.index

    print("Fitting GJR-GARCH(1,1) Model via Maximum Likelihood Estimation (MLE)...")
    garch_model = GARCHModel(model_type="GJR-GARCH")
    fit_res = garch_model.fit(spy_returns)
    print(fit_res.summary())

    print(f"Persistence (alpha + beta + gamma/2): {fit_res.persistence:.4f}")
    print(f"Unconditional Volatility (Annualized): {fit_res.unconditional_volatility_ann:.2%}")
    print(f"Volatility Shock Half-Life:           {fit_res.half_life_days:.1f} trading days")

    # Multi-step 30-day forecast
    forecast_res = garch_model.forecast(horizon=30)
    print(f"\n30-Day Term Structure Volatility Forecast: {forecast_res.cumulative_annualized_volatility[-1]:.2%}")

    actual_realized_vol = spy_returns.rolling(21).std().dropna()
    common_vol_dates = actual_realized_vol.index
    garch_vol_series = fit_res.conditional_volatility.loc[common_vol_dates]

    garch_plot_data = {
        "dates": common_vol_dates,
        "actual_vol": actual_realized_vol,
        "garch_vol": garch_vol_series,
    }
    plot_garch_forecast(
        dates=garch_plot_data["dates"],
        actual_vol=garch_plot_data["actual_vol"],
        garch_vol=garch_plot_data["garch_vol"],
        output_path=str(output_dir / "21_garch_volatility_forecast.png"),
    )
    print(f"  -> Saved chart: {output_dir / '21_garch_volatility_forecast.png'}")

    # =========================================================================
    # MODULE 22: YIELD CURVE MODELING & BOOTSTRAPPING
    # =========================================================================
    print_section("YIELD CURVE MODELING & TERM STRUCTURE", "22")
    yields_df = load_yield_curve_data(data_dir=str(data_dir))
    latest_par_yields = yields_df.iloc[-1].values
    latest_date_str = yields_df.index[-1].strftime("%Y-%m-%d")

    print(f"Calibrating Nelson-Siegel Model to US Treasury Yield Curve ({latest_date_str})...")
    ns_model = NelsonSiegelModel()
    ns_fit = ns_model.fit(YIELD_CURVE_MATURITIES, latest_par_yields)
    print(ns_fit.summary())

    # Bootstrapping zero rates
    boot_res = YieldCurveBootstrapper.bootstrap_par_yields(YIELD_CURVE_MATURITIES, latest_par_yields)
    print(f"10Y Zero-Coupon Spot Rate:  {boot_res.get_zero_rate(10.0):.4f}%")
    print(f"10Y Instantaneous Forward:  {ns_fit.predict_forward(10.0):.4f}%")
    print(f"10Y Discount Factor P(0,T): {boot_res.get_discount_factor(10.0):.4f}")

    # Yield curve PCA
    pca = YieldCurvePCA(n_components=3).fit(yields_df)
    print(f"\nYield Curve PCA (Level={pca.explained_variance_ratio[0]:.1%}, Slope={pca.explained_variance_ratio[1]:.1%}, Twist={pca.explained_variance_ratio[2]:.1%})")

    yield_plot_data = {
        "maturities": YIELD_CURVE_MATURITIES,
        "par_yields": latest_par_yields,
        "tenor_labels": YIELD_CURVE_TENORS,
    }
    plot_yield_curve(
        maturities=YIELD_CURVE_MATURITIES,
        par_yields=latest_par_yields,
        tenor_labels=YIELD_CURVE_TENORS,
        output_path=str(output_dir / "22_yield_curve_modeling.png"),
    )
    print(f"  -> Saved chart: {output_dir / '22_yield_curve_modeling.png'}")

    # =========================================================================
    # MODULE 23: KALMAN FILTER FOR DYNAMIC PAIRS TRADING
    # =========================================================================
    print_section("KALMAN FILTER FOR DYNAMIC PAIRS TRADING", "23")
    eq_returns = load_equity_returns(data_dir=str(data_dir))
    spy_p = (1.0 + eq_returns["SPY"]).cumprod() * 100.0
    qqq_p = (1.0 + eq_returns["QQQ"]).cumprod() * 100.0

    print("Running Online Recursive Kalman Filter for Time-Varying Beta...")
    kf = KalmanFilterPairs(delta=1e-4, observation_cov=1e-3)
    kf_res = kf.filter(y=qqq_p, x=spy_p)

    print(f"Initial Hedge Ratio β_0: {kf_res.beta.iloc[0]:.4f}")
    print(f"Final Hedge Ratio β_T:   {kf_res.beta.iloc[-1]:.4f}")
    print(f"Mean Hedge Ratio:        {np.mean(kf_res.beta):.4f} (Std: {np.std(kf_res.beta):.4f})")

    strat = KalmanPairsStrategy(delta=1e-4, observation_cov=1e-3, z_entry=1.8, z_exit=0.4)
    strat_res = strat.backtest(y=qqq_p, x=spy_p)
    print("\nAdaptive Kalman Pairs Strategy Performance:")
    print(strat_res.summary_table().to_string())

    time_steps = np.arange(len(kf_res.beta))
    kalman_plot_data = {
        "time_steps": time_steps[:252],
        "hedge_ratios": kf_res.beta[:252],
    }
    plot_kalman_hedge_ratio(
        time_steps=kalman_plot_data["time_steps"],
        hedge_ratios=kalman_plot_data["hedge_ratios"],
        output_path=str(output_dir / "23_kalman_filter_pairs.png"),
    )
    print(f"  -> Saved chart: {output_dir / '23_kalman_filter_pairs.png'}")

    # =========================================================================
    # MODULE 24: MACHINE LEARNING RETURN PREDICTOR
    # =========================================================================
    print_section("MACHINE LEARNING RETURN PREDICTOR", "24")
    ohlc_df = pd.DataFrame({
        "Open": spy_p * (1.0 - 0.002),
        "High": spy_p * (1.0 + 0.008),
        "Low": spy_p * (1.0 - 0.008),
        "Close": spy_p,
        "Volume": np.random.lognormal(16, 0.4, len(spy_p)),
    }, index=spy_p.index)

    print("Engineering High-Signal Features (Fractional Diff, Momentum, Volatility, Oscillators)...")
    fe = FinancialFeatureEngineer(momentum_windows=[1, 5, 21, 63], volatility_windows=[21, 63], frac_diff_d=0.40)
    X, y = fe.engineer_features(ohlc_df, include_target=True)
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    print(f"Engineered {X_clean.shape[1]} features across {len(X_clean)} trading days.")

    print("Training ML Model with Purged TimeSeries Cross-Validation...")
    ml_predictor = MLReturnPredictor(model_type="ridge", alpha=1.0, n_splits=5, purge_window=5, embargo_window=5)
    ml_res = ml_predictor.fit_predict_cv(X_clean, y_clean)

    print(f"Out-of-Sample Information Coefficient (IC): {ml_res.information_coefficient:+.4f} (p-value: {ml_res.ic_pvalue:.4e})")
    print(f"Out-of-Sample Rank IC (Spearman):           {ml_res.rank_ic:+.4f}")
    print(f"Directional Hit Rate:                       {ml_res.directional_hit_rate:.1%}")

    ml_plot_data = {
        "actual_returns": y_clean.values,
        "predicted_returns": ml_res.predictions.values,
        "ic": ml_res.information_coefficient,
    }
    plot_ml_predicted_returns(
        actual_returns=ml_plot_data["actual_returns"],
        predicted_returns=ml_plot_data["predicted_returns"],
        ic=ml_res.information_coefficient,
        output_path=str(output_dir / "24_ml_return_prediction.png"),
    )
    print(f"  -> Saved chart: {output_dir / '24_ml_return_prediction.png'}")

    # =========================================================================
    # MODULE 25: ALTERNATIVE DATA ALPHA MODEL
    # =========================================================================
    print_section("ALTERNATIVE DATA ALPHA MODEL", "25")
    alt_data_raw = load_alternative_data(data_dir=str(data_dir))
    sentiment_signal = alt_data_raw["Sentiment_Score"].loc[dates]
    forward_1d_returns = spy_returns.shift(-1).loc[dates].fillna(0.0)

    # Standardize signal
    norm_signal = (sentiment_signal - sentiment_signal.rolling(60).mean()) / sentiment_signal.rolling(60).std()
    norm_signal = norm_signal.fillna(0.0)

    # Ingest synthetic multi-asset alternative data
    synth_data = AlternativeDataAlphaModel.generate_synthetic_data(n_stocks=20, n_days=800, seed=42)
    alt_model = AlternativeDataAlphaModel(decay_factor=0.85, n_quantiles=5)

    signals_dict = {
        "sentiment": synth_data["sentiment"],
        "web_traffic": synth_data["web_traffic"],
        "supply_chain": synth_data["supply_chain"],
    }
    combined_signal = alt_model.combine_signals(signals_dict)
    smoothed_signal = alt_model.exponential_decay_smoothing(combined_signal)
    neutral_signal = alt_model.neutralize_factors(smoothed_signal, synth_data["risk_loadings"])

    ic_report = alt_model.compute_ic_decay(neutral_signal, synth_data["prices"], horizons=[1, 2, 5, 10, 21])
    print("Alternative Data Multi-Horizon IC Decay Analysis:")
    print(ic_report.summary_table().to_string())

    bt_alt = alt_model.backtest_long_short(neutral_signal, synth_data["prices"], transaction_cost_bps=5.0)
    print("\nDollar-Neutral Long/Short Alternative Alpha Strategy:")
    print(bt_alt.summary_table().to_string())

    alt_plot_data = {
        "dates": dates[-252:],
        "signal_strength": norm_signal.iloc[-252:],
        "forward_returns": forward_1d_returns.iloc[-252:],
    }
    plot_alternative_data_signal(
        dates=alt_plot_data["dates"],
        signal_strength=alt_plot_data["signal_strength"],
        forward_returns=alt_plot_data["forward_returns"],
        output_path=str(output_dir / "25_alternative_data_alpha.png"),
    )
    print(f"  -> Saved chart: {output_dir / '25_alternative_data_alpha.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (21-25)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (21-25)")
    master_path = output_dir / "advanced_quant_ml_infographic.png"
    plot_master_advanced_quant_infographic(
        garch_data=garch_plot_data,
        yield_data=yield_plot_data,
        kalman_data=kalman_plot_data,
        ml_data=ml_plot_data,
        alt_data=alt_plot_data,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll Advanced Quant & ML demos executed successfully!")


if __name__ == "__main__":
    main()
