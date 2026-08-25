#!/usr/bin/env python3
"""Master demonstration runner for Climate Quantitative Finance & Carbon Markets (36-40).

Executes all 5 modules:
1. Carbon Allowance Pricing & ETS Fuel-Switching Dynamics (36)
2. Green Bond Valuation & Greenium Decomposition Engine (37)
3. NGFS Climate Scenario Stress Testing & Physical/Transition Risk Engine (38)
4. Geospatial & Satellite GHG Emissions Alternative Data Alpha (39)
5. Renewable Energy PPA Valuation & Weather Derivatives (40)

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

from climate_quant.data.loader import (
    generate_carbon_market_data,
    generate_green_bond_pairs,
    generate_corporate_climate_universe,
    generate_satellite_plume_observations,
    generate_disclosed_emissions_universe,
    generate_climate_equity_prices,
)
from climate_quant.carbon_pricing import CarbonAllowanceModel
from climate_quant.green_bonds import GreenBondValuationEngine
from climate_quant.climate_stress import NGFSClimateStressEngine, NGFSScenarioType
from climate_quant.emissions_satellite import SatelliteEmissionsAlpha
from climate_quant.renewable_energy import (
    RenewablePPAValuator,
    RenewableAssetConfig,
    PPAContract,
    PPAType,
    WeatherDerivativePricer,
    TemperatureModelParams,
    WeatherDerivativeContract,
    WeatherContractType,
)
from climate_quant.visualization.plots import (
    plot_fuel_switching_parity,
    plot_greenium_term_structure,
    plot_climate_var_stress,
    plot_satellite_emissions_alpha,
    plot_ppa_cannibalization_curve,
    plot_master_climate_infographic,
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

    print_section("CLIMATE QUANTITATIVE FINANCE & CARBON MARKETS (36-40) DEMO SUITE")
    print(f"Working Directory: {project_root}")
    print(f"Output Directory:  {output_dir}")

    # =========================================================================
    # MODULE 36: CARBON ALLOWANCE PRICING & ETS FUEL-SWITCHING DYNAMICS
    # =========================================================================
    print_section("CARBON ALLOWANCE PRICING & ETS FUEL-SWITCHING DYNAMICS", "36")
    carbon_model = CarbonAllowanceModel(
        efficiency_gas=0.50,
        efficiency_coal=0.38,
        emission_factor_gas=0.37,
        emission_factor_coal=0.95,
    )

    power_p = 105.0
    gas_p = 26.5
    coal_p = 13.5
    carbon_prices_arr = np.linspace(20.0, 120.0, 100)

    clean_sparks = power_p - (gas_p / 0.50) - 0.37 * carbon_prices_arr
    clean_darks = power_p - (coal_p / 0.38) - 0.95 * carbon_prices_arr
    switching_price = ((gas_p / 0.50) - (coal_p / 0.38)) / (0.95 - 0.37)

    snap = carbon_model.evaluate_spread_snapshot(power_p, gas_p, coal_p, carbon_price=75.0)
    print(f"EU ETS Market Snapshot Evaluation (€75.0/tCO2 Carbon Allowance):")
    print(f"  - Clean Spark Spread (Gas):  €{snap.clean_spark_spread:+.2f} / MWh_e")
    print(f"  - Clean Dark Spread (Coal):  €{snap.clean_dark_spread:+.2f} / MWh_e")
    print(f"  - Fuel-Switching Parity:     €{snap.fuel_switch_parity_price:.2f} / tCO2")
    print(f"  - Dominant Merit Fuel:       {snap.dominant_fuel} (Gas favored: {snap.clean_spark_spread > snap.clean_dark_spread})")

    fuel_plot_data = {
        "carbon_prices": carbon_prices_arr,
        "clean_spark_spreads": clean_sparks,
        "clean_dark_spreads": clean_darks,
        "switching_price": switching_price,
    }
    plot_fuel_switching_parity(
        carbon_prices=carbon_prices_arr,
        clean_spark_spreads=clean_sparks,
        clean_dark_spreads=clean_darks,
        switching_price=switching_price,
        output_path=str(output_dir / "36_fuel_switching_parity.png"),
    )
    print(f"  -> Saved chart: {output_dir / '36_fuel_switching_parity.png'}")

    # =========================================================================
    # MODULE 37: GREEN BOND VALUATION & GREENIUM DECOMPOSITION ENGINE
    # =========================================================================
    print_section("GREEN BOND VALUATION & GREENIUM DECOMPOSITION ENGINE", "37")
    gb_engine = GreenBondValuationEngine(default_yield_curve_slope=0.08)
    pairs_df = generate_green_bond_pairs(n_pairs=30, seed=42)

    decomp_df = gb_engine.decompose_universe(pairs_df)
    print("Matched-Pair Green vs Conventional Twin Bond Decomposition (Top 6 Pairs):")
    print(decomp_df[["Issuer", "Raw Spread (bps)", "Pure Fundamental Greenium (bps)", "Liquidity Adj (bps)"]].head(6).to_string(index=False))

    attr_res = gb_engine.attribute_factors(pairs_df)
    print("\nGreenium Factor Attribution Regression:")
    print(attr_res.summary_dataframe().to_string(index=False))

    maturities_arr = np.linspace(1.0, 30.0, 50)
    sample_mat = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    sample_gr = np.array([2.1, 2.9, 4.2, 5.1, 6.2, 7.0, 7.8, 8.4])
    ns_fit = gb_engine.fit_nelson_siegel_term_structure(sample_mat, sample_gr)
    fitted_gr = ns_fit.predict(maturities_arr)

    greenium_plot_data = {
        "maturities": sample_mat,
        "twin_bond_greeniums": sample_gr,
        "fitted_ns_greenium": ns_fit.predict(sample_mat),
    }
    plot_greenium_term_structure(
        maturities=sample_mat,
        twin_bond_greeniums=sample_gr,
        fitted_ns_greenium=ns_fit.predict(sample_mat),
        output_path=str(output_dir / "37_greenium_term_structure.png"),
    )
    print(f"  -> Saved chart: {output_dir / '37_greenium_term_structure.png'}")

    # =========================================================================
    # MODULE 38: NGFS CLIMATE SCENARIO STRESS TESTING ENGINE
    # =========================================================================
    print_section("NGFS CLIMATE SCENARIO STRESS TESTING ENGINE", "38")
    climate_portfolio = generate_corporate_climate_universe()
    ngfs_engine = NGFSClimateStressEngine()

    ngfs_comp_df = ngfs_engine.multi_scenario_comparison(climate_portfolio, year=2030)
    print("NGFS Phase IV Multi-Scenario Stress Test Results (Year: 2030):")
    print(ngfs_comp_df.to_string(index=False))

    sectors = ["Energy & Fossil", "Utilities & Power", "Materials & Steel", "Industrials", "Tech & Cloud", "Financials"]
    orderly_v = [-28.4, -24.1, -19.5, -8.2, -0.4, -1.2]
    disorderly_v = [-38.2, -32.5, -27.1, -12.4, -0.8, -2.1]
    hot_house_v = [-14.2, -18.6, -15.1, -16.8, -8.5, -9.4]

    stress_plot_data = {
        "sectors": sectors,
        "orderly_var": orderly_v,
        "disorderly_var": disorderly_v,
        "hot_house_var": hot_house_v,
    }
    plot_climate_var_stress(
        sectors=sectors,
        orderly_var=orderly_v,
        disorderly_var=disorderly_v,
        hot_house_var=hot_house_v,
        output_path=str(output_dir / "38_climate_var_stress.png"),
    )
    print(f"  -> Saved chart: {output_dir / '38_climate_var_stress.png'}")

    # =========================================================================
    # MODULE 39: GEOSPATIAL & SATELLITE GHG EMISSIONS ALPHA
    # =========================================================================
    print_section("GEOSPATIAL & SATELLITE GHG EMISSIONS ALPHA", "39")
    disclosed_emissions = generate_disclosed_emissions_universe()
    observations = generate_satellite_plume_observations()
    prices_df = generate_climate_equity_prices()

    alpha_engine = SatelliteEmissionsAlpha(decay_half_life_days=30.0, transaction_cost_bps=5.0)
    satellite_aggr = alpha_engine.aggregate_facility_plumes(observations)
    surprise_df = alpha_engine.compute_emissions_surprises(disclosed_emissions, satellite_aggr)

    print("Satellite GHG Plume Surprise Z-Scores & Alpha Signals (Top 6):")
    print(surprise_df[["ticker", "sector", "sector_z_score", "alpha_signal", "recommendation"]].head(6).to_string(index=False))

    signals_dict = {prices_df.index[0]: surprise_df}
    bt_res = alpha_engine.backtest_strategy(prices_df=prices_df, signals_dict_by_date=signals_dict, rebalance_freq_days=21)
    print("\nSatellite Emissions Long/Short Strategy Performance:")
    print(bt_res.summary_table().to_string(index=False))

    satellite_plot_data = {
        "dates": bt_res.dates,
        "strategy_equity": bt_res.strategy_equity,
        "long_leg_equity": bt_res.long_leg_equity,
        "short_leg_equity": bt_res.short_leg_equity,
    }
    plot_satellite_emissions_alpha(
        dates=bt_res.dates,
        strategy_equity=bt_res.strategy_equity,
        long_leg_equity=bt_res.long_leg_equity,
        short_leg_equity=bt_res.short_leg_equity,
        output_path=str(output_dir / "39_satellite_emissions_alpha.png"),
    )
    print(f"  -> Saved chart: {output_dir / '39_satellite_emissions_alpha.png'}")

    # =========================================================================
    # MODULE 40: RENEWABLE ENERGY PPA VALUATION & WEATHER DERIVATIVES
    # =========================================================================
    print_section("RENEWABLE ENERGY PPA VALUATION & WEATHER DERIVATIVES", "40")
    config = RenewableAssetConfig(asset_type="solar", capacity_mw=100.0, latitude_deg=35.0, opex_per_mwh=7.50, discount_rate=0.065)
    valuator = RenewablePPAValuator(config)

    solar_gen = valuator.simulate_hourly_solar_profile(year=2026, seed=42)
    spot_prices = valuator.simulate_electricity_spot_prices(solar_gen, base_price_mwh=60.0, cannibalization_beta=0.35, seed=42)

    base_p, cap_p, cap_rate, cann_disc = valuator.calculate_capture_metrics(solar_gen.generation_mwh, spot_prices.values)
    print(f"100 MW Solar Asset Market Merit-Order Diagnostics:")
    print(f"  - Baseload Market Price:     ${base_p:.2f} / MWh")
    print(f"  - Solar Capture Price:       ${cap_p:.2f} / MWh")
    print(f"  - Solar Capture Rate:        {cap_rate:.2%}")
    print(f"  - Cannibalization Discount:  {cann_disc:.2f}%")

    pap_contract = PPAContract(contract_type=PPAType.PAY_AS_PRODUCED, strike_price=52.0, tenor_years=10)
    ppa_res = valuator.value_ppa_contract(solar_gen, spot_prices, pap_contract)
    print(f"  - 10-Year PAP PPA 10Y NPV:   ${ppa_res.npv_net_cashflow_usd:,.0f} (${ppa_res.npv_per_kw_installed:.2f}/kW)")

    # Diurnal 24-hour snapshot for Duck Curve visualization
    hours_24 = np.arange(24)
    solar_24 = solar_gen.generation_mwh[120:144]  # Midday day
    prices_24 = spot_prices.values[120:144]

    ppa_plot_data = {
        "hours": hours_24,
        "solar_generation": solar_24,
        "market_spot_price": prices_24,
    }
    plot_ppa_cannibalization_curve(
        hours=hours_24,
        solar_generation=solar_24,
        market_spot_price=prices_24,
        output_path=str(output_dir / "40_ppa_duck_curve_cannibalization.png"),
    )
    print(f"  -> Saved chart: {output_dir / '40_ppa_duck_curve_cannibalization.png'}")

    # =========================================================================
    # MASTER COMPOSITE INFOGRAPHIC (36-40)
    # =========================================================================
    print_section("COMPOSITING MASTER INFOGRAPHIC (36-40)")
    master_path = output_dir / "climate_quant_infographic.png"
    plot_master_climate_infographic(
        fuel_data=fuel_plot_data,
        greenium_data=greenium_plot_data,
        stress_data=stress_plot_data,
        satellite_data=satellite_plot_data,
        ppa_data=ppa_plot_data,
        output_path=str(master_path),
    )
    print(f"  -> Successfully generated master infographic: {master_path}")
    print("\nAll Climate Quantitative Finance & Carbon Markets Demos completed successfully!")


if __name__ == "__main__":
    main()
