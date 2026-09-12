"""
Example 05: Climate Quantitative Finance, Carbon Markets & ESG Desk Workflow
=============================================================================
"""
import nexus_quant as nq

# 1. EU ETS Fuel-Switching Carbon Parity
carbon_model = nq.macro.carbon_ets.CarbonAllowanceModel()
spreads = carbon_model.evaluate_spreads(
    power_price_mwh=105.0,
    gas_price_mwh=26.0,
    coal_price_mwh=13.0,
    carbon_price_tco2=75.0,
)
print("EU ETS Carbon Allowance & Fuel-Switching Economics:")
print(f"  - Clean Spark Spread (Gas):  €{spreads.clean_spark_spread:+.2f}/MWh")
print(f"  - Clean Dark Spread (Coal):  €{spreads.clean_dark_spread:+.2f}/MWh")
print(f"  - Fuel-Switch Parity Price:  €{spreads.fuel_switch_parity_price:.2f}/tCO2")
print(f"  - Dominant Merit Order Fuel: {spreads.merit_order_dominant_fuel}")

# 2. Green Bond Greenium Decomposition
green_engine = nq.macro.carbon_ets.GreenBondValuationEngine()
decomp = green_engine.decompose_twin_pair(
    green_yield_pct=3.85,
    vanilla_yield_pct=3.92,
    green_bid_ask_bps=4.0,
    vanilla_bid_ask_bps=2.0,
    green_duration=7.5,
    vanilla_duration=7.5,
)
print("\nGreen Bond Greenium Decomposition:")
print(f"  - Raw Spread:       {decomp.raw_greenium_bps:.1f} bps")
print(f"  - Pure Greenium:    {decomp.pure_fundamental_greenium_bps:.1f} bps")
print(f"  - Liquidity Impact: {decomp.liquidity_adjustment_bps:.1f} bps")

# 3. NGFS Climate Scenario DCF Climate VaR
climate_engine = nq.risk.climate_risk.NGFSClimateStressEngine()
c_var = climate_engine.evaluate_corporate_climate_var(
    company_name="HeavyIndustries Inc",
    market_cap_m=20_000.0,
    ebitda_m=4_000.0,
    scope1_t=8_000_000.0,
    scope2_t=1_500_000.0,
    scope3_t=25_000_000.0,
    scenario_name="Net Zero 2050",
)
print("\nNGFS Climate Scenario Stress Test (Net Zero 2050):")
print(f"  - Carbon Tax Level:            ${c_var['carbon_price_usd']:.0f}/tCO2")
print(f"  - Annual Transition Carbon Tax: ${c_var['annual_cost_m']:,.1f}M")
print(f"  - EBITDA Impairment:             {c_var['ebitda_hit_pct']:.1%}")
print(f"  - DCF Climate VaR Equity Loss:   {c_var['climate_var_pct']:.1%}")
