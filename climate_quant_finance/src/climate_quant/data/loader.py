"""Climate Quantitative Finance Data Generation and Ingestion Suite.

Provides data generators and loaders for:
1. Carbon Market Data: EU ETS EUA spot and futures, Gas TTF, Coal ARA, Electricity Baseload/Peakload.
2. Green Bond Twin Pairs: Matched green vs conventional twin bonds with issuer ratings and spreads.
3. Corporate Emissions Data: Scope 1, Scope 2, Scope 3 emissions, sector classifications, carbon intensity.
4. Satellite Plume Emissions Data: Facility-level satellite plume observations vs reported emissions.
5. Renewable Generation Series: Hourly wind, solar, hydro generation, spot power, and capture prices.
6. Temperature Series: Multi-year daily temperature observations, Heating/Cooling Degree Days (HDD/CDD).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


# =========================================================================
# 1. CARBON MARKET DATA GENERATOR (Module 36)
# =========================================================================

def generate_carbon_market_data(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generates realistic daily EU ETS EUA carbon allowance and energy market prices.
    
    Includes:
    - EUA_Carbon_Spot (€/tCO2): Rising from ~€25 in 2020 to ~€80-€100 in 2023-2024 with policy jumps.
    - Gas_TTF (€/MWh_th): European benchmark natural gas price (including 2022 energy crisis spike).
    - Coal_ARA (€/MWh_th): European thermal coal CIF ARA converted to thermal energy equivalent.
    - Power_Baseload (€/MWh_e): German/Nordic baseload wholesale electricity price.
    - Power_Peakload (€/MWh_e): Peakload wholesale electricity price.
    - EUA_Futures_Dec1Y (€/tCO2): 1-year forward EUA December futures contract.
    - EUA_Futures_Dec2Y (€/tCO2): 2-year forward EUA December futures contract.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)
    rng = np.random.default_rng(seed)

    # Base EUA path with mean reversion, long-term upward regulatory trend, and crisis jumps
    t_years = np.linspace(0, n_days / 252.0, n_days)
    trend_eua = 25.0 + 15.0 * t_years + 10.0 * np.sin(2 * np.pi * t_years / 3.0)
    
    # Gas TTF path with 2022 Russian supply shock spike
    gas_base = 20.0 + 8.0 * np.sin(2 * np.pi * t_years)
    # Inject 2022 spike (around day 500-750)
    spike_weight = np.exp(-0.5 * ((np.arange(n_days) - 650) / 90.0) ** 2)
    gas_prices = gas_base + 180.0 * spike_weight + rng.normal(0, 3.5, n_days)
    gas_prices = np.clip(gas_prices, 12.0, 340.0)

    # Coal ARA path (CIF ARA $/t converted to €/MWh_th, thermal energy factor ~8.14 MWh_th/t)
    coal_base = 12.0 + 4.0 * np.sin(2 * np.pi * t_years + 0.5)
    coal_prices = coal_base + 45.0 * spike_weight + rng.normal(0, 1.8, n_days)
    coal_prices = np.clip(coal_prices, 8.0, 95.0)

    # EUA spot with Ornstein-Uhlenbeck + jump diffusion
    eua_prices = np.zeros(n_days)
    eua_prices[0] = 24.50
    kappa_eua = 0.015
    vol_eua = 0.025
    jump_intensity = 0.015  # ~4 jumps per year (EU policy / Fit for 55 announcements)

    for i in range(1, n_days):
        dt = 1.0 / 252.0
        drift = kappa_eua * (trend_eua[i] - eua_prices[i-1]) * dt
        diffusion = vol_eua * np.sqrt(dt) * rng.normal(0, 1.0)
        jump = 0.0
        if rng.random() < jump_intensity:
            jump = rng.normal(0.02, 0.08)  # Upward/downward policy jump
        
        eua_prices[i] = max(10.0, eua_prices[i-1] * (1.0 + drift + diffusion + jump))

    # Power prices (Marginal cost stack driven by Gas/Coal/Carbon)
    # Gas plant efficiency ~50%, emission factor ~0.37 tCO2/MWh_e
    # Coal plant efficiency ~38%, emission factor ~0.95 tCO2/MWh_e
    marginal_gas_cost = (gas_prices / 0.50) + (0.37 * eua_prices)
    marginal_coal_cost = (coal_prices / 0.38) + (0.95 * eua_prices)
    
    # Baseload power is weighted blend with renewable merit-order downward pressure
    power_baseload = np.maximum(marginal_gas_cost, marginal_coal_cost) * 0.85 + 15.0 + rng.normal(0, 4.0, n_days)
    power_peakload = power_baseload * 1.35 + rng.normal(0, 6.0, n_days)

    # EUA Futures Curve (Cost of carry with storage & convenience yield)
    r = 0.03
    conv_yield_1y = 0.008
    conv_yield_2y = 0.012
    eua_futures_1y = eua_prices * np.exp((r - conv_yield_1y) * 1.0) + rng.normal(0, 0.4, n_days)
    eua_futures_2y = eua_prices * np.exp((r - conv_yield_2y) * 2.0) + rng.normal(0, 0.7, n_days)

    df = pd.DataFrame({
        "Date": dates,
        "EUA_Carbon_Spot": np.round(eua_prices, 2),
        "Gas_TTF": np.round(gas_prices, 2),
        "Coal_ARA": np.round(coal_prices, 2),
        "Power_Baseload": np.round(power_baseload, 2),
        "Power_Peakload": np.round(power_peakload, 2),
        "EUA_Futures_Dec1Y": np.round(eua_futures_1y, 2),
        "EUA_Futures_Dec2Y": np.round(eua_futures_2y, 2),
    }).set_index("Date")

    return df


def load_carbon_market_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads or generates carbon market dataset."""
    return generate_carbon_market_data()


# =========================================================================
# 2. GREEN BOND TWIN PAIRS GENERATOR (Module 37)
# =========================================================================

def generate_green_bond_pairs(
    n_pairs: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates matched-pair Green Bonds vs Conventional Vanilla Twin Bonds.
    
    Controls for issuer, seniority, currency, rating, and approximates maturity matching.
    Calculates the raw Greenium: Spread = Vanilla_Yield - Green_Yield (in basis points).
    """
    rng = np.random.default_rng(seed)

    issuers = [
        ("Republic of France (OAT)", "Sovereign", "AA", 92.0),
        ("Federal Republic of Germany (Bund)", "Sovereign", "AAA", 96.0),
        ("KfW Bankengruppe", "Supranational", "AAA", 94.0),
        ("European Investment Bank (EIB)", "Supranational", "AAA", 95.0),
        ("Kingdom of the Netherlands", "Sovereign", "AAA", 91.0),
        ("Enel S.p.A.", "Utilities", "BBB+", 88.0),
        ("Iberdrola S.A.", "Utilities", "A-", 89.0),
        ("Orsted A/S", "Renewable Energy", "A-", 93.0),
        ("Electricite de France (EDF)", "Utilities", "BBB", 82.0),
        ("Engie S.A.", "Utilities", "BBB+", 85.0),
        ("Apple Inc.", "Technology", "AA+", 87.0),
        ("Alphabet Inc.", "Technology", "AA+", 86.0),
        ("Schneider Electric SE", "Industrials", "A-", 90.0),
        ("BNP Paribas SA", "Financials", "A+", 84.0),
        ("Societe Generale SA", "Financials", "A", 81.0),
        ("ING Groep NV", "Financials", "A+", 86.0),
        ("Volkswagen AG", "Automotive", "BBB+", 78.0),
        ("Mercedes-Benz Group AG", "Automotive", "A", 83.0),
        ("Siemens Energy AG", "Capital Goods", "BBB-", 85.0),
        ("Repsol S.A.", "Energy", "BBB", 75.0),
    ]

    records = []
    for i in range(n_pairs):
        issuer_idx = i % len(issuers)
        name, sector, rating, esg_score = issuers[issuer_idx]
        
        maturity_years = float(rng.choice([3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]))
        # Vanilla base yield according to maturity & rating
        base_rate = 2.20 + 0.08 * maturity_years
        rating_spread_map = {"AAA": 0.15, "AA+": 0.25, "AA": 0.40, "A+": 0.65, "A": 0.80, "A-": 0.95, "BBB+": 1.30, "BBB": 1.60, "BBB-": 2.10}
        credit_spread = rating_spread_map.get(rating, 1.0)
        
        vanilla_yield = base_rate + credit_spread + rng.normal(0, 0.05)
        
        # Fundamental Greenium effect: Green bonds trade at lower yield (typically 2 to 9 bps lower)
        # Driven by ESG score, rating quality, and maturity
        fundamental_greenium_bps = 2.0 + (esg_score / 100.0) * 4.0 + (0.15 * maturity_years) + rng.normal(0, 0.8)
        fundamental_greenium_bps = np.clip(fundamental_greenium_bps, 0.5, 12.0)
        
        green_yield = vanilla_yield - (fundamental_greenium_bps / 100.0)
        
        # Liquidity bid-ask spreads
        green_ba_bps = float(rng.uniform(1.8, 4.5))
        vanilla_ba_bps = float(rng.uniform(1.2, 3.8))
        
        records.append({
            "Pair_ID": f"PAIR_{i+1:03d}",
            "Issuer": name,
            "Sector": sector,
            "Credit_Rating": rating,
            "Maturity_Years": maturity_years,
            "Green_Bond_ISIN": f"XS{200000000 + i*10 + 1:09d}",
            "Green_Yield_Pct": np.round(green_yield, 4),
            "Green_Coupon_Pct": np.round(green_yield + rng.uniform(-0.3, 0.3), 2),
            "Green_Duration": np.round(maturity_years * 0.88, 2),
            "Green_Bid_Ask_bps": np.round(green_ba_bps, 2),
            "Green_Issuance_EUR_M": int(rng.choice([500, 750, 1000, 1500, 2000])),
            "Vanilla_Bond_ISIN": f"XS{200000000 + i*10 + 2:09d}",
            "Vanilla_Yield_Pct": np.round(vanilla_yield, 4),
            "Vanilla_Coupon_Pct": np.round(vanilla_yield + rng.uniform(-0.3, 0.3), 2),
            "Vanilla_Duration": np.round(maturity_years * 0.87, 2),
            "Vanilla_Bid_Ask_bps": np.round(vanilla_ba_bps, 2),
            "Vanilla_Issuance_EUR_M": int(rng.choice([500, 750, 1000, 1500, 2000])),
            "Raw_Greenium_bps": np.round(fundamental_greenium_bps, 2),
            "ESG_Score": esg_score,
            "Carbon_Intensity_tCO2_EUR_M": np.round(float(rng.uniform(15.0, 350.0)), 1),
        })

    return pd.DataFrame(records)


def load_green_bond_pairs(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads green bond twin pairs dataset."""
    return generate_green_bond_pairs()


# =========================================================================
# 3. CORPORATE EMISSIONS DATA GENERATOR (Module 38 / 39)
# =========================================================================

def generate_corporate_emissions_data(
    n_companies: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates corporate Scope 1, Scope 2, Scope 3 greenhouse gas emissions profiles."""
    rng = np.random.default_rng(seed)
    sectors = ["Energy", "Utilities", "Materials", "Industrials", "Information Technology", "Financials", "Consumer Goods"]
    
    tickers = [f"CORP_{i+1:02d}" for i in range(n_companies)]
    records = []
    
    for i, ticker in enumerate(tickers):
        sec = sectors[i % len(sectors)]
        rev = float(rng.uniform(2.0, 95.0))  # EUR Billions
        mkt_cap = rev * float(rng.uniform(1.2, 5.5))

        # Emission profiles depend on sector
        if sec == "Energy":
            s1 = rev * rng.uniform(800.0, 2500.0)
            s2 = rev * rng.uniform(50.0, 150.0)
            s3 = rev * rng.uniform(3000.0, 8000.0)
        elif sec == "Utilities":
            s1 = rev * rng.uniform(1200.0, 3500.0)
            s2 = rev * rng.uniform(40.0, 100.0)
            s3 = rev * rng.uniform(500.0, 2000.0)
        elif sec == "Materials":
            s1 = rev * rng.uniform(600.0, 1800.0)
            s2 = rev * rng.uniform(100.0, 400.0)
            s3 = rev * rng.uniform(800.0, 3000.0)
        elif sec == "Industrials":
            s1 = rev * rng.uniform(100.0, 400.0)
            s2 = rev * rng.uniform(50.0, 180.0)
            s3 = rev * rng.uniform(500.0, 2200.0)
        elif sec == "Information Technology":
            s1 = rev * rng.uniform(2.0, 15.0)
            s2 = rev * rng.uniform(20.0, 80.0)
            s3 = rev * rng.uniform(100.0, 500.0)
        else:
            s1 = rev * rng.uniform(20.0, 100.0)
            s2 = rev * rng.uniform(25.0, 80.0)
            s3 = rev * rng.uniform(200.0, 900.0)

        total_s12 = s1 + s2
        intensity_s12 = total_s12 / rev  # tCO2e / EUR M
        
        records.append({
            "Ticker": ticker,
            "Sector": sec,
            "Revenue_EUR_B": np.round(rev, 2),
            "Market_Cap_EUR_B": np.round(mkt_cap, 2),
            "Scope_1_Emissions_ktCO2e": np.round(s1 / 1000.0, 2),
            "Scope_2_Emissions_ktCO2e": np.round(s2 / 1000.0, 2),
            "Scope_3_Emissions_ktCO2e": np.round(s3 / 1000.0, 2),
            "Total_Scope12_ktCO2e": np.round(total_s12 / 1000.0, 2),
            "Carbon_Intensity_Scope12_Rev": np.round(intensity_s12, 1),
            "SBTi_Committed": bool(rng.random() > 0.40),
            "Internal_Carbon_Price_EUR": float(rng.choice([0.0, 35.0, 50.0, 80.0, 100.0])),
            "Transition_Risk_Score": np.round(float(rng.uniform(15.0, 95.0)), 1),
        })

    return pd.DataFrame(records)


def load_corporate_emissions_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads corporate emissions dataset."""
    return generate_corporate_emissions_data()


# =========================================================================
# 4. SATELLITE PLUME EMISSIONS DATA (Module 38)
# =========================================================================

def generate_satellite_emissions_data(
    n_facilities: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates facility-level satellite plume observations vs corporate reported emissions."""
    rng = np.random.default_rng(seed)
    facility_types = ["Gas-Fired Power Plant", "Coal-Fired Power Plant", "Integrated Steel Mill", "Oil & Gas Refinery", "Cement Production Kiln"]
    countries = ["Germany", "Poland", "United States", "India", "China", "United Kingdom", "Netherlands", "Italy"]

    records = []
    for i in range(n_facilities):
        f_type = facility_types[i % len(facility_types)]
        country = countries[i % len(countries)]
        
        reported_kt = float(rng.uniform(200.0, 4500.0))
        # Satellite observation has measurement noise + systematic under-reporting bias
        under_reporting_bias = float(rng.choice([1.0, 1.05, 1.15, 1.35, 1.50], p=[0.4, 0.25, 0.2, 0.1, 0.05]))
        satellite_obs_kt = reported_kt * under_reporting_bias + rng.normal(0, reported_kt * 0.04)
        
        discrepancy_kt = satellite_obs_kt - reported_kt
        discrepancy_pct = (discrepancy_kt / reported_kt) * 100.0
        
        records.append({
            "Facility_ID": f"FAC_{i+1:04d}",
            "Facility_Type": f_type,
            "Country": country,
            "Latitude": np.round(float(rng.uniform(25.0, 58.0)), 4),
            "Longitude": np.round(float(rng.uniform(-100.0, 120.0)), 4),
            "Reported_Annual_Emissions_ktCO2": np.round(reported_kt, 2),
            "Satellite_Observed_Plume_ktCO2": np.round(satellite_obs_kt, 2),
            "Discrepancy_ktCO2": np.round(discrepancy_kt, 2),
            "Discrepancy_Pct": np.round(discrepancy_pct, 2),
            "Detection_Confidence_Pct": np.round(float(rng.uniform(85.0, 99.5)), 1),
            "Audit_Priority_Flag": bool(discrepancy_pct > 20.0),
        })

    return pd.DataFrame(records)


def load_satellite_emissions_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads satellite plume emissions dataset."""
    return generate_satellite_emissions_data()


# =========================================================================
# 5. RENEWABLE GENERATION & POWER CAPTURE PRICES (Module 39)
# =========================================================================

def generate_renewable_generation_data(
    n_hours: int = 8760,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates 8,760 hourly time-series of renewable power generation, spot electricity, and capture prices."""
    times = pd.date_range("2024-01-01 00:00:00", periods=n_hours, freq="h")
    rng = np.random.default_rng(seed)

    hour_of_day = times.hour.values
    day_of_year = times.dayofyear.values

    # Solar: Daily bell curve peaking at noon + seasonal summer peak
    solar_diurnal = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12.0))
    solar_seasonal = 0.5 + 0.5 * np.sin(np.pi * (day_of_year - 80) / 180.0)
    solar_mwh = 8000.0 * solar_diurnal * solar_seasonal + rng.normal(0, 300.0, n_hours)
    solar_mwh = np.maximum(0, solar_mwh)

    # Wind: Weibull/Rayleigh distributed, higher in winter and night
    wind_seasonal = 0.7 + 0.3 * np.cos(2 * np.pi * day_of_year / 365.0)
    wind_mwh = rng.weibull(2.0, n_hours) * 6000.0 * wind_seasonal
    wind_mwh = np.maximum(0, wind_mwh)

    # Base load demand: peak in morning (08:00) and evening (19:00)
    demand_diurnal = 35000.0 + 12000.0 * np.sin(np.pi * (hour_of_day - 4) / 12.0) ** 2
    demand_mwh = demand_diurnal + rng.normal(0, 1500.0, n_hours)

    # Residual thermal demand = Demand - Wind - Solar
    residual_demand = np.maximum(0, demand_mwh - wind_mwh - solar_mwh)
    
    # Merit-order price formation (Can go negative during high renewable midday hours)
    spot_price = 25.0 + 0.003 * residual_demand - 0.0015 * (wind_mwh + solar_mwh) + rng.normal(0, 8.0, n_hours)
    spot_price = np.where((wind_mwh + solar_mwh) > demand_mwh * 0.95, -15.0 + rng.normal(0, 5.0, n_hours), spot_price)

    # Capture price = Generation-weighted realized price
    capture_price_wind = spot_price * (wind_mwh / (np.mean(wind_mwh) + 1e-5))
    capture_price_solar = spot_price * (solar_mwh / (np.mean(solar_mwh) + 1e-5))

    return pd.DataFrame({
        "Timestamp": times,
        "Total_Demand_MWh": np.round(demand_mwh, 1),
        "Wind_Generation_MWh": np.round(wind_mwh, 1),
        "Solar_Generation_MWh": np.round(solar_mwh, 1),
        "Residual_Demand_MWh": np.round(residual_demand, 1),
        "Spot_Power_Price_EUR_MWh": np.round(spot_price, 2),
        "Capture_Price_Wind_EUR_MWh": np.round(capture_price_wind, 2),
        "Capture_Price_Solar_EUR_MWh": np.round(capture_price_solar, 2),
    }).set_index("Timestamp")


def load_renewable_generation_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads renewable generation dataset."""
    return generate_renewable_generation_data()


# =========================================================================
# 6. TEMPERATURE TIME-SERIES & WEATHER DERIVATIVES (Module 40)
# =========================================================================

def generate_temperature_series(
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generates daily temperature observations, HDD, and CDD for weather derivatives."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    rng = np.random.default_rng(seed)

    day_of_year = dates.dayofyear.values
    t_years = np.linspace(0, n_days / 365.25, n_days)

    # Seasonal annual cycle + climate warming trend (+0.035°C/year)
    seasonal_temp = 10.5 - 9.0 * np.cos(2 * np.pi * (day_of_year - 25) / 365.25)
    climate_trend = 0.035 * t_years
    
    # Auto-regressive AR(1) thermal persistence
    noise = np.zeros(n_days)
    phi = 0.75
    sigma_eps = 2.8
    for i in range(1, n_days):
        noise[i] = phi * noise[i-1] + rng.normal(0, sigma_eps * np.sqrt(1 - phi**2))

    daily_avg_temp = seasonal_temp + climate_trend + noise
    daily_max_temp = daily_avg_temp + rng.uniform(3.0, 7.0, n_days)
    daily_min_temp = daily_avg_temp - rng.uniform(3.0, 7.0, n_days)

    # Degree Days (Base 18°C)
    hdd = np.maximum(0.0, 18.0 - daily_avg_temp)
    cdd = np.maximum(0.0, daily_avg_temp - 18.0)

    return pd.DataFrame({
        "Date": dates,
        "Daily_Avg_Temp_C": np.round(daily_avg_temp, 2),
        "Daily_Max_Temp_C": np.round(daily_max_temp, 2),
        "Daily_Min_Temp_C": np.round(daily_min_temp, 2),
        "Heating_Degree_Days_HDD": np.round(hdd, 2),
        "Cooling_Degree_Days_CDD": np.round(cdd, 2),
    }).set_index("Date")


def load_temperature_series(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads daily temperature series."""
    return generate_temperature_series()


# =========================================================================
# SUBAGENT 2 HELPERS (Module 38 & 39)
# =========================================================================

"""Data loader & synthetic generator for Climate Quant Finance."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from climate_quant.climate_stress.ngfs_engine import CompanyClimateProfile
from climate_quant.emissions_satellite.plume_alpha import SatelliteObservation


def generate_corporate_climate_universe() -> List[CompanyClimateProfile]:
    """Generates representative multi-sector corporate universe with climate & balance sheet metrics."""
    profiles = [
        CompanyClimateProfile(
            ticker="XOM", name="ExxonMobil Corp", sector="Energy",
            market_cap=450000.0, total_debt=42000.0, cash=30000.0, ebitda=55000.0,
            asset_volatility=0.28, scope1_emissions_t=95000000.0, scope2_emissions_t=18000000.0,
            scope3_emissions_t=540000000.0, carbon_pass_through_rate=0.40, scope3_liability_weight=0.15,
            physical_hazard_exposure=0.45, asset_replacement_cost=320000.0,
        ),
        CompanyClimateProfile(
            ticker="CVX", name="Chevron Corp", sector="Energy",
            market_cap=290000.0, total_debt=21000.0, cash=15000.0, ebitda=36000.0,
            asset_volatility=0.26, scope1_emissions_t=58000000.0, scope2_emissions_t=12000000.0,
            scope3_emissions_t=380000000.0, carbon_pass_through_rate=0.45, scope3_liability_weight=0.15,
            physical_hazard_exposure=0.40, asset_replacement_cost=210000.0,
        ),
        CompanyClimateProfile(
            ticker="NEE", name="NextEra Energy Inc", sector="Utilities",
            market_cap=155000.0, total_debt=72000.0, cash=2500.0, ebitda=12500.0,
            asset_volatility=0.18, scope1_emissions_t=38000000.0, scope2_emissions_t=2000000.0,
            scope3_emissions_t=15000000.0, carbon_pass_through_rate=0.85, scope3_liability_weight=0.05,
            physical_hazard_exposure=0.65, asset_replacement_cost=140000.0,
        ),
        CompanyClimateProfile(
            ticker="DUK", name="Duke Energy Corp", sector="Utilities",
            market_cap=82000.0, total_debt=78000.0, cash=1200.0, ebitda=10500.0,
            asset_volatility=0.17, scope1_emissions_t=74000000.0, scope2_emissions_t=1500000.0,
            scope3_emissions_t=22000000.0, carbon_pass_through_rate=0.80, scope3_liability_weight=0.05,
            physical_hazard_exposure=0.55, asset_replacement_cost=110000.0,
        ),
        CompanyClimateProfile(
            ticker="NUE", name="Nucor Corp", sector="Materials",
            market_cap=38000.0, total_debt=6800.0, cash=4500.0, ebitda=7200.0,
            asset_volatility=0.32, scope1_emissions_t=22000000.0, scope2_emissions_t=8500000.0,
            scope3_emissions_t=45000000.0, carbon_pass_through_rate=0.60, scope3_liability_weight=0.10,
            physical_hazard_exposure=0.25, asset_replacement_cost=30000.0,
        ),
        CompanyClimateProfile(
            ticker="LIN", name="Linde Plc", sector="Materials",
            market_cap=210000.0, total_debt=18500.0, cash=4800.0, ebitda=11800.0,
            asset_volatility=0.19, scope1_emissions_t=36000000.0, scope2_emissions_t=14000000.0,
            scope3_emissions_t=28000000.0, carbon_pass_through_rate=0.75, scope3_liability_weight=0.10,
            physical_hazard_exposure=0.20, asset_replacement_cost=95000.0,
        ),
        CompanyClimateProfile(
            ticker="UNP", name="Union Pacific Corp", sector="Industrials",
            market_cap=145000.0, total_debt=33000.0, cash=1800.0, ebitda=10200.0,
            asset_volatility=0.21, scope1_emissions_t=9800000.0, scope2_emissions_t=450000.0,
            scope3_emissions_t=6500000.0, carbon_pass_through_rate=0.70, scope3_liability_weight=0.10,
            physical_hazard_exposure=0.48, asset_replacement_cost=85000.0,
        ),
        CompanyClimateProfile(
            ticker="CAT", name="Caterpillar Inc", sector="Industrials",
            market_cap=180000.0, total_debt=37000.0, cash=7000.0, ebitda=14200.0,
            asset_volatility=0.24, scope1_emissions_t=1800000.0, scope2_emissions_t=1100000.0,
            scope3_emissions_t=85000000.0, carbon_pass_through_rate=0.65, scope3_liability_weight=0.20,
            physical_hazard_exposure=0.22, asset_replacement_cost=65000.0,
        ),
        CompanyClimateProfile(
            ticker="MSFT", name="Microsoft Corp", sector="Technology",
            market_cap=3100000.0, total_debt=75000.0, cash=110000.0, ebitda=125000.0,
            asset_volatility=0.22, scope1_emissions_t=150000.0, scope2_emissions_t=450000.0,
            scope3_emissions_t=16000000.0, carbon_pass_through_rate=0.90, scope3_liability_weight=0.05,
            physical_hazard_exposure=0.15, asset_replacement_cost=250000.0,
        ),
        CompanyClimateProfile(
            ticker="AAPL", name="Apple Inc", sector="Technology",
            market_cap=3400000.0, total_debt=105000.0, cash=65000.0, ebitda=132000.0,
            asset_volatility=0.23, scope1_emissions_t=55000.0, scope2_emissions_t=0.0,
            scope3_emissions_t=21000000.0, carbon_pass_through_rate=0.95, scope3_liability_weight=0.05,
            physical_hazard_exposure=0.12, asset_replacement_cost=180000.0,
        ),
    ]
    return profiles


def generate_disclosed_emissions_universe() -> pd.DataFrame:
    """Returns self-reported corporate disclosures."""
    data = [
        {"ticker": "XOM", "sector": "Energy", "reported_scope1_t": 95000000.0, "reported_scope2_t": 18000000.0},
        {"ticker": "CVX", "sector": "Energy", "reported_scope1_t": 58000000.0, "reported_scope2_t": 12000000.0},
        {"ticker": "OXY", "sector": "Energy", "reported_scope1_t": 28000000.0, "reported_scope2_t": 4000000.0},
        {"ticker": "EOG", "sector": "Energy", "reported_scope1_t": 11000000.0, "reported_scope2_t": 1200000.0},
        {"ticker": "KMI", "sector": "Energy", "reported_scope1_t": 14500000.0, "reported_scope2_t": 2100000.0},
        {"ticker": "NEE", "sector": "Utilities", "reported_scope1_t": 38000000.0, "reported_scope2_t": 2000000.0},
        {"ticker": "DUK", "sector": "Utilities", "reported_scope1_t": 74000000.0, "reported_scope2_t": 1500000.0},
        {"ticker": "SO",  "sector": "Utilities", "reported_scope1_t": 82000000.0, "reported_scope2_t": 1800000.0},
        {"ticker": "AEP", "sector": "Utilities", "reported_scope1_t": 61000000.0, "reported_scope2_t": 1400000.0},
        {"ticker": "NUE", "sector": "Materials", "reported_scope1_t": 22000000.0, "reported_scope2_t": 8500000.0},
        {"ticker": "LIN", "sector": "Materials", "reported_scope1_t": 36000000.0, "reported_scope2_t": 14000000.0},
        {"ticker": "DOW", "sector": "Materials", "reported_scope1_t": 31000000.0, "reported_scope2_t": 6800000.0},
        {"ticker": "UNP", "sector": "Industrials", "reported_scope1_t": 9800000.0, "reported_scope2_t": 450000.0},
        {"ticker": "CAT", "sector": "Industrials", "reported_scope1_t": 1800000.0, "reported_scope2_t": 1100000.0},
        {"ticker": "MSFT", "sector": "Technology", "reported_scope1_t": 150000.0, "reported_scope2_t": 450000.0},
        {"ticker": "AAPL", "sector": "Technology", "reported_scope1_t": 55000.0, "reported_scope2_t": 0.0},
    ]
    return pd.DataFrame(data)


def generate_satellite_plume_observations(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> List[SatelliteObservation]:
    """Generates realistic geospatial satellite plume detections across facilities."""
    rng = np.random.default_rng(seed)
    tickers_facilities = {
        "XOM": [("Permian Basin Pad 42", 31.95, -102.35, 1850.0), ("Baytown Refinery", 29.74, -95.01, 3200.0)],
        "CVX": [("Gorgon LNG Vent", -20.80, 115.40, 1200.0), ("Pascagoula Plant", 30.35, -88.50, 1400.0)],
        "OXY": [("Delaware Basin Gathering", 32.10, -103.50, 3100.0)],  # Heavy fugitive leaks
        "EOG": [("Eagle Ford Flare 12", 28.70, -98.20, 350.0)],         # Clean abater
        "KMI": [("Midland Compressor Station", 32.00, -102.10, 2400.0)],# High leak
        "NEE": [("FPL Clean Energy Center", 26.85, -80.10, 800.0)],      # Clean
        "DUK": [("Gibson Generating Station", 38.37, -87.75, 4500.0)],  # Coal heavy
        "SO":  [("Plant Scherer", 33.06, -83.80, 5200.0)],              # Coal heavy
        "AEP": [("Rockport Plant", 37.92, -87.03, 3900.0)],
        "NUE": [("Decatur EAF Steel Mill", 34.60, -87.00, 1100.0)],
        "LIN": [("Texas City Syngas", 29.38, -94.92, 1900.0)],
        "DOW": [("Freeport Petrochemical", 28.95, -95.35, 2200.0)],
        "UNP": [("Bailey Yard Logistics", 41.15, -100.75, 450.0)],
        "CAT": [("East Peoria Manufacturing", 40.67, -89.58, 120.0)],
        "MSFT": [("Quincy Datacenter Backup Gens", 47.23, -119.85, 15.0)],
        "AAPL": [("Mesa Renewable Datacenter", 33.30, -111.60, 5.0)],
    }

    dates = pd.date_range(start_date, end_date, freq="W-FRI")
    observations = []
    obs_id = 1

    for dt in dates:
        for ticker, fac_list in tickers_facilities.items():
            for fac_name, lat, lon, base_rate in fac_list:
                # Add stochastic flux variation
                noise = rng.normal(0, base_rate * 0.12)
                rate = max(1.0, base_rate + noise)
                gas = "CH4" if ticker in ["XOM", "CVX", "OXY", "EOG", "KMI"] else "CO2"
                conf = float(rng.uniform(0.90, 0.99))

                observations.append(
                    SatelliteObservation(
                        observation_id=f"PLUME_{obs_id:06d}",
                        ticker=ticker,
                        facility_name=fac_name,
                        latitude=lat,
                        longitude=lon,
                        gas_type=gas,
                        plume_rate_kg_hr=rate,
                        timestamp=dt,
                        confidence_score=conf,
                    )
                )
                obs_id += 1

    return observations


def generate_climate_equity_prices(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    alpha_signals: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Generates realistic daily price series with decarbonization lead-lag alpha response."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="B")
    n_days = len(dates)

    tickers = ["XOM", "CVX", "OXY", "EOG", "KMI", "NEE", "DUK", "SO", "AEP", "NUE", "LIN", "DOW", "UNP", "CAT", "MSFT", "AAPL"]
    
    # If explicit alpha signals provided, map to fundamental return drift
    if alpha_signals is not None:
        drift_dict = {t: 0.05 + 0.16 * alpha_signals.get(t, 0.0) for t in tickers}
    else:
        # Default drift aligning with satellite emissions surprise ranking
        drift_dict = {
            "XOM": +0.14, "CVX": +0.10, "DUK": +0.08, "SO": +0.09, "LIN": +0.06, "UNP": +0.05,
            "CAT": +0.04, "MSFT": +0.03, "AAPL": +0.03, "OXY": +0.00, "KMI": -0.05,
            "EOG": -0.06, "AEP": -0.07, "NUE": -0.09, "DOW": -0.10, "NEE": -0.12
        }

    prices = {}
    for t in tickers:
        ann_mu = drift_dict.get(t, 0.05) / 252.0
        ann_vol = 0.20 / np.sqrt(252.0)
        daily_rets = rng.normal(ann_mu, ann_vol, n_days)
        prices[t] = 100.0 * np.cumprod(1.0 + daily_rets)

    return pd.DataFrame(prices, index=dates)
