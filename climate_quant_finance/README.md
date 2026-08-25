# Climate Quantitative Finance & Carbon Markets (Projects 36-40)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-36%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"Climate finance is the fastest-growing frontier in quantitative modeling. Mastering carbon allowance pricing, green bond term structures, NGFS macro scenario stress testing, satellite emissions alpha, and renewable PPA risk transforms climate risks into quantitative alpha."*

A production-grade Python package implementing **Projects 36 through 40** of the comprehensive quantitative curriculum, covering carbon emissions trading economics, sustainable debt pricing, physical/transition climate stress testing, satellite alternative data alpha, and renewable energy volumetric derivatives.

---

## 📸 Overview: Projects 36-40

| Project | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **36. Carbon Allowance Pricing & ETS Dynamics** | Carbon emissions trading & power generation economics | Clean Spark Spread (Gas), Clean Dark Spread (Coal), Fuel-Switching Carbon Price Parity, Mean-Reverting Jump-Diffusion, Futures Cost-of-Carry |
| **37. Green Bond Valuation & Greenium Decomposition** | Sustainable fixed income & green premium analysis | Matched-Pair Twin Bond yield spread decomposition, Factor attribution regression (Liquidity, Credit Rating, ESG disclosure), Nelson-Siegel Greenium curve |
| **38. NGFS Climate Scenario Stress Testing Engine** | Physical & transition risk portfolio stress testing | NGFS Phase IV pathways (Net Zero 2050, Delayed Transition, Hot House World), Scope 1/2/3 Shadow Carbon Tax EBITDA deductions, Geospatial damage functions, Merton credit migration |
| **39. Geospatial & Satellite GHG Emissions Alpha** | Non-traditional satellite telemetry alpha modeling | Sentinel-5P / GHGSat methane/CO2 plume detection, Cross-sectional Emissions Surprise Z-scores, Dollar-neutral Long/Short ESG alpha strategy |
| **40. Renewable Energy PPA & Weather Derivatives** | Volumetric intermittency & energy risk pricing | Hourly Weibull wind & diurnal solar yield modeling, Merit-Order Duck Curve cannibalization capture price, Pay-As-Produced vs Baseload PPAs, HDD/CDD Weather Swaps |

---

## 🏛️ Repository Architecture

```text
climate_quant_finance/
├── data/                                 # Datasets (EUA carbon quotes, twin bonds, emissions, plumes, solar/wind)
├── output/                               # Rendered dark-theme quantitative artifacts
│   ├── 36_fuel_switching_parity.png     # Clean spark/dark spreads vs carbon price parity
│   ├── 37_greenium_term_structure.png    # Matched-pair green bond greenium curve
│   ├── 38_climate_var_stress.png         # Cross-sector NGFS Climate VaR % equity impairment
│   ├── 39_satellite_emissions_alpha.png  # Satellite emissions alpha cumulative wealth curve
│   ├── 40_ppa_duck_curve_cannibalization.png # Solar generation vs Duck Curve spot price suppression
│   └── climate_quant_infographic.png     # Full composite 5-module capstone dashboard
├── scripts/
│   └── run_all_demos.py                  # End-to-end demo execution suite
├── src/
│   └── climate_quant/
│       ├── carbon_pricing/               # Module 36: EU ETS & Carbon Allowance Pricing
│       │   └── eua_model.py              # Clean Spreads, Fuel-Switching Parity, Jump Diffusion, Futures
│       ├── climate_stress/               # Module 38: NGFS Climate Stress Testing Engine
│       │   └── ngfs_engine.py            # NGFS Scenarios, Transition/Physical risk, Climate VaR, Merton
│       ├── data/                         # Data layer & synthetic market simulator
│       │   └── loader.py                 # Multi-market carbon, green bonds, emissions, satellite & renewable loaders
│       ├── emissions_satellite/          # Module 39: Satellite Emissions Alpha Engine
│       │   └── plume_alpha.py            # Plume aggregation, Surprise z-scores, Strategy backtester
│       ├── green_bonds/                  # Module 37: Green Bond Valuation & Sustainability Premia
│       │   └── greenium.py               # Twin Bond Decomposition, Factor Attribution OLS, Nelson-Siegel
│       ├── renewable_energy/             # Module 40: Renewable PPA & Weather Derivatives
│       │   └── ppa_weather.py            # RenewablePPAValuator, WeatherDerivativePricer
│       └── visualization/                # Dark-theme quant visualization suite
│           └── plots.py                  # Infographic, greenium curve, Duck curve & stress renderers
├── tests/                                # 36 Unit tests covering all 5 modules
│   ├── test_carbon_pricing.py
│   ├── test_climate_stress.py
│   ├── test_emissions_satellite.py
│   ├── test_green_bonds.py
│   └── test_renewable_energy.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/climate-quant-finance.git
cd climate-quant-finance
pip install -r requirements.txt
```

### Run All Demos & Generate Visualizations

```bash
make demo
# or: PYTHONPATH=src python3 scripts/run_all_demos.py
```

### Run Full Test Suite

```bash
make test
# or: PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 🔬 Mathematical Formulations

### 36. Carbon Allowance Pricing & ETS Fuel-Switching
- **Clean Spark Spread (Gas) & Clean Dark Spread (Coal)**:
  $$\text{CSS} = P_{\text{power}} - \frac{P_{\text{gas}}}{\eta_{\text{gas}}} - \text{EF}_{\text{gas}} \cdot P_{\text{carbon}}$$
  $$\text{CDS} = P_{\text{power}} - \frac{P_{\text{coal}}}{\eta_{\text{coal}}} - \text{EF}_{\text{coal}} \cdot P_{\text{carbon}}$$
- **Theoretical Fuel-Switching Carbon Price Parity**:
  $$P_{\text{switch}} = \frac{\frac{P_{\text{gas}}}{\eta_{\text{gas}}} - \frac{P_{\text{coal}}}{\eta_{\text{coal}}}}{\text{EF}_{\text{coal}} - \text{EF}_{\text{gas}}}$$
- **Mean-Reverting Jump-Diffusion Spot SDE**:
  $$d\ln(S_t) = \kappa (\ln\theta - \ln S_t)dt + \sigma dW_t + J_t dN_t$$

---

### 37. Green Bond Valuation & Greenium Decomposition
- **Greenium Yield Spread**:
  $$\text{Greenium}_t = (y_{\text{vanilla}, t} - y_{\text{green}, t}) \times 100 \quad (\text{in bps})$$
- **Twin Bond Decomposition**:
  $$\text{Pure Greenium} = \text{Raw Spread} - \Delta y_{\text{maturity}} - \Delta y_{\text{duration}} + \Delta y_{\text{liquidity}}$$
- **Nelson-Siegel Parametric Term Structure**:
  $$\text{Greenium}(\tau) = \beta_0 + \beta_1 \left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}\right) + \beta_2 \left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda}\right)$$

---

### 38. NGFS Climate Scenario Stress Testing & Merton Default Migration
- **Transition Risk Shadow Carbon Tax**:
  $$\Delta \text{EBITDA}_i(t) = -\text{CarbonPrice}(t) \times (\text{Scope 1}_i + \text{Scope 2}_i + \alpha \text{Scope 3}_i) \times (1 - \beta_{\text{pass}, i})$$
- **Physical Asset Damage**:
  $$D_{\text{phys}, i}(t) = \text{AssetReplacementCost}_i \times \text{HazardExposure}_i \times \text{BaseDamageRate} \times \text{HazardMultiplier}(t)$$
- **Discounted Cash Flow Climate VaR**:
  $$\text{Climate VaR}_i = \frac{\text{Stressed Equity}_i - \text{Market Cap}_i}{\text{Market Cap}_i} \in [-100\%, 0\%]$$
- **Merton Structural Stressed Default Probability**:
  $$d_{2, \text{stress}} = \frac{\ln(V_{\text{stressed}} / D) + (r - \frac{1}{2}\sigma_V^2)T}{\sigma_V \sqrt{T}}, \quad PD_{\text{stressed}} = \Phi(-d_{2, \text{stress}})$$

---

### 39. Geospatial & Satellite GHG Emissions Alpha Model
- **Cross-Sectional Emissions Surprise Z-Score**:
  $$Z_{i, s} = \frac{(\text{Satellite Measured Emissions}_{i, s} - \text{Disclosed Scope 1}_{i, s}) - \mu_{\text{sector}}}{\sigma_{\text{sector}}}$$
- **Dollar-Neutral Alpha Strategy**:
  $$\text{Signal}_i = -\tanh(0.75 \times Z_{i, s}) \implies \text{Long Low-Plume Leaders, Short Heavy Fugitive Plumers}$$

---

### 40. Renewable Energy PPA Valuation & Weather Derivatives
- **Capture Price & Cannibalization Discount (Duck Curve)**:
  $$\text{Capture Price} = \frac{\sum_{t=1}^T P_t \times Q_t}{\sum_{t=1}^T Q_t}, \quad \text{Capture Rate} = \frac{\text{Capture Price}}{\bar{P}_{\text{baseload}}}$$
- **Degree Day Indices**:
  $$\text{HDD} = \sum_{t=1}^M \max(18^\circ\text{C} - \bar{T}_t, 0), \quad \text{CDD} = \sum_{t=1}^M \max(\bar{T}_t - 18^\circ\text{C}, 0)$$
- **Weather Swap & Option Payoffs**:
  $$\text{Swap Payoff} = \text{Tick} \times (I_M - K_{\text{swap}}), \quad \text{Option Payoff} = \min\left(\text{Cap}, \text{Tick} \times \max(I_M - K, 0)\right)$$

---

## 📊 Code Walkthrough

### 1. Carbon Allowance Fuel-Switching Parity
```python
from climate_quant.carbon_pricing import CarbonAllowanceModel

model = CarbonAllowanceModel(
    efficiency_gas=0.50,
    efficiency_coal=0.38,
    emission_factor_gas=0.37,
    emission_factor_coal=0.95,
)

snap = model.evaluate_spread_snapshot(
    power_price=105.0, gas_price=26.5, coal_price=13.5, carbon_price=75.0
)
print(f'Clean Spark Spread: €{snap.clean_spark_spread:+.2f}/MWh')
print(f'Fuel Switch Parity: €{snap.fuel_switch_parity_price:.2f}/tCO2')
print(f'Dominant Merit Fuel: {snap.dominant_fuel}')
```

### 2. Green Bond Greenium Valuation
```python
from climate_quant.data.loader import generate_green_bond_pairs
from climate_quant.green_bonds import GreenBondValuationEngine

engine = GreenBondValuationEngine()
pairs = generate_green_bond_pairs()

# Decompose greenium into fundamental vs liquidity components
decomp = engine.decompose_universe(pairs)
print(decomp[['Issuer', 'Raw Spread (bps)', 'Pure Fundamental Greenium (bps)']])
```

### 3. NGFS Climate Scenario Stress Testing
```python
from climate_quant.climate_stress import (
    NGFSClimateStressEngine,
    NGFSScenarioType,
)
from climate_quant.data.loader import generate_corporate_climate_universe

portfolio = generate_corporate_climate_universe()
engine = NGFSClimateStressEngine()

# Run multi-scenario stress comparison
report = engine.multi_scenario_comparison(portfolio, year=2030)
print(report.to_string(index=False))
```

### 4. Satellite GHG Emissions Alpha
```python
from climate_quant.data.loader import (
    generate_climate_equity_prices,
    generate_disclosed_emissions_universe,
    generate_satellite_plume_observations,
)
from climate_quant.emissions_satellite import SatelliteEmissionsAlpha

disclosed = generate_disclosed_emissions_universe()
plumes = generate_satellite_plume_observations()
prices = generate_climate_equity_prices()

alpha = SatelliteEmissionsAlpha()
satellite_aggr = alpha.aggregate_facility_plumes(plumes)
surprise = alpha.compute_emissions_surprises(disclosed, satellite_aggr)

# Backtest dollar-neutral strategy
bt = alpha.backtest_strategy(prices, {prices.index[0]: surprise})
print(bt.summary_table())
```

### 5. Renewable PPA & Weather Derivatives
```python
from climate_quant.renewable_energy import (
    PPAContract,
    PPAType,
    RenewableAssetConfig,
    RenewablePPAValuator,
)

config = RenewableAssetConfig(
    asset_type='solar', capacity_mw=100.0, latitude_deg=35.0
)
valuator = RenewablePPAValuator(config)

solar = valuator.simulate_hourly_solar_profile()
spot = valuator.simulate_electricity_spot_prices(solar)

# Value 10-year Pay-As-Produced PPA
pap = PPAContract(contract_type=PPAType.PAY_AS_PRODUCED, strike_price=52.0)
res = valuator.value_ppa_contract(solar, spot, pap)
print(res.summary_table())
```

---

## 🧪 Unit Testing

```bash
$ make test
....................................
----------------------------------------------------------------------
Ran 36 tests in 1.571s

OK
```

---

## 📜 References

1. **Carmona, R., Fehr, M., & Hinz, J. (2009)**. *Optimal design of emission trading schemes*. SIAM Review, 51(2), 263-294.
2. **Larcker, D. F., & Watts, E. M. (2020)**. *Where's the greenium?* Journal of Accounting and Economics, 69(2-3), 101312.
3. **NGFS (2023)**. *NGFS Climate Scenarios for central banks and supervisors (Phase IV)*. Network for Greening the Financial System.
4. **Cushnan, D., et al. (2022)**. *Methane plume detection from satellite imagery*. Nature Climate Change, 12, 450-456.
5. **Benth, F. E., Benth, J. Š., & Koekebakker, S. (2008)**. *Stochastic Modeling of Electricity and Related Markets*. World Scientific.
