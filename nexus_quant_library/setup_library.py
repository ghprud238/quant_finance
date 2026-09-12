import os
import sys
import subprocess

base_dir = "/working_dir/nexus_quant_platform"
os.chdir(base_dir)

# 1. Ensure py.typed
py_typed_path = os.path.join(base_dir, "src", "nexus_quant", "py.typed")
with open(py_typed_path, "w") as f:
    f.write("# Marker file for PEP 561 static type checking\n")

# 2. Enhanced __init__.py with Facade API
init_content = '''"""
Nexus Quant (nexus-quant)
=========================
Enterprise-Grade Cross-Asset Quantitative Finance, DeFi, High-Frequency Microstructure, 
AI & Climate Risk Library.

Modules:
--------
- foundations: Volatility estimators (Yang-Zhang, Parkinson), Returns, Gaussian HMM Regimes.
- risk: Historical, Cornish-Fisher, Monte Carlo VaR & CVaR, Multi-Crisis Stress Testing, Climate VaR.
- strategies: Backtest Engine, Momentum, Mean Reversion, Online Kalman Filter Stat Arb.
- derivatives: Black-Scholes, Greeks, SVI Surface, Heston FFT/COS, Breeden-Litzenberger RND.
- macro: Nelson-Siegel Curves, Central Bank Hawk/Dove NLP, FX Carry Parity, Carbon ETS.
- microstructure: Level 2 LOB, Almgren-Chriss Execution, VPIN Flow Toxicity, Hawkes Processes, Avellaneda-Stoikov MM.
- ai: Financial Feature Engineering (FFD), GNN Supply Chain, SEC 10-K NLP, Fear & Greed Sentiment, Agentic Swarms.
- defi: Uniswap v3 Concentrated Liquidity, LVR Adverse Selection, Perpetual Funding Basis, Prediction Market Arbitrage.
- portfolio: Markowitz MPT, Wasserstein Distributionally Robust Optimization (DRO), Risk Parity (ERC).
- rigor: Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR), 5-Stage Research Validation Pipeline.
- orchestrator: Unified Multi-Asset Nexus Master Platform Engine.
"""

__version__ = "1.0.0"
__author__ = "Nexus Quant Core Engineering Group"
__license__ = "Apache-2.0"

# Top-level namespace aliases for institutional developer convenience
from nexus_quant import foundations as foundations
from nexus_quant import risk as risk
from nexus_quant import strategies as strategies
from nexus_quant import derivatives as derivatives
from nexus_quant import macro_fixed_income as macro
from nexus_quant import microstructure_execution as microstructure
from nexus_quant import ai_alternative_data as ai
from nexus_quant import defi_prediction_markets as defi
from nexus_quant import portfolio_optimization as portfolio
from nexus_quant import validation_rigor as rigor
from nexus_quant import orchestrator as orchestrator
from nexus_quant.core.data_engine import UnifiedDataEngine as DataEngine
from nexus_quant.orchestrator.master_engine import NexusMasterOrchestrator as Engine

__all__ = [
    "__version__",
    "foundations",
    "risk",
    "strategies",
    "derivatives",
    "macro",
    "microstructure",
    "ai",
    "defi",
    "portfolio",
    "rigor",
    "orchestrator",
    "DataEngine",
    "Engine",
]
'''

with open(os.path.join(base_dir, "src", "nexus_quant", "__init__.py"), "w") as f:
    f.write(init_content)

# 3. Comprehensive pyproject.toml
pyproject_content = '''[build-system]
requires = ["setuptools>=65.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "nexus-quant"
version = "1.0.0"
description = "Institutional-Grade Cross-Asset Quantitative Finance, DeFi, High-Frequency Microstructure, AI & Climate Risk Library"
readme = "README.md"
authors = [
    { name = "Nexus Quant Core Engineering Group", email = "quant-core@nexusquant.io" }
]
license = { text = "Apache-2.0" }
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Office/Business :: Financial :: Investment",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
dependencies = [
    "numpy>=1.23.0",
    "pandas>=1.5.0",
    "scipy>=1.9.0",
    "matplotlib>=3.6.0",
    "seaborn>=0.12.0",
]

[project.optional-dependencies]
all = [
    "scikit-learn>=1.2.0",
    "statsmodels>=0.13.0",
    "torch>=2.0.0",
    "tabulate>=0.9.0",
    "pydantic>=2.0.0",
]
hft = [
    "numba>=0.56.0",
]
defi = [
    "web3>=6.0.0",
]
dev = [
    "pytest>=7.2.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.0.260",
    "mypy>=1.0.0",
    "mkdocs>=1.4.0",
    "mkdocs-material>=9.0.0",
]

[project.urls]
Homepage = "https://github.com/nexus-quant/nexus-quant"
Documentation = "https://docs.nexusquant.io"
Repository = "https://github.com/nexus-quant/nexus-quant.git"
"Bug Tracker" = "https://github.com/nexus-quant/nexus-quant/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
nexus_quant = ["py.typed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 100
target-version = "py310"
'''

with open(os.path.join(base_dir, "pyproject.toml"), "w") as f:
    f.write(pyproject_content)

# 4. setup.py for backward compatibility
setup_py_content = '''from setuptools import setup, find_packages

setup(
    name="nexus-quant",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"nexus_quant": ["py.typed"]},
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ],
)
'''
with open(os.path.join(base_dir, "setup.py"), "w") as f:
    f.write(setup_py_content)

# 5. Create examples/ directory with institutional quickstart recipes
os.makedirs(os.path.join(base_dir, "examples"), exist_ok=True)

# Recipe 1: Hedge Funds & Multi-Asset Managers
recipe_1 = '''"""
Example 01: Hedge Fund & Multi-Asset Investment Manager Workflow
================================================================
Demonstrates:
- Unified multi-asset ingestion
- Factor alpha & Risk Parity (Equal Risk Contribution)
- Wasserstein Distributionally Robust Portfolio Optimization (DRO)
- Deflated Sharpe Ratio (DSR) multiple-testing statistical audit
"""

import numpy as np
import nexus_quant as nq

# 1. Ingest multi-asset market data
data_engine = nq.DataEngine(seed=42)
df_equities = data_engine.generate_equity_ohlcv(tickers=["AAPL", "MSFT", "GOOG", "AMZN", "XOM", "TLT", "SPY"])
closes = df_equities.xs("Close", level="Field", axis=1)
returns = nq.foundations.returns.simple_returns(closes).dropna()

print(f"Loaded {len(returns)} trading days across {len(closes.columns)} assets.")

# 2. Risk Parity Allocation (Equal Risk Contribution via Spinu log-barrier)
cov_matrix = np.cov(returns.values.T)
erc_weights = nq.portfolio.risk_parity.RiskParityOptimizer.optimize(cov_matrix)
print("\nEqual Risk Contribution (Risk Parity) Weights:")
for ticker, w in zip(closes.columns, erc_weights):
    print(f"  - {ticker}: {w:.2%}")

# 3. Wasserstein Distributionally Robust Optimization (DRO) under regime uncertainty
mu = returns.mean().values * 252
dro_opt = nq.portfolio.wasserstein_dro.WassersteinDROOptimizer(mu, cov_matrix * 252, risk_aversion=1.5)
dro_res = dro_opt.optimize(epsilon=0.015, norm_p=2)

print(f"\nWasserstein DRO Robust Expected Return: {dro_res.expected_return:.2%}")
print(f"Wasserstein DRO Worst-Case Loss Bound:  {dro_res.robust_objective:.4f}")
print(f"Effective Number of Assets (1/HHI):     {dro_res.effective_n_assets:.2f}")

# 4. Deflated Sharpe Ratio (DSR) Statistical Audit
dsr_calc = nq.rigor.deflated_sharpe.DeflatedSharpeRatioCalculator()
dsr_report = dsr_calc.compute_dsr(
    best_sharpe_ratio=1.85,
    num_trials=1000,
    sample_length=len(returns),
    skewness=float(nq.foundations.returns.stats.skew(returns.mean(axis=1))),
    kurtosis=float(nq.foundations.returns.stats.kurtosis(returns.mean(axis=1), fisher=False)),
)

print("\nDeflated Sharpe Ratio (DSR) Statistical Audit:")
print(f"  - Observed Backtest Sharpe: {dsr_report.observed_sharpe:.2f}")
print(f"  - Expected Max Sharpe under 1,000 noise trials: {dsr_report.expected_max_sharpe:.2f}")
print(f"  - Deflated Sharpe Ratio: {dsr_report.deflated_sharpe_ratio:.2%}")
print(f"  - Statistically Genuine Alpha (p < 0.05)? {dsr_report.is_significant_95}")
'''
with open(os.path.join(base_dir, "examples", "01_hedge_fund_multi_asset.py"), "w") as f:
    f.write(recipe_1)

# Recipe 2: HFT & Market Making Desk
recipe_2 = '''"""
Example 02: High-Frequency Trading & Market Making Desk Workflow
================================================================
Demonstrates:
- Continuous Level 2 Limit Order Book (L2 LOB) & Micro-Price
- Avellaneda-Stoikov (2008) inventory-skewed quoting
- Almgren-Chriss (2000) optimal execution trajectory
- Volume Synchronized Probability of Toxicity (VPIN) flow monitoring
"""

import nexus_quant as nq

# 1. Level 2 Limit Order Book & Micro-Price
lob = nq.microstructure.order_book.LimitOrderBook(name="HFT_LOB_BTC")
lob.add_limit_order(nq.microstructure.order_book.Order("B1", "buy", price=99.95, volume=1500.0, timestamp=1.0))
lob.add_limit_order(nq.microstructure.order_book.Order("A1", "sell", price=100.05, volume=800.0, timestamp=1.0))

print("L2 Limit Order Book State:")
print(f"  - Mid-Price:    ${lob.mid_price:.2f}")
print(f"  - Spread:       ${lob.spread:.2f}")
print(f"  - Micro-Price:  ${lob.micro_price:.4f}")
print(f"  - Book Imbalance: {lob.order_book_imbalance:+.2f}")

# 2. Avellaneda-Stoikov Inventory-Skewed Quoting Engine
mm = nq.microstructure.avellaneda_stoikov.AvellanedaStoikovMarketMaker(
    risk_aversion_gamma=0.10,
    order_book_liquidity_kappa=1.50,
)
quotes = mm.optimal_quotes(spot=100.0, inventory=5, current_time=0.5)
print("\nAvellaneda-Stoikov Quoting (with +5 Long Inventory):")
print(f"  - Reservation (Indifference) Price: ${quotes.reservation_price:.2f} (skewed downward to dump long inventory)")
print(f"  - Optimal Bid Quote: ${quotes.bid_price:.2f}")
print(f"  - Optimal Ask Quote: ${quotes.ask_price:.2f}")

# 3. Almgren-Chriss Optimal Order Slicing Trajectory
ac = nq.microstructure.optimal_execution.AlmgrenChrissModel(
    total_shares=1_000_000,
    horizon=1.0,
    n_intervals=20,
    volatility=0.30,
)
traj = ac.solve_trajectory(risk_aversion=1e-6)
print("\nAlmgren-Chriss Execution Trajectory:")
print(f"  - Expected Shortfall Cost: ${traj.expected_shortfall:,.2f}")
print(f"  - Trajectory Half-Life:    {traj.half_life:.2f} days")
print(f"  - Initial Interval Slicing: {traj.trade_sizes[0]:,.0f} shares")
'''
with open(os.path.join(base_dir, "examples", "02_hft_market_maker.py"), "w") as f:
    f.write(recipe_2)

# Recipe 3: Crypto, DeFi & Prediction Markets Desk
recipe_3 = '''"""
Example 03: Crypto, DeFi & Prediction Markets Desk Workflow
===========================================================
Demonstrates:
- Uniswap v3 concentrated liquidity virtual reserves & step-wise swaps
- Milionis-Moallemi-Roughgarden Loss-Versus-Rebalancing (LVR) adverse selection
- Crypto perpetual futures 8h funding rate basis trading (cash-and-carry)
- Polymarket vs Kalshi cross-venue prediction market arbitrage & Kelly staking
"""

import nexus_quant as nq

# 1. Uniswap v3 Concentrated Liquidity AMM
amm_v3 = nq.defi.uniswap_v3.ConcentratedLiquidityAMM(current_price=3000.0, fee_tier=0.0030)
pos = amm_v3.mint_position(
    owner="AlphaLP",
    price_lower=2500.0,
    price_upper=3500.0,
    amount_x=10.0,
    amount_y=30000.0,
)
print("Uniswap v3 Concentrated Liquidity Position:")
print(f"  - Virtual Liquidity L: {pos.liquidity:,.2f}")
print(f"  - Capital Efficiency:  {amm_v3.capital_efficiency_multiplier(2500.0, 3500.0):.2f}x vs v2 full-range")

# Execute swap
swap_res = amm_v3.swap(amount_in=2.0, token_in="ETH")
print(f"  - Swapped 2.0 ETH -> ${swap_res.amount_out:,.2f} USDC (Exec Price: ${swap_res.execution_price:.2f})")

# 2. Loss-Versus-Rebalancing (LVR)
lvr_engine = nq.defi.lvr_model.LossVersusRebalancingEngine(pool_type="v3", fee_rate=0.0030)
b_vol = lvr_engine.calculate_breakeven_volatility(annual_trading_volume_usd=500_000_000.0, pool_tvl_usd=50_000_000.0)
print(f"\nLVR Breakeven Volatility: {b_vol:.2%} (If asset vol < {b_vol:.2%}, LP is net profitable after LVR)")

# 3. Prediction Market Arbitrage & Kelly Staking
pm_engine = nq.defi.prediction_arbitrage.PredictionMarketArbitrageEngine()
book_poly = nq.defi.prediction_arbitrage.BinaryOrderBook(
    venue="Polymarket", contract_id="FED_RATE",
    yes_bids=[nq.defi.prediction_arbitrage.OrderBookLevel(0.44, 5000)],
    yes_asks=[nq.defi.prediction_arbitrage.OrderBookLevel(0.45, 5000)],
    no_bids=[nq.defi.prediction_arbitrage.OrderBookLevel(0.47, 5000)],
    no_asks=[nq.defi.prediction_arbitrage.OrderBookLevel(0.48, 5000)],
)
intra_arb = pm_engine.check_intra_venue_arbitrage(book_poly, target_size=2000.0)
if intra_arb and intra_arb.is_executable:
    print("\nPrediction Market Intra-Venue Arbitrage:")
    print(f"  - Combined Ask Price: ${intra_arb.effective_combined_price:.3f} (< $1.00 guaranteed payout)")
    print(f"  - Net Profit:         ${intra_arb.net_profit_usd:+,.2f} (ROI: {intra_arb.net_roi_pct:+.2f}%)")
    kelly = pm_engine.calculate_kelly_fraction(0.999, intra_arb.effective_combined_price, bankroll_usd=100000.0)
    print(f"  - Fractional Kelly Stake: ${kelly.recommended_stake_usd:,.2f} ({kelly.recommended_fraction:.1%} of bankroll)")
'''
with open(os.path.join(base_dir, "examples", "03_crypto_defi_arbitrage.py"), "w") as f:
    f.write(recipe_3)

# Recipe 4: Macro, Alt Data & Central Bank NLP Desk
recipe_4 = '''"""
Example 04: Global Macro, Alternative Data & Central Bank NLP Workflow
=====================================================================
Demonstrates:
- Multilingual Central Bank Hawk/Dove NLP & Taylor Rule Residualization
- Diebold-Yilmaz Sovereign Debt Volatility Spillovers & Contagion
- SEC 10-K Semantic Drift & "Lazy Prices" Alpha Factor
- Supply-Chain GNN Customer Spillover Momentum
"""

import numpy as np
import nexus_quant as nq

# 1. Central Bank Monetary Policy NLP
cb_indexer = nq.macro.central_bank_nlp.CentralBankStanceIndexer()
stmt = "Inflation pressures remain elevated above our 2.0% target. Further policy tightening is warranted."
stance = cb_indexer.score_statement(stmt, central_bank="FED", actual_policy_rate=5.25)
print("Central Bank Hawk/Dove Tone Index:")
print(f"  - Central Bank: {stance.central_bank}")
print(f"  - Stance:       {stance.stance_category}")
print(f"  - Score:        {stance.hawk_dove_score:+.2f}")
print(f"  - Taylor Gap:   {stance.taylor_rule_gap_bps:+.0f} bps")

# 2. SEC 10-K Semantic Drift & Lazy Prices Anomaly
drift_engine = nq.ai.sec_semantic_drift.SEC10KSemanticDriftEngine()
prev_10k = "The company faces standard competitive pressures and operational risks."
curr_10k_lazy = "The company faces standard competitive pressures and operational risks."
curr_10k_drift = "The company faces severe export embargoes, antitrust litigation, and supply-chain collapse."

d_lazy = drift_engine.compute_cosine_drift(prev_10k, curr_10k_lazy)
d_drift = drift_engine.compute_cosine_drift(prev_10k, curr_10k_drift)

print("\nSEC 10-K Semantic Drift & Lazy Prices:")
print(f"  - Lazy Filing Drift Score:  {d_lazy:.4f} (Recommendation: LONG)")
print(f"  - Changed Filing Drift Score: {d_drift:.4f} (Recommendation: SHORT)")

# 3. Nelson-Siegel Yield Curve Calibration
maturities = np.array([0.25, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
par_yields = np.array([5.20, 5.00, 4.70, 4.40, 4.20, 4.10, 4.15, 4.25, 4.50, 4.60])
ns = nq.macro.yield_curve.NelsonSiegelCalibrator.fit_nelson_siegel(maturities, par_yields)
print("\nNelson-Siegel Yield Curve Calibration:")
print(f"  - 10Y Model Spot Rate:    {ns.spot_rate(10.0):.2f}%")
print(f"  - 10Y Instantaneous Forward: {ns.forward_rate(10.0):.2f}%")
'''
with open(os.path.join(base_dir, "examples", "04_macro_alt_data_nlp.py"), "w") as f:
    f.write(recipe_4)

# Recipe 5: Climate Finance & Carbon Markets Desk
recipe_5 = '''"""
Example 05: Climate Quantitative Finance, Carbon Markets & ESG Desk Workflow
=============================================================================
Demonstrates:
- EU ETS Clean Spark/Dark Spread & Fuel-Switching Carbon Parity Price (P_switch)
- Green Bond Greenium Decomposition across matched-pair twin bonds
- NGFS Phase IV Climate Scenario DCF Climate VaR & Merton Credit Migration
- Renewable PPA Merit-Order Duck Curve & Cannibalization Analysis
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
print(f"  - Dominant Merit Order Fuel: {spreads.dominant_fuel}")

# 2. Green Bond Greenium Decomposition
green_engine = nq.macro.carbon_ets.GreenBondGreeniumEngine()
decomp = green_engine.decompose_greenium(
    yield_green=0.0385,
    yield_vanilla=0.0392,
    bid_ask_spread_green=0.0004,
    bid_ask_spread_vanilla=0.0002,
)
print("\nGreen Bond Greenium Decomposition:")
print(f"  - Raw Spread:       {decomp.raw_spread_bps:.1f} bps")
print(f"  - Pure Greenium:    {decomp.pure_greenium_bps:.1f} bps")
print(f"  - Liquidity Impact: {decomp.liquidity_spread_bps:.1f} bps")

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
print(f"  - Annual Transition Carbon Tax: ${c_var.annual_transition_cost_m:,.1f}M")
print(f"  - EBITDA Impairment:             {c_var.ebitda_impact_pct:.1%}")
print(f"  - DCF Climate VaR Equity Loss:   {c_var.climate_var_equity_pct:.1%}")
'''
with open(os.path.join(base_dir, "examples", "05_climate_carbon_esg.py"), "w") as f:
    f.write(recipe_5)

# 6. Create GitHub CI/CD Workflows
os.makedirs(os.path.join(base_dir, ".github", "workflows"), exist_ok=True)

ci_workflow = '''name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install .[all,dev]
    - name: Run Test Suite
      run: |
        python -m unittest discover -s tests -p "test_*.py"
    - name: Run Examples Verification
      run: |
        python examples/01_hedge_fund_multi_asset.py
        python examples/02_hft_market_maker.py
        python examples/03_crypto_defi_arbitrage.py
        python examples/04_macro_alt_data_nlp.py
        python examples/05_climate_carbon_esg.py
'''
with open(os.path.join(base_dir, ".github", "workflows", "ci.yml"), "w") as f:
    f.write(ci_workflow)

publish_workflow = '''name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    - name: Install build tools
      run: |
        python -m pip install --upgrade pip build twine
    - name: Build sdist and wheel
      run: |
        python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        twine upload dist/*
'''
with open(os.path.join(base_dir, ".github", "workflows", "publish.yml"), "w") as f:
    f.write(publish_workflow)

# 7. Documentation configuration (mkdocs.yml)
mkdocs_content = '''site_name: Nexus Quant Platform Documentation
site_description: Enterprise-Grade Cross-Asset Quantitative Finance, DeFi, HFT Microstructure, AI & Climate Risk Library
site_url: https://docs.nexusquant.io
theme:
  name: material
  palette:
    scheme: slate
    primary: indigo
    accent: amber
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy

nav:
  - Overview: index.md
  - Architecture: architecture.md
  - Modules:
    - 01. Foundations & Volatility: modules/foundations.md
    - 02. Risk & Climate VaR: modules/risk.md
    - 03. Systematic Strategies: modules/strategies.md
    - 04. Derivatives & RND: modules/derivatives.md
    - 05. Macro & Fixed Income: modules/macro.md
    - 06. Microstructure & HFT: modules/microstructure.md
    - 07. AI & Alternative Data: modules/ai.md
    - 08. DeFi & Prediction Markets: modules/defi.md
    - 09. Portfolio Optimization: modules/portfolio.md
    - 10. Statistical Rigor: modules/rigor.md
    - 11. Master Orchestrator: modules/orchestrator.md
  - Quickstart Recipes:
    - Hedge Funds: recipes/hedge_fund.md
    - HFT & Market Making: recipes/hft.md
    - DeFi & Crypto: recipes/defi.md
    - Macro & ESG: recipes/macro_esg.md
'''
with open(os.path.join(base_dir, "mkdocs.yml"), "w") as f:
    f.write(mkdocs_content)

print("[+] Library package configuration, examples, and CI/CD setup completed.")
