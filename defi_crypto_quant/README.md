# DeFi, AMM Liquidity & Crypto Quantitative Finance (Projects 41-45)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"Decentralized Finance replaces institutional market makers with deterministic smart contracts. Quantitative mastery over Automated Market Maker (AMM) concentrated liquidity, Loss-Versus-Rebalancing (LVR), Maximal Extractable Value (MEV) atomic arbitrage, perpetual funding basis trading, and on-chain blockchain telemetry defines the bleeding edge of crypto alpha."*

A production-grade Python package implementing **Projects 41 through 45** of the comprehensive quantitative curriculum, covering CFMM liquidity mathematics, passive LP adverse selection, spatial/triangular MEV searchers, delta-neutral perpetual futures basis yield, and on-chain macro regime classification.

---

## 📸 Overview: Projects 41-45

| Project | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **41. CFMMs & Uniswap v3 Concentrated Liquidity** | Automated Market Maker liquidity & pricing math | Constant Product ($xy=k$), Virtual Reserves $(x + \frac{L}{\sqrt{P_b}})(y + L\sqrt{P_a}) = L^2$, Tick math ($P=1.0001^i$), Capital Efficiency ($40\times$), Curve Stableswap |
| **42. Impermanent Loss & Loss-Versus-Rebalancing (LVR)** | Passive LP adverse selection & profitability | Exact Impermanent Loss ($IL(k) = \frac{2\sqrt{k}}{1+k}-1$), Milionis-Moallemi-Roughgarden LVR ($\int \frac{\sigma^2}{8} S_t L_t dt$), Net LP Alpha, Breakeven Volatility |
| **43. Cross-DEX Flash Loans & MEV Searcher Engine** | Atomic spatial/triangular arbitrage & mempool MEV | Closed-form optimal flash loan sizing ($\Delta x^*$), Bellman-Ford negative-cycle triangular arb, 4-Stage Mempool Sandwich attack simulation, Flashbots builder bribe optimization |
| **44. Perpetual Funding Arbitrage & Basis Trading** | High-yield delta-neutral basis harvesting | 8-Hour funding rate clamping mechanism, Cash-and-carry basis trader, Staking yield overlay, Margin ratios & liquidation buffer stress testing |
| **45. On-Chain Blockchain Telemetry & Whale Alpha** | Alternative blockchain data & market cycle regimes | MVRV Ratio & Z-score, Net Exchange Flow Imbalance (EFI), Whale Accumulation Index ($\ge 1,000\,\text{BTC}$), NVT Signal, Multi-factor macro regime classifier |

---

## 🏛️ Repository Architecture

```text
defi_crypto_quant/
├── data/                                 # Synthetic order ticks, DEX pools, funding rates & on-chain series
├── output/                               # Rendered dark-theme quantitative artifacts
│   ├── 41_v3_concentrated_liquidity.png  # Uniswap v3 liquidity density vs v2 constant product
│   ├── 42_lvr_vs_fee_revenue.png         # Cumulative LVR adverse selection vs Fee revenue
│   ├── 43_mev_sandwich_dynamics.png      # 4-Stage price trajectory during atomic sandwich attack
│   ├── 44_perp_funding_basis_equity.png  # Delta-neutral basis trading cumulative equity curve
│   ├── 45_onchain_mvrv_regimes.png       # On-chain MVRV Z-score & cycle regime overlay
│   └── defi_crypto_quant_infographic.png # Full composite 5-module capstone dashboard
├── scripts/
│   └── run_all_demos.py                  # End-to-end demo execution suite
├── src/
│   └── defi_crypto_quant/
│       ├── data/                         # Data layer & synthetic crypto market generators
│       │   └── loader.py                 # Multi-asset spot, v3 ticks, funding rates, on-chain metrics
│       ├── lvr_impermanent_loss/         # Module 42: LVR & Impermanent Loss Engine
│       │   └── lvr_model.py              # ImpermanentLossCalculator, LossVersusRebalancingEngine
│       ├── mev_arbitrage/                # Module 43: Cross-DEX & MEV Searcher Engine
│       │   └── mev_engine.py             # Spatial Arbitrage, Triangular Searcher, Sandwich Simulator
│       ├── onchain_alpha/                # Module 45: On-Chain Blockchain Telemetry
│       │   └── onchain_model.py          # MVRV, Exchange Flow Imbalance, Whale Index, Macro Regimes
│       ├── perp_funding/                 # Module 44: Perpetual Futures & Basis Trading
│       │   └── basis_trading.py          # PerpetualFundingEngine, CashAndCarryBasisTrader
│       ├── uniswap_amm/                  # Module 41: CFMM & Concentrated Liquidity AMMs
│       │   └── amm_engine.py             # ConstantProductAMM (v2), ConcentratedLiquidityAMM (v3), Stableswap
│       └── visualization/                # Dark-theme quant visualization suite
│           └── plots.py                  # Infographic, liquidity density, LVR, sandwich & basis renderers
├── tests/                                # 38 Unit tests covering all 5 modules
│   ├── test_lvr_model.py
│   ├── test_mev_arbitrage.py
│   ├── test_onchain_alpha.py
│   ├── test_perp_funding.py
│   └── test_uniswap_amm.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/defi-crypto-quant.git
cd defi-crypto-quant
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

### 41. Constant Function Market Makers & Uniswap v3
- **Uniswap v2 Constant Product**: $x \cdot y = k \implies \Delta y = \frac{y \cdot \gamma \Delta x}{x + \gamma \Delta x}$.
- **Uniswap v3 Concentrated Liquidity**:
  $$(x + \frac{L}{\sqrt{P_b}})(y + L\sqrt{P_a}) = L^2$$
  where $P(i) = 1.0001^i$, $\sqrt{P(i)} = 1.0001^{i/2}$.
- **Capital Efficiency Multiplier**: $\text{Efficiency} = \frac{1}{1 - \sqrt{P_a / P_b}} \gg 1$.
- **Curve Stableswap Invariant**: $A n^n \sum x_i + D = A D n^n + \frac{D^{n+1}}{n^n \prod x_i}$.

---

### 42. Impermanent Loss & Loss-Versus-Rebalancing (LVR)
- **Standard Impermanent Loss**:
  $$IL(k) = \frac{2\sqrt{k}}{1 + k} - 1, \quad \text{where } k = \frac{P_t}{P_0}$$
- **Continuous-Time LVR (Milionis et al. 2022)**:
  $$\text{LVR}_t = \int_0^t \frac{\sigma^2}{8} S_u L_u du$$
- **Net LP Alpha**: $\text{Net LP P&L} = \text{Fee Revenue} - \text{LVR} - \text{Gas Overhead}$.

---

### 43. Cross-DEX Flash Loans & Atomic MEV Searcher
- **Closed-Form Optimal Flash Loan Sizing ($\Delta x^*$)**:
  $$\Delta x^* = \frac{\sqrt{x_1 x_2 y_1 y_2 \gamma_1 \gamma_2} - x_1 y_2}{\gamma_1 y_2 + \gamma_1 \gamma_2 y_1}$$
- **Triangular Arbitrage Cycle**: $\prod R_{u, v} > 1.0 \iff \sum -\ln(R_{u, v}) < 0$ (Bellman-Ford).
- **Mempool Sandwich Attack**: Frontrun swap pushes price to victim's max slippage bound $\rightarrow$ victim executes $\rightarrow$ searcher backrun dumps tokens for atomic risk-free profit.

---

### 44. Perpetual Futures Funding Arbitrage & Basis Trading
- **8-Hour Funding Rate Mechanism**:
  $$\text{Funding Rate} = \text{Clamp}\left( \text{Premium Index} + \text{Clamp}(\text{Interest Rate} - \text{Premium Index}, -0.05\%, +0.05\%), -0.75\%, +0.75\% \right)$$
- **Cash-and-Carry Delta-Neutral Strategy**: Long Spot + Short Perpetual Futures collecting 8-hour funding cashflows:
  $$\text{APY}_{\text{funding}} = \text{Mean(Funding Rate)} \times 3 \times 365$$

---

### 45. On-Chain Blockchain Telemetry & Whale Alpha
- **MVRV Ratio & Z-Score**:
  $$\text{MVRV} = \frac{\text{Market Cap}}{\text{Realized Cap}}, \quad Z_{\text{MVRV}} = \frac{\text{Market Cap} - \text{Realized Cap}}{\sigma(\text{Market Cap})}$$
- **Net Exchange Flow Imbalance (EFI)**: $\text{EFI}_t = \frac{\text{Inflows}_t - \text{Outflows}_t}{\text{Inflows}_t + \text{Outflows}_t} \in [-1.0, +1.0]$.
- **Whale Wallet Accumulation**: Tracks net supply absorption across tier addresses ($\ge 1,000\,\text{BTC}$).

---

## 📊 Code Walkthrough

### 1. Uniswap v3 Position Minting & Concentrated Swaps
```python
from defi_crypto_quant.uniswap_amm import ConcentratedLiquidityAMM

# Initialize v3 pool at $3,000 ETH/USDC (0.30% fee tier)
amm = ConcentratedLiquidityAMM(current_price=3000.0, fee_tier=0.0030)

# Mint concentrated position in $2,500-$3,500 range
pos = amm.mint_position(
    "WhaleLP",
    price_lower=2500.0,
    price_upper=3500.0,
    amount_x=10.0,
    amount_y=30000.0,
)
print(f"Liquidity L: {pos.liquidity:,.2f}")
print(
    f"Capital Efficiency: "
    f"{amm.capital_efficiency_multiplier(2500.0, 3500.0):.2f}x"
)

# Execute swap
swap = amm.swap(amount_in=2.0, token_in="ETH")
print(
    f"Received ${swap.amount_out:,.2f} USDC (Exec Price: ${swap.execution_price:.2f})"
)
```

### 2. Loss-Versus-Rebalancing (LVR) Simulation
```python
from defi_crypto_quant.data.loader import generate_crypto_market_data
from defi_crypto_quant.lvr_impermanent_loss import LossVersusRebalancingEngine

data = generate_crypto_market_data(n_days=180, freq="1h")
eth_prices = data[("ETH/USDC", "Close")]
eth_volumes = data[("ETH/USDC", "Volume")]

lvr_engine = LossVersusRebalancingEngine(pool_type="v3", fee_rate=0.0030)
sim = lvr_engine.simulate_lp_performance(
    eth_prices, eth_volumes, initial_capital_usd=100000.0
)
print(sim.summary_table)
```

### 3. Cross-DEX Flash Loan Arbitrage
```python
from defi_crypto_quant.mev_arbitrage import (
    CrossDEXArbitrageEngine,
    LiquidityPool,
    PoolType,
)

p_uni = LiquidityPool(
    "Uniswap_WETH_USDC",
    PoolType.UNISWAP_V2,
    "WETH",
    "USDC",
    2500.0,
    7500000.0,
    fee=0.0030,
)
p_sushi = LiquidityPool(
    "Sushiswap_WETH_USDC",
    PoolType.SUSHISWAP,
    "WETH",
    "USDC",
    1800.0,
    5580000.0,
    fee=0.0030,
)

engine = CrossDEXArbitrageEngine(eth_price_usd=3000.0)
arb = engine.evaluate_spatial_arbitrage(p_uni, p_sushi, token_borrow="USDC")
print(f"Optimal Flash Loan: ${arb.optimal_input:,.2f} USDC")
print(f"Net Arbitrage Profit: ${arb.net_profit:+,.2f}")
```

### 4. Perpetual Funding Rate Cash-and-Carry Basis Trading
```python
from defi_crypto_quant.data.loader import generate_synthetic_funding_rates
from defi_crypto_quant.perp_funding import CashAndCarryBasisTrader

funding_df = generate_synthetic_funding_rates(n_periods=1095)
trader = CashAndCarryBasisTrader(initial_capital_usd=1000000.0)
basis_res = trader.backtest(funding_df)
print(basis_res.summary_table())
```

### 5. On-Chain Blockchain Telemetry Alpha
```python
from defi_crypto_quant.onchain_alpha import OnChainAlphaEngine

engine = OnChainAlphaEngine()
data = engine.generate_synthetic_onchain_data(n_days=1500)
bt = engine.backtest_strategy(data, initial_capital=100000.0)
print(bt.summary_table())
```

---

## 🧪 Unit Testing

```bash
$ make test
......................................
----------------------------------------------------------------------
Ran 38 tests in 0.176s

OK
```

---

## 📜 References

1. **Milionis, J., Moallemi, C. C., Roughgarden, T., & Adams, A. (2022)**. *Automated Market Making and Loss-Versus-Rebalancing*. arXiv:2208.06046.
2. **Adams, H., Zinsmeister, N., Salem, M., Keefer, R., & Robinson, D. (2021)**. *Uniswap v3 Core Whitepaper*. Uniswap Labs.
3. **Egorov, M. (2019)**. *StableSwap - automated liquidity modeling for Stablecoins*. Curve Finance.
4. **Daian, P., et al. (2020)**. *Flash Boys 2.0: Frontrunning, Transaction Reordering, and Consensus Instability in Decentralized Exchanges*. IEEE S&P.
5. **Mahmudov, M., & Puell, D. (2018)**. *Bitcoin Market-Value-to-Realized-Value (MVRV) Ratio*. Woobull Charts.
