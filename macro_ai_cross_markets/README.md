# Global Macro AI, Crypto & Cross-Economy Sentiment (Projects 46-50)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"Modern global macro quantitative research requires unifying multilingual Central Bank NLP, cross-economy sovereign risk contagion spillovers, multi-source retail and institutional sentiment, cross-currency carry dynamics, and autonomous multi-agent LLM investment committees."*

A production-grade Python package implementing **Projects 46 through 50** of the comprehensive quantitative curriculum, completing a landmark **50-project quantitative finance master track** spanning developed and emerging markets, crypto on-chain telemetry, and agentic generative AI.

---

## 📸 Overview: Projects 46-50

| Project | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **46. Multilingual Central Bank Hawk/Dove NLP** | Monetary policy stance & sovereign yield impact | Sublinear TF-IDF & domain tone scoring across Fed, ECB, BOJ, RBI, BCB, Banxico, PBOC; Taylor Rule gap residualization; 2Y sovereign yield predictive regressions |
| **47. Sovereign Contagion & Volatility Spillovers** | Cross-border sovereign risk & debt connectedness | Vector Autoregression (VAR), Diebold-Yilmaz (2012) Generalized Forecast Error Variance Decomposition (GFEVD) Spillover Matrix, Clayton/Gumbel Copula lower tail dependence |
| **48. Multi-Source News & Crypto Sentiment Engine** | Real-time narrative momentum & market psychology | Aspect-Based Sentiment Analysis (ABSA) on news wires & social media, Reconstructed 6-component Crypto Fear & Greed Index, Lead-lag return cross-correlation |
| **49. Cross-Economy FX Carry Trade & Vol Surface** | Currency carry harvesting & interest rate parity | Covered (CIP) & Uncovered (UIP) Interest Rate Parity, Fama Forward Rate Bias anomaly, Long High-Yield EM / Short G10 Carry, Malz (1997) 25-delta Risk Reversal & Butterfly surface |
| **50. Autonomous Multi-Agent Hedge Fund Swarm** | Collaborative multi-agent quantitative portfolio committee | Specialized LLM Personas (Macro Economist, Crypto Specialist, Sentiment Analyst, PM Chair), Black-Litterman subjective view blending, Constrained quadratic risk optimization |

---

## 🏛️ Repository Architecture

```text
macro_ai_cross_markets/
├── data/                                 # Central bank transcripts, yields, FX quotes & news text
├── output/                               # Rendered dark-theme quantitative artifacts
│   ├── 46_central_bank_hawk_dove.png     # Hawk/Dove monetary policy stance index timeline
│   ├── 47_sovereign_spillover_matrix.png # Diebold-Yilmaz sovereign risk spillover heatmap
│   ├── 48_crypto_fear_greed_timeline.png # Fear & Greed index vs BTC/USD price action
│   ├── 49_fx_carry_equity_curve.png      # Cross-economy EM vs DM FX carry equity compounding
│   ├── 50_agentic_swarm_allocations.png  # Multi-agent committee consensus & asset weights
│   └── macro_ai_cross_markets_infographic.png # Full composite 5-module capstone dashboard
├── scripts/
│   └── run_all_demos.py                  # End-to-end demo execution suite
├── src/
│   └── macro_ai_cross_markets/
│       ├── agentic_hedge_fund/           # Module 50: Autonomous Multi-Agent Swarm
│       │   └── hedge_fund_swarm.py       # Macro/Crypto/Sentiment Personas, PM Black-Litterman Chair
│       ├── central_bank_nlp/             # Module 46: Central Bank Hawk/Dove NLP
│       │   └── hawk_dove.py              # Multilingual Tone Scoring, Taylor Rule Residualization
│       ├── crypto_sentiment/             # Module 48: Multi-Source Sentiment Engine
│       │   └── sentiment_engine.py       # ABSA, Fear & Greed Reconstruction, Lead-Lag Correlations
│       ├── data/                         # Data layer & market simulators
│       │   └── loader.py                 # Central bank statements, CDS spreads, FX quotes, social streams
│       ├── fx_carry_parity/              # Module 49: FX Carry & Parity Engine
│       │   └── carry_engine.py           # CIP/UIP Basis, Forward Bias, EM Carry, Malz Vol Surface
│       ├── sovereign_contagion/          # Module 47: Sovereign Risk & Contagion
│       │   └── spillover.py              # VAR GFEVD Spillovers, Clayton/Gumbel Copula Tail Dependence
│       └── visualization/                # Dark-theme quant visualization suite
│           └── plots.py                  # Infographic, stance timeline, heatmap, carry & swarm renderers
├── tests/                                # 28 Unit tests covering all 5 modules
│   ├── test_agentic_hedge_fund.py
│   ├── test_central_bank_nlp.py
│   ├── test_crypto_sentiment.py
│   ├── test_fx_carry_parity.py
│   └── test_sovereign_contagion.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/macro-ai-cross-markets.git
cd macro-ai-cross-markets
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

### 46. Multilingual Central Bank Hawk/Dove NLP
- **Hawk-Dove Tone Stance Index**:
  $$H_t = \frac{N_{\text{hawkish}} - N_{\text{dovish}}}{N_{\text{hawkish}} + N_{\text{dovish}} + \epsilon} \in [-1.0, +1.0]$$
- **Taylor Rule Gap Residualization**:
  $$i_t^* = r^* + \pi_t + 0.5(\pi_t - \pi^*) + 0.5(y_t - y^*)$$
  $$\Delta i_t = \alpha + \beta_1 (i_t - i_t^*) + \gamma H_t + \epsilon_t$$

---

### 47. Sovereign Contagion & Diebold-Yilmaz Volatility Spillovers
- **Vector Autoregression (VAR)** on $N$ sovereign debt spreads:
  $$Y_t = \sum_{p=1}^P \Phi_p Y_{t-p} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma)$$
- **Generalized Forecast Error Variance Decomposition (GFEVD)**:
  $$\theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H-1} (e_i' A_h \Sigma e_j)^2}{\sum_{h=0}^{H-1} (e_i' A_h \Sigma A_h' e_i)}, \quad \tilde{\theta}_{ij}(H) = \frac{\theta_{ij}(H)}{\sum_{j=1}^N \theta_{ij}(H)}$$
- **Total Spillover Index**: $S(H) = \frac{\sum_{i \ne j} \tilde{\theta}_{ij}(H)}{N} \times 100\%$.
- **Clayton Copula Lower Tail Dependence**:
  $$C(u, v) = \left( u^{-\theta} + v^{-\theta} - 1 \right)^{-1/\theta} \implies \lambda_L = 2^{-1/\theta}$$

---

### 48. Multi-Source News & Crypto Fear/Greed Sentiment
- **Reconstructed 6-Component Fear & Greed Formula**:
  $$\text{FGI} = 0.25 \cdot S_{\text{vol}} + 0.25 \cdot S_{\text{mom}} + 0.15 \cdot S_{\text{social}} + 0.10 \cdot S_{\text{dom}} + 0.10 \cdot S_{\text{search}} + 0.15 \cdot S_{\text{whale}}$$
- **Lead-Lag Cross-Correlation**:
  $$\rho_k = \text{Corr}\left(\text{FGI}_t, R_{t+k}\right), \quad k \in [-10, +10] \text{ days}$$

---

### 49. Cross-Economy FX Carry Trade & Volatility Surface
- **Covered (CIP) & Uncovered (UIP) Interest Rate Parity**:
  $$F_{t, T} = S_t \frac{1 + r_d T}{1 + r_f T}, \quad \text{CIP Basis (bps)} = \left( \frac{F_{t, T}}{S_t} (1 + r_f T) - (1 + r_d T) \right) \times 10^4$$
- **Malz (1997) 25-Delta Volatility Surface Smile**:
  $$\sigma(\Delta) = \sigma_{\text{ATM}} - 2 \cdot RR_{25} \cdot (\Delta - 0.5) + 16 \cdot BF_{25} \cdot (\Delta - 0.5)^2$$

---

### 50. Autonomous Multi-Agent Macro/Crypto Hedge Fund Swarm
- **Specialized Committee Personas**: Macro Economist Agent $\rightarrow$ Crypto Microstructure Agent $\rightarrow$ Sentiment Alpha Agent.
- **Black-Litterman Committee View Blending**:
  $$\mathbb{E}[R] = \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1} \left[ (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q \right]$$
- **Constrained Portfolio Optimization**: Solves for weights $w^*$ maximizing Sharpe ratio subject to $\sum w_i = 1$, $w_i \ge 0$, and $w_i \le w_{\text{max}, i}$.

---

## 📊 Code Walkthrough

### 1. Central Bank Hawk/Dove Tone Indexing
```python
from macro_ai_cross_markets.central_bank_nlp import CentralBankStanceIndexer
from macro_ai_cross_markets.data.loader import generate_central_bank_statements

statements = generate_central_bank_statements()
indexer = CentralBankStanceIndexer()

results = indexer.analyze_corpus(statements)
print(results[['Central_Bank', 'Date', 'Hawk_Dove_Score', 'Stance', 'Top_Topic']])
```

### 2. Diebold-Yilmaz Sovereign Risk Spillovers
```python
from macro_ai_cross_markets.data.loader import generate_macro_market_data
from macro_ai_cross_markets.sovereign_contagion import SovereignContagionEngine

macro = generate_macro_market_data()
engine = SovereignContagionEngine(var_lags=2, forecast_horizon=10)

spillover = engine.compute_diebold_yilmaz_spillovers(macro['cds_spreads'])
print(f'Total Spillover Index: {spillover.total_spillover_index:.2f}%')
print(f'Net Transmitters: {spillover.net_transmitters[:3]}')
print(spillover.spillover_matrix)
```

### 3. Reconstructing the Crypto Fear & Greed Index
```python
from macro_ai_cross_markets.crypto_sentiment import MultiSourceSentimentEngine

engine = MultiSourceSentimentEngine()
fgi_df = engine.compute_fear_greed_index(
    volatility_series=btc_vols,
    price_series=btc_prices,
    volume_series=btc_volumes,
)
print(fgi_df[['Composite_FGI', 'Regime', 'Volatility_Score', 'Social_Score']])
```

### 4. Cross-Economy FX Carry Trade Backtest
```python
from macro_ai_cross_markets.fx_carry_parity import FXCarryParityEngine

engine = FXCarryParityEngine()
carry_res = engine.backtest_fx_carry_strategy(
    fx_spot_df=fx_spots,
    interest_rates_df=fx_rates,
    funding_currencies=['USD', 'EUR', 'JPY'],
    target_currencies=['BRL', 'MXN', 'ZAR', 'INR'],
)
print(carry_res.metrics_table)
```

### 5. Autonomous Multi-Agent Hedge Fund Committee
```python
from macro_ai_cross_markets.agentic_hedge_fund import (
    CryptoData,
    MacroData,
    MultiAgentHedgeFundSwarm,
    SentimentData,
)

swarm = MultiAgentHedgeFundSwarm(target_vol_annual=0.12)
memo = swarm.conduct_investment_committee(
    macro_data=MacroData(
        gdp_growth_pct=2.4, cpi_inflation_pct=2.8, central_bank_rate_pct=4.75
    ),
    crypto_data=CryptoData(
        btc_price=64000.0, eth_price=3400.0, mvrv_z_score=1.20
    ),
    sentiment_data=SentimentData(
        fear_and_greed_index=62.0, news_sentiment_score=0.35
    ),
    date='2026-08-25',
)

print(memo.to_markdown())
```

---

## 🧪 Unit Testing

```bash
$ make test
............................
----------------------------------------------------------------------
Ran 28 tests in 0.681s

OK
```

---

## 📜 References

1. **Diebold, F. X., & Yilmaz, K. (2012)**. *Better to give than to receive: Predictive directional measurement of volatility spillovers*. International Journal of Forecasting, 28(1), 57-71.
2. **Malz, A. M. (1997)**. *Estimating the probability distribution of the future exchange rate from option prices*. The Journal of Derivatives, 5(2), 18-36.
3. **Fama, E. F. (1984)**. *Forward and spot exchange rates*. Journal of Monetary Economics, 14(3), 319-338.
4. **Black, F., & Litterman, R. (1992)**. *Global portfolio optimization*. Financial Analysts Journal, 48(5), 28-43.
5. **Nyman, R., et al. (2021)**. *News and narratives in financial systems: Exploiting big data for systemic risk assessment*. Bank of England Staff Working Paper No. 703.
