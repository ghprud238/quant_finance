# Frontier Quantitative AI, Advanced Math & Alternative Data (Projects 31-35)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-36%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"At the frontier of quantitative finance, alpha lies at the intersection of Generative AI, Stochastic Calculus, Market Microstructure Toxicity, Graph Neural Networks, and Distributionally Robust Optimization."*

A production-grade Python package implementing **Projects 31 through 35**, extending the curriculum into cutting-edge domain areas: Financial NLP semantic drift, Heston stochastic volatility Fourier inversion, order flow toxicity (VPIN), supply-chain network momentum, and Wasserstein optimal transport portfolio optimization.

---

## 📸 Overview: Projects 31-35

| Project | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **31. Financial LLM & SEC 10-K Semantic Drift** | NLP-driven alpha from corporate disclosure shifts | Sublinear TF-IDF embeddings, Cosine & Jaccard dissimilarity, Loughran-McDonald financial sentiment, Cohen-Malloy-Nguyen 'Lazy Prices' anomaly |
| **32. Heston Stochastic Volatility & FFT Calibration** | Continuous-time derivatives pricing & surface inversion | Heston SDE, Albrecher stable characteristic function, Carr-Madan Fast Fourier Transform (FFT), Fang-Oosterlee COS method, Feller condition ($2\kappa\theta > \xi^2$) |
| **33. Volume Synchronized Probability of Toxicity (VPIN)** | High-frequency flow toxicity & adverse selection | Information-time Volume Clock, Bulk Volume Classification (BVC), Rolling VPIN imbalance index, Flash-Crash early warnings |
| **34. Supply-Chain Knowledge Graph & GNN Alpha** | Economic network spillover & lead-lag momentum | Directed customer-supplier graph, Graph Convolutional Network (GCN) message-passing, Customer earnings surprise propagation, Network PageRank |
| **35. Wasserstein Distributionally Robust Optimization (DRO)** | Non-stationary distribution shift protection | Optimal transport ambiguity ball ($W_1$), Kuhn-Esfahani convex dual reformulation, $L_2 / L_1$ norm shrinkage, Markowitz error-maximization mitigation |

---

## 🏛️ Repository Architecture

```text
genai_advanced_quant/
├── data/                                 # Multi-year SEC text, Option surfaces, Tick trades, Network graphs
│   ├── sample_test_returns.csv           # Stressed regime return matrix
│   ├── sample_train_returns.csv          # In-sample return matrix
│   ├── supply_chain_prices.csv           # 22-node customer-supplier daily prices (2018-2024)
│   └── vpin_trades.csv                   # High-frequency tick trades with flash-crash shock
├── output/                               # Rendered dark-theme quantitative artifacts
│   ├── 31_sec_semantic_drift.png        # Cross-sectional 10-K cosine drift rankings
│   ├── 32_heston_volatility_surface_3d.png # 3D Calibrated Heston implied volatility surface
│   ├── 33_vpin_toxicity_timeline.png     # Spot price & VPIN order flow toxicity alerts
│   ├── 34_supply_chain_gnn_alpha.png     # Supply-chain GNN alpha vs Standalone Momentum
│   ├── 35_robust_vs_nominal_frontier.png # Nominal Markowitz vs Wasserstein Robust DRO Frontier
│   └── genai_advanced_quant_infographic.png # Full composite 5-module capstone dashboard
├── scripts/
│   └── run_all_demos.py                  # End-to-end demo execution suite
├── src/
│   └── genai_advanced_quant/
│       ├── data/                         # Synthetic multi-asset, option & text loaders
│       │   └── loader.py                 # SEC filings, option surfaces, tick flows, graph links
│       ├── graph_alpha/                  # Module 34: Supply-Chain Knowledge Graph & GNN
│       │   └── supply_chain.py           # Network adjacency, GCN message passing, Lead-Lag Alpha
│       ├── heston_fft/                   # Module 32: Heston Stochastic Volatility & FFT
│       │   └── model.py                  # Heston SDE, Characteristic function, Carr-Madan, COS, Calibration
│       ├── llm_alpha/                    # Module 31: Financial LLM & Semantic Drift
│       │   └── semantic_drift.py         # TF-IDF, Cosine drift, Loughran-McDonald lexicon, Lazy Prices
│       ├── robust_dro/                   # Module 35: Wasserstein DRO Portfolio Optimizer
│       │   └── dro_optimizer.py          # Optimal transport ambiguity ball, Convex dual, Robust Frontier
│       ├── visualization/                # Dark-theme quant visualization suite
│       │   └── plots.py                  # 3D surface, VPIN timeline, Network equity & Infographic
│       └── vpin_microstructure/          # Module 33: VPIN Microstructure Engine
│           └── vpin.py                   # Volume Clock, Bulk Volume Classification, Toxicity Alerts
├── tests/                                # 36 Unit tests covering all 5 modules
│   ├── test_graph_alpha.py
│   ├── test_heston_fft.py
│   ├── test_llm_alpha.py
│   ├── test_robust_dro.py
│   └── test_vpin.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/genai-advanced-quant.git
cd genai-advanced-quant
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

### 31. Financial LLM & SEC 10-K Semantic Drift
- **Sublinear TF-IDF Weighting**:
  $$\text{TF}(t, d) = 1 + \ln(\text{count}(t, d)), \quad \text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1, \quad \mathbf{v}_d = \frac{\mathbf{w}_d}{\|\mathbf{w}_d\|_2}$$
- **Cosine Dissimilarity / Semantic Drift**:
  $$\text{Drift}_t = 1 - \frac{\mathbf{v}_t \cdot \mathbf{v}_{t-1}}{\|\mathbf{v}_t\|_2 \|\mathbf{v}_{t-1}\|_2} \in [0, 2]$$
- **Loughran-McDonald Financial Sentiment**:
  $$\text{Sentiment} = \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{pos}} + N_{\text{neg}} + \epsilon} \in [-1, +1]$$
- **Lazy Prices Anomaly (Cohen, Malloy, Nguyen 2020)**:
  Sorts stocks into Low-Drift (verbatim/lazy disclosures $\rightarrow$ Long) vs. High-Drift (narrative revisions $\rightarrow$ Short).

---

### 32. Heston Stochastic Volatility & Carr-Madan FFT / COS Pricing
- **Heston (1993) SDE**:
  $$dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^S$$
  $$dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v, \quad \text{with } \mathbb{E}[dW_t^S dW_t^v] = \rho dt$$
  - **Feller Condition**: $2\kappa\theta > \xi^2$ guarantees variance remains strictly positive.
- **Stable Characteristic Function (Albrecher et al. 2007)**:
  $$\phi(u) = \exp\left( C(u, T) + D(u, T) v_0 + i u \ln(S_0) \right)$$
- **Carr-Madan (1999) Fast Fourier Transform (FFT)**:
  $$C(K) = \frac{e^{-\alpha \ln K}}{\pi} \int_0^\infty e^{-i v \ln K} \frac{e^{-rT} \phi(v - (\alpha + 1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1)v} dv$$
- **Fang-Oosterlee (2008) COS Method**: Sub-millisecond series expansion using orthogonal cosine basis functions.

---

### 33. Volume Synchronized Probability of Toxicity (VPIN)
- **Volume Clock Discretization**:
  $$V = \frac{\text{ADV}}{N_{\text{buckets}}}$$
- **Bulk Volume Classification (BVC)**:
  $$V_\tau^B = V \cdot \Phi\left(\frac{P_\tau - P_{\tau-1}}{\sigma_{\Delta P}}\right), \quad V_\tau^S = V - V_\tau^B$$
- **VPIN Flow Toxicity Index**:
  $$\text{VPIN} = \frac{\sum_{\tau=1}^N |V_\tau^B - V_\tau^S|}{N \cdot V} = \frac{1}{N} \sum_{\tau=1}^N \left| 2 \Phi\left(\frac{\Delta P_\tau}{\sigma_{\Delta P}}\right) - 1 \right|$$

---

### 34. Supply-Chain Knowledge Graph & GNN Alpha
- **Economic Directed Adjacency Matrix**: $A_{i,j}$ = % revenue supplier $i$ derives from customer $j$.
- **Graph Convolutional Network (GCN) Message Passing**:
  $$H^{(l+1)} = \text{ReLU}\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$
- **Customer-to-Supplier Spillover Momentum**:
  $$\text{CustomerMomentum}_{i, t} = \sum_{j} A_{i,j} \cdot R_{j, t-W:t-1}$$

---

### 35. Wasserstein Distributionally Robust Optimization (DRO)
- **Worst-Case Optimal Transport Problem**:
  $$\min_{w \in \mathcal{W}} \max_{\mathbb{Q}: W_1(\mathbb{Q}, \hat{\mathbb{P}}_N) \le \epsilon} \mathbb{E}_{\mathbb{Q}} \left[ -w^T \xi + \frac{\gamma}{2} (w^T \xi - w^T \hat{\mu})^2 \right]$$
- **Exact Convex Dual Reformulation**:
  $$\min_{w \in \mathcal{W}} \left( -w^T \hat{\mu} + \frac{\gamma}{2} w^T \hat{\Sigma} w + \epsilon \|w\|_p \right) \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0$$
  Shrinks extreme portfolio weights towards robust diversification, controlling out-of-sample degradation.

---

## 📊 Code Walkthrough

### 1. Financial LLM Semantic Drift
```python
from genai_advanced_quant.data.loader import generate_synthetic_sec_filings
from genai_advanced_quant.llm_alpha import (
    LazyPricesStrategy,
    SemanticDriftEngine,
)

filings = generate_synthetic_sec_filings()
engine = SemanticDriftEngine()

# Analyze year-over-year textual drift
drift_df = engine.analyze_universe(filings, target_year=2023)
print(drift_df[['Ticker', 'Cosine_Drift_Total', 'Sentiment_Score', 'Category']])

# Dollar-neutral allocation
strat = LazyPricesStrategy(quantile_cutoff=0.30)
positions = strat.generate_positions(drift_df)
print(positions[['Ticker', 'Weight', 'Recommendation']])
```

### 2. Heston Stochastic Volatility Calibration & Fast Pricing
```python
from genai_advanced_quant.data.loader import generate_market_option_surface
from genai_advanced_quant.heston_fft import (
    HestonOptionPricer,
    carr_madan_fft_price,
    fang_oosterlee_cos_price,
)

mkt_surface = generate_market_option_surface()
pricer = HestonOptionPricer()

# Calibrate Heston model
calib_res = pricer.calibrate(mkt_surface, spot=100.0, r=0.05, q=0.01)
print(
    f'Calibrated v0: {calib_res.calibrated_params.v0:.4f}, kappa:'
    f' {calib_res.calibrated_params.kappa:.2f}'
)
print(f'Calibration RMSE: ${calib_res.rmse:.4f}')

# Fast sub-millisecond COS method pricing
cos_price = fang_oosterlee_cos_price(
    100.0, 100.0, 1.0, calib_res.calibrated_params
)
print(f'ATM Call Price (COS): ${cos_price:.4f}')
```

### 3. VPIN Order Flow Toxicity Monitoring
```python
from genai_advanced_quant.data.loader import load_vpin_sample_data
from genai_advanced_quant.vpin_microstructure import VPINEngine

ticks_df = load_vpin_sample_data()
vpin_engine = VPINEngine(n_buckets=50)
res = vpin_engine.compute_vpin(ticks_df)

print(res.summary_table())
print(f'Peak Toxicity VPIN: {res.max_vpin:.2%}')
```

### 4. Supply-Chain Knowledge Graph Alpha
```python
from genai_advanced_quant.data.loader import generate_supply_chain_network
from genai_advanced_quant.graph_alpha import SupplyChainGraphAlpha

net = generate_supply_chain_network()
graph_alpha = SupplyChainGraphAlpha()

# Backtest customer spillover momentum
res = graph_alpha.backtest_strategy(net['prices'], n_quantiles=4)
print(res.summary_table())
```

### 5. Wasserstein Distributionally Robust Portfolio Optimization
```python
from genai_advanced_quant.data.loader import load_dro_returns_data
from genai_advanced_quant.robust_dro import WassersteinDROOptimizer

train_returns, test_returns = load_dro_returns_data()
optimizer = WassersteinDROOptimizer(
    returns_data=train_returns, risk_aversion=1.5
)

# Solve for Wasserstein radius epsilon = 0.015
dro_res = optimizer.optimize(epsilon=0.015, norm_p=2)
print(dro_res.summary_table())
print(f'Worst-Case Loss Bound: {dro_res.robust_objective:.4f}')
```

---

## 🧪 Unit Testing

```bash
$ make test
....................................
----------------------------------------------------------------------
Ran 36 tests in 13.919s

OK
```

---

## 📜 References

1. **Cohen, L., Malloy, C., & Nguyen, Q. (2020)**. *Lazy Prices*. The Journal of Finance, 75(3), 1371-1415.
2. **Heston, S. L. (1993)**. *A closed-form solution for options with stochastic volatility with applications to bond and currency options*. The Review of Financial Studies, 6(2), 327-343.
3. **Carr, P., & Madan, D. (1999)**. *Option valuation using the fast Fourier transform*. Journal of Computational Finance, 2(4), 61-73.
4. **Fang, F., & Oosterlee, C. W. (2008)**. *A novel pricing method for European options based on Fourier-cosine series expansions*. SIAM Journal on Scientific Computing, 31(2), 826-848.
5. **Easley, D., López de Prado, M., & O'Hara, M. (2012)**. *Flow toxicity and liquidity in a high-frequency world*. The Review of Financial Studies, 25(5), 1457-1493.
6. **Cohen, L., & Frazzini, A. (2008)**. *Economic links and predictable returns*. The Journal of Finance, 63(4), 1977-2011.
7. **Kuhn, D., Esfahani, P. M., Nguyen, V. A., & Shafieezadeh-Abadeh, S. (2019)**. *Wasserstein distributionally robust optimization: Theory and applications in machine learning*. INFORMS Tutorials in Operations Research, 130-166.
