# Derivatives & Pricing Models (Projects 16-20)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"This is where your mathematics + programming + finance start coming together.*  
> *Compare your models against: Market Prices $\rightarrow$ Pricing Error $\rightarrow$ Greeks $\rightarrow$ Volatility Assumptions."*

A production-grade Python package implementing **Projects 16 through 20** of the 30 Quant Finance Projects curriculum, focusing on closed-form analytical solutions, numerical root-finding solvers, sensitivity Greeks, discrete binomial lattices, and Monte Carlo engines for vanilla and exotic derivatives.

---

## 📸 Overview: Projects 16-20

| Project | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **16. Black-Scholes Option Pricing Engine** | Closed-form European options valuation | Merton dividend yield ($q$), Exact Put-Call Parity, High-throughput option chain pricing |
| **17. Implied Volatility Solver & Smile** | Root-finding inversion & surface modeling | Newton-Raphson with analytical Vega, Brent fallback, SVI (Gatheral) calibration, 3D Vol Surface, Dupire Local Vol |
| **18. Option Greeks Calculator** | 1st, 2nd & 3rd-order risk sensitivities | Analytical $\Delta, \Gamma, \Theta, \nu, \rho$, Vanna, Volga/Vomma, Charm, Speed, Finite-difference numerical validation |
| **19. Binomial Option Pricing Model** | Discrete lattice dynamic programming | Cox-Ross-Rubinstein (CRR), Jarrow-Rudd, Leisen-Reimer, American early exercise premium, Lattice Greeks |
| **20. Monte Carlo Option Pricing Engine** | Stochastic simulation & exotic derivatives | Vectorized GBM paths ($N=100,000$), Antithetic & Control Variates, Asian, Barrier, Lookback, Longstaff-Schwartz LSM |

---

## 🏛️ Repository Architecture

```text
pricing_models/
├── data/                                 # Sample options market data & chains
│   └── sample_market.py
├── output/                               # Rendered dark-theme charts
│   ├── 16_black_scholes_card.png         # Option chain & BSM formula card
│   ├── 17_volatility_smile.png           # SVI calibrated volatility smile
│   ├── 17_volatility_surface_3d.png      # 3D Implied Volatility Surface
│   ├── 18_option_greeks_card.png         # Delta, Gamma, Theta, Vega, Rho cards
│   ├── 19_binomial_tree_diagram.png      # 3-step binomial lattice diagram
│   ├── 20_monte_carlo_simulation.png     # Monte Carlo option path fan chart
│   └── pricing_models_infographic.png    # Full composite 5-module dashboard
├── scripts/
│   └── run_all_demos.py                  # End-to-end demo execution suite
├── src/
│   └── pricing_models/
│       ├── black_scholes/                # Module 16: Black-Scholes Engine
│       │   └── engine.py                 # Closed-form European BSM pricer & option chains
│       ├── implied_vol/                  # Module 17: Implied Volatility & Surfaces
│       │   ├── smile.py                  # VolatilitySmile & Gatheral Raw SVI
│       │   ├── solver.py                 # Newton-Raphson & Brent root-finders
│       │   └── surface.py                # 2D/3D Vol Surface & Dupire Local Volatility
│       ├── greeks/                       # Module 18: Option Greeks Suite
│       │   ├── analytical.py             # 1st, 2nd & 3rd-order closed-form Greeks
│       │   └── numerical.py              # Central finite-difference numerical engine
│       ├── binomial_tree/                # Module 19: Binomial Lattice Models
│       │   └── lattice.py                # CRR, JR, LR lattices, European & American options
│       ├── monte_carlo/                  # Module 20: Monte Carlo & Exotic Derivatives
│       │   ├── exotics.py                # Asian, Barrier, Lookback, Longstaff-Schwartz LSM
│       │   └── pricer.py                 # Vectorized GBM, Antithetic & Control Variates
│       └── visualization/                # Dark-theme quant visualization suite
│           └── plots.py                  # Infographic, smile, tree & 3D surface renderers
├── tests/                                # 48 Unit tests covering all 5 modules
│   ├── test_black_scholes.py
│   ├── test_implied_vol.py
│   ├── test_greeks.py
│   ├── test_binomial_tree.py
│   └── test_monte_carlo.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/pricing-models.git
cd pricing-models
pip install -r requirements.txt
```

### Run All Demos & Generate Visualizations

```bash
python3 scripts/run_all_demos.py
```

### Run Full Test Suite

```bash
make test
# or: python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 🔬 Mathematical Formulations

### 16. Black-Scholes-Merton (BSM) Model
For an underlying asset $S_0$ with continuous dividend yield $q$, strike $K$, risk-free rate $r$, volatility $\sigma$, and time to maturity $T$:
$$d_1 = \frac{\ln(S_0 / K) + \left(r - q + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}$$

- **European Call**: $C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$
- **European Put**: $P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$
- **Put-Call Parity**: $C - P = S_0 e^{-qT} - K e^{-rT}$

---

### 17. Implied Volatility Inversion & SVI Smile
Inverts $BS(\sigma) - C_{\text{market}} = 0$ using **Newton-Raphson**:
$$\sigma_{n+1} = \sigma_n - \frac{BS(\sigma_n) - C_{\text{market}}}{\text{Vega}(\sigma_n)}$$

- **Gatheral Raw SVI Parametric Smile**:
  $$w(k) = \sigma_{\text{IV}}^2(k) \cdot T = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)$$
  where $k = \ln(K / F)$ is forward log-moneyness.
- **Dupire Local Volatility**:
  $$\sigma_{\text{local}}^2(K, T) = \frac{\frac{\partial C}{\partial T} + q C + (r - q) K \frac{\partial C}{\partial K}}{\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}}$$

---

### 18. Option Greeks Suite

| Greek | Mathematical Definition | Call Formula | Put Formula |
| :--- | :--- | :--- | :--- |
| **Delta ($\Delta$)** | $\frac{\partial V}{\partial S}$ | $e^{-qT} N(d_1)$ | $-e^{-qT} N(-d_1)$ |
| **Gamma ($\Gamma$)** | $\frac{\partial^2 V}{\partial S^2}$ | $\frac{e^{-qT} \phi(d_1)}{S_0 \sigma \sqrt{T}}$ | $\frac{e^{-qT} \phi(d_1)}{S_0 \sigma \sqrt{T}}$ |
| **Theta ($\Theta$)** | $-\frac{\partial V}{\partial T}$ | $-\frac{S_0 e^{-qT} \phi(d_1) \sigma}{2 \sqrt{T}} - r K e^{-rT} N(d_2) + q S_0 e^{-qT} N(d_1)$ | $-\frac{S_0 e^{-qT} \phi(d_1) \sigma}{2 \sqrt{T}} + r K e^{-rT} N(-d_2) - q S_0 e^{-qT} N(-d_1)$ |
| **Vega ($\nu$)** | $\frac{\partial V}{\partial \sigma}$ | $S_0 e^{-qT} \sqrt{T} \phi(d_1)$ | $S_0 e^{-qT} \sqrt{T} \phi(d_1)$ |
| **Rho ($\rho$)** | $\frac{\partial V}{\partial r}$ | $K T e^{-rT} N(d_2)$ | $-K T e^{-rT} N(-d_2)$ |
| **Vanna** | $\frac{\partial^2 V}{\partial S \partial \sigma}$ | $-e^{-qT} \phi(d_1) \frac{d_2}{\sigma}$ | $-e^{-qT} \phi(d_1) \frac{d_2}{\sigma}$ |
| **Volga / Vomma** | $\frac{\partial^2 V}{\partial \sigma^2}$ | $\nu \cdot \frac{d_1 d_2}{\sigma}$ | $\nu \cdot \frac{d_1 d_2}{\sigma}$ |

---

### 19. Binomial Option Pricing Lattice

- **CRR Up/Down Factors**:
  $$u = e^{\sigma \sqrt{\Delta t}}, \quad d = \frac{1}{u} = e^{-\sigma \sqrt{\Delta t}}, \quad p = \frac{e^{(r - q)\Delta t} - d}{u - d}$$
- **American Backward Induction**:
  $$V_{i, j} = \max\left( \text{Intrinsic}(S_{i, j}), e^{-r \Delta t}(p V_{i+1, j+1} + (1 - p) V_{i+1, j}) \right)$$
- **Early Exercise Premium**:
  $$\text{Premium}_{\text{early}} = V_{\text{American}} - V_{\text{European}}$$

---

### 20. Monte Carlo Pricing & Exotics

- **Risk-Neutral Vectorized Simulation**:
  $$S_T = S_0 \exp\left( \left(r - q - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z \right)$$
- **Variance Reduction**:
  - **Antithetic Variates**: Evaluates $(Z, -Z)$ path pairs.
  - **Control Variates**: $\hat{V}_{\text{CV}} = \hat{V}_{\text{MC}} - c \cdot (\hat{S}_T - S_0 e^{(r-q)T})$.
- **Longstaff-Schwartz LSM for American Options**:
  Performs backward cross-sectional polynomial regression on in-the-money paths to estimate conditional continuation values $\mathbb{E}[C_t \mid S_t]$ and optimal stopping times.

---

## 📊 Code Walkthrough

### 1. Black-Scholes Pricing & Parity
```python
from pricing_models.black_scholes import BlackScholesModel

call = BlackScholesModel.calculate(S0=100, K=100, T=0.5, r=0.05, sigma=0.20, q=0.01, option_type="call")
put = BlackScholesModel.calculate(S0=100, K=100, T=0.5, r=0.05, sigma=0.20, q=0.01, option_type="put")

print(f"Call: ${call.call_price:.4f} | Put: ${put.put_price:.4f}")
print(f"Parity Error: {call.put_call_parity_diff:.2e}")
```

### 2. Inverting Implied Volatility & SVI Smile
```python
from pricing_models.implied_vol import ImpliedVolatilitySolver, VolatilitySmile

solver = ImpliedVolatilitySolver()
iv = solver.solve(market_price=5.80, spot=100, strike=100, time_to_expiry=0.5, risk_free_rate=0.05)
print(f"Implied Volatility: {iv:.2%}")

smile = VolatilitySmile(spot=100, time_to_expiry=0.5, risk_free_rate=0.05)
svi = smile.fit_svi(strikes=[80, 90, 100, 110, 120], implied_vols=[0.24, 0.21, 0.18, 0.20, 0.24])
```

### 3. Calculating Full Greek Sensitivities
```python
from pricing_models.greeks import AnalyticalGreeks

greeks = AnalyticalGreeks.calculate_all(S0=100, K=100, T=0.5, r=0.05, sigma=0.20, q=0.01, option_type="call")
print(f"Delta: {greeks.delta:.4f} | Gamma: {greeks.gamma:.4f} | Vega (1%): {greeks.vega_pct:.4f}")
print(f"Daily Theta: {greeks.theta_daily:.4f} | Vanna: {greeks.vanna:.4f} | Volga: {greeks.volga:.4f}")
```

### 4. Binomial Lattice (American vs European)
```python
from pricing_models.binomial_tree import BinomialTreePricer

tree = BinomialTreePricer(S0=100, K=100, T=0.5, r=0.05, sigma=0.20, q=0.01, n_steps=200)
american_put = tree.price(option_type="put", exercise_style="american", model="crr")
print(f"American Put: ${american_put.american_price:.4f} (Early Prem: ${american_put.early_exercise_premium:.4f})")
```

### 5. Monte Carlo Exotics & Longstaff-Schwartz LSM
```python
from pricing_models.monte_carlo import ExoticOptionPricer

exotics = ExoticOptionPricer(S0=100, K=100, T=0.5, r=0.05, sigma=0.20, q=0.01)
asian = exotics.price_asian(option_type="call", averaging_type="arithmetic", n_simulations=100000)
barrier = exotics.price_barrier(option_type="call", barrier_type="up_and_out", barrier_level=125.0)
lsm_put = exotics.price_american_lsm(option_type="put", n_simulations=50000, n_steps=50)

print(f"Asian Call:   ${asian.price:.4f}")
print(f"Barrier Call: ${barrier.price:.4f}")
print(f"LSM Put:      ${lsm_put.price:.4f}")
```

---

## 🧪 Unit Testing

```bash
$ make test
................................................
----------------------------------------------------------------------
Ran 48 tests in 4.341s

OK
```

---

## 📜 References

1. **Black, F., & Scholes, M. (1973)**. *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637-654.
2. **Merton, R. C. (1973)**. *Theory of Rational Option Pricing*. Bell Journal of Economics and Management Science, 4(1), 141-183.
3. **Cox, J. C., Ross, S. A., & Rubinstein, M. (1979)**. *Option Pricing: A Simplified Approach*. Journal of Financial Economics, 7(3), 229-263.
4. **Gatheral, J. (2004)**. *A parsimonious arbitrage-free parameterization of the volatility surface*. Presentation at Global Derivatives.
5. **Longstaff, F. A., & Schwartz, E. S. (2001)**. *Valuing American Options by Simulation: A Simple Least-Squares Approach*. Review of Financial Studies, 14(1), 113-147.
6. **Dupire, B. (1994)**. *Pricing with a Smile*. Risk Magazine, 7(1), 18-20.
