# Advanced Quantitative Research & Machine Learning (Projects 21-25)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"The goal isn't to throw XGBoost or neural networks at prices. The goal is to answer:*  
> *1. What is the signal?*  
> *2. Why should it exist?*  
> *3. How stable is it?*  
> *4. Does it survive costs and regime changes?"*

A production-grade Python package implementing **Projects 21 through 25** of the 30 Quant Finance Projects curriculum, moving beyond basic heuristics to econometric time-series modeling, fixed-income term structure calibration, adaptive state-space Kalman filters, leakage-free financial machine learning, and alternative data alpha engineering.

---

## 📸 Overview: Projects 21-25

| Project | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **21. Volatility Forecasting with GARCH** | Time-varying volatility & volatility clustering | GARCH(1,1), GJR-GARCH asymmetric leverage ($\gamma$), MLE estimation, Persistence ($\alpha+\beta$), Multi-step term structure |
| **22. Yield Curve Term Structure Modeling** | Sovereign interest rate curves & bond pricing | Nelson-Siegel (Level, Slope, Curvature), Nelson-Siegel-Svensson (NSS), Zero-coupon bootstrapping, Forward rates, Yield PCA |
| **23. Kalman Filter for Dynamic Pairs Trading** | Online adaptive state-space estimation | Recursive Kalman predict/update steps, Dynamic hedge ratio ($\beta_t$), Innovation Z-score ($e_t / \sqrt{Q_t}$), Adaptive stat arb |
| **24. Machine Learning Return Predictor** | Predictive alpha modeling with leakage controls | Fixed-width Fractional Differencing (FFD), High-signal technical features, Purged TimeSeries CV with embargo, Out-of-sample IC / Rank IC |
| **25. Alternative Data Alpha Model** | Non-traditional datasets & orthogonal alpha | Sentiment / Web traffic / Satellite signals, OLS Factor Neutralization (Market/Momentum/Size), Multi-horizon IC decay, Dollar-neutral L/S |

---

## 🏛️ Repository Architecture

```text
advanced_quant_ml/
├── data/                                 # Datasets (Macro yields, equities & alternative data)
│   ├── alternative_data.csv              # Sentiment scores, web traffic, order flow imbalance
│   ├── equity_returns.csv                # SPY, QQQ, AAPL, NVDA
│   └── yield_curve_treasury.csv          # US Treasury Par Yields (1M to 30Y tenors)
├── output/                               # Rendered dark-theme charts
│   ├── 21_garch_volatility_forecast.png  # GARCH(1,1) volatility forecast vs actual
│   ├── 22_yield_curve_modeling.png       # Nelson-Siegel calibrated yield curve
│   ├── 23_kalman_filter_pairs.png        # Time-varying Kalman hedge ratio βt
│   ├── 24_ml_return_prediction.png       # Actual vs Predicted returns scatter & OOS IC
│   ├── 25_alternative_data_alpha.png     # Alternative signal strength vs forward returns
│   └── advanced_quant_ml_infographic.png # Full composite 5-module dashboard
├── scripts/
│   └── run_all_demos.py                  # End-to-end demo execution suite
├── src/
│   └── advanced_quant_ml/
│       ├── alternative_data/             # Module 25: Alternative Data Alpha Engine
│       │   └── alpha_model.py            # Factor orthogonalization, IC decay, dollar-neutral backtest
│       ├── data/                         # Data layer & synthetic market simulators
│       │   └── loader.py                 # Multi-asset & yield curve data loaders
│       ├── garch/                        # Module 21: GARCH Volatility Engine
│       │   └── model.py                  # GARCH(1,1), GJR-GARCH, MLE, Multi-step term structure
│       ├── kalman/                       # Module 23: Kalman Filter State-Space
│       │   └── filter.py                 # Dynamic pairs trading & Kalman Z-scores
│       ├── ml_predictor/                 # Module 24: Financial Machine Learning
│       │   ├── features.py               # Fractional differencing (FFD) & feature engineering
│       │   └── model.py                  # Regularized models & Purged TimeSeries CV
│       ├── visualization/                # Dark-theme quant visualization suite
│       │   └── plots.py                  # Infographic and module chart renderers
│       └── yield_curve/                  # Module 22: Fixed Income Term Structure
│           └── models.py                 # Nelson-Siegel, NSS, Bootstrapping, Yield PCA
├── tests/                                # 30 Unit tests covering all 5 modules
│   ├── test_alternative_data.py
│   ├── test_garch.py
│   ├── test_kalman.py
│   ├── test_ml_predictor.py
│   └── test_yield_curve.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/advanced-quant-ml.git
cd advanced-quant-ml
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

### 21. Volatility Forecasting with GARCH

#### Standard GARCH(1,1) & GJR-GARCH(1,1)
- **GJR-GARCH(1,1) with Asymmetric Leverage**:
  $$\sigma_t^2 = \omega + \left(\alpha + \gamma I_{\{\epsilon_{t-1} < 0\}}\right) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
  where $I_{\{\epsilon_{t-1} < 0\}} = 1$ if $\epsilon_{t-1} < 0$ (negative shock) and $0$ otherwise.
- **Maximum Likelihood Estimation (MLE)**:
  $$\mathcal{L}(\theta) = -\frac{1}{2} \sum_{t=1}^T \left( \ln(2\pi) + \ln(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2} \right)$$
- **Volatility Persistence & Unconditional Variance**:
  $$P = \alpha + \beta + \frac{1}{2}\gamma, \quad \sigma_L^2 = \frac{\omega}{1 - P}, \quad t_{1/2} = \frac{\ln(0.5)}{\ln(P)} \text{ days}$$
- **Multi-Step Volatility Term Structure**:
  $$\mathbb{E}_t[\sigma_{t+h}^2] = \sigma_L^2 + P^{h-1}\left(\sigma_{t+1}^2 - \sigma_L^2\right)$$

---

### 22. Yield Curve Term Structure Modeling

#### Nelson-Siegel 4-Parameter Model (1987)
$$y(\tau) = \beta_0 + \beta_1 \left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}\right) + \beta_2 \left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda}\right)$$
- **Level ($\beta_0$)**: Long-term asymptotic rate ($\lim_{\tau \to \infty} y(\tau) = \beta_0$).
- **Slope ($\beta_1$)**: Short-to-long spread ($\lim_{\tau \to 0} y(\tau) = \beta_0 + \beta_1$).
- **Curvature ($\beta_2$)**: Medium-term hump (peaks at $\tau \approx 1.793\lambda$).
- **Instantaneous Forward Rate**:
  $$f(\tau) = \beta_0 + \beta_1 e^{-\tau/\lambda} + \beta_2 \frac{\tau}{\lambda} e^{-\tau/\lambda}$$

#### Nelson-Siegel-Svensson (NSS) 6-Parameter Model (1994)
Adds a second curvature hump with decay $\lambda_2$:
$$y(\tau) = \text{NS}(\tau) + \beta_3 \left(\frac{1 - e^{-\tau/\lambda_2}}{\tau/\lambda_2} - e^{-\tau/\lambda_2}\right)$$

#### Zero-Coupon Bootstrapping
Extracts discount factors $P(0, \tau_n) = \frac{1 - \frac{C_n}{m}\sum_{i=1}^{n-1} P(0, \tau_i)}{1 + \frac{C_n}{m}}$ and spot zero rates $z(\tau) = -\frac{\ln P(0, \tau)}{\tau}$.

---

### 23. Kalman Filter for Dynamic Pairs Trading

#### State-Space Formulation
- **State Transition (Hidden Beta Walk)**:
  $$\theta_t = \theta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q_t), \quad \text{where } \theta_t = [\alpha_t, \beta_t]^T$$
- **Measurement Equation**:
  $$y_t = H_t \theta_t + v_t, \quad v_t \sim \mathcal{N}(0, R), \quad \text{where } H_t = [1, x_t]$$

#### Online Recursive Equations
1. **Prior Prediction**: $\hat{\theta}_{t|t-1} = \hat{\theta}_{t-1|t-1}, \quad P_{t|t-1} = P_{t-1|t-1} + Q_t$
2. **Measurement Innovation**: $e_t = y_t - H_t \hat{\theta}_{t|t-1}, \quad Q_t = H_t P_{t|t-1} H_t^T + R$
3. **Kalman Gain**: $K_t = P_{t|t-1} H_t^T / Q_t$
4. **Posterior State Update**: $\hat{\theta}_{t|t} = \hat{\theta}_{t|t-1} + K_t e_t, \quad P_{t|t} = (I - K_t H_t) P_{t|t-1}$
5. **Standardized Kalman Z-Score**: $Z_t = \frac{e_t}{\sqrt{Q_t}}$

---

### 24. Machine Learning Return Predictor

#### Fractional Differencing (FFD)
Preserves maximum multi-period memory while establishing strict stationarity (ADF test $p < 0.05$):
$$(1 - B)^d = \sum_{k=0}^\infty w_k B^k, \quad w_0 = 1, \quad w_k = -w_{k-1} \frac{d - k + 1}{k}$$

#### Purged & Embargoed TimeSeries Cross-Validation
Eliminates information leakage across overlapping forward return labels:
- **Purging**: Removes training samples whose forward outcome overlaps with the test set evaluation window.
- **Embargoing**: Drops training samples immediately following test periods to eliminate post-test autoregressive correlation.

#### Performance Metrics
- **Information Coefficient (IC)**: $\text{IC} = \text{Corr}(\hat{y}, y)$ (Pearson).
- **Rank IC**: $\text{Rank IC} = \text{SpearmanCorr}(\text{Rank}(\hat{y}), \text{Rank}(y))$.

---

### 25. Alternative Data Alpha Model

#### Factor Orthogonalization / Neutralization
Residualizes raw alternative signals against traditional Fama-French risk factors ($X = [\text{Mkt}, \text{Mom}, \text{Size}]$) via OLS projection:
$$S_{\text{neutral}} = S - X (X^T X)^{-1} X^T S$$
ensuring the strategy extracts **pure idiosyncratic alpha** without disguised systematic beta exposure.

#### Multi-Horizon IC Decay
Tracks the temporal persistence of predictive power across lookaheads $h \in [1, 2, 5, 10, 21, 42, 63]$ days:
$$\text{IC}(h) = \text{Corr}\left( S_{\text{neutral}, t}, \frac{P_{t+h} - P_t}{P_t} \right), \quad IR_{\text{IC}} = \frac{\overline{\text{IC}}}{\sigma_{\text{IC}}}$$

---

## 📊 Code Walkthrough

### 1. GARCH Volatility Fitting & Forecasting
```python
from advanced_quant_ml.data.loader import load_equity_returns
from advanced_quant_ml.garch import GARCHModel

spy_returns = load_equity_returns(ticker="SPY")

# Fit GJR-GARCH(1,1) via MLE
garch = GARCHModel(model_type="GJR-GARCH")
fit_res = garch.fit(spy_returns)
print(fit_res.summary())

# 30-day term structure volatility forecast
forecast = garch.forecast(horizon=30)
print(f"30-day Vol Forecast: {forecast.cumulative_annualized_volatility[-1]:.2%}")
```

### 2. Nelson-Siegel Yield Curve Calibration
```python
from advanced_quant_ml.data.loader import (
    YIELD_CURVE_MATURITIES,
    load_yield_curve_data,
)
from advanced_quant_ml.yield_curve import (
    NelsonSiegelModel,
    YieldCurveBootstrapper,
)

yields_df = load_yield_curve_data()
latest_par = yields_df.iloc[-1].values

ns_model = NelsonSiegelModel()
ns_fit = ns_model.fit(YIELD_CURVE_MATURITIES, latest_par)
print(ns_fit.summary())

# Bootstrap zero-coupon spot rates
boot = YieldCurveBootstrapper.bootstrap_par_yields(
    YIELD_CURVE_MATURITIES, latest_par
)
print(f"10Y Spot Rate: {boot.get_zero_rate(10.0):.4f}%")
```

### 3. Online Kalman Filter Pairs Trading
```python
from advanced_quant_ml.kalman import KalmanFilterPairs, KalmanPairsStrategy

kf = KalmanFilterPairs(delta=1e-4, observation_cov=1e-3)
kf_res = kf.filter(y=qqq_prices, x=spy_prices)

print(f"Dynamic Hedge Ratio β_T: {kf_res.beta.iloc[-1]:.4f}")

# Backtest dynamic pairs strategy
strat = KalmanPairsStrategy(z_entry=1.8, z_exit=0.4)
res = strat.backtest(y=qqq_prices, x=spy_prices)
print(res.summary_table())
```

### 4. Financial ML with Fractional Differencing & Purged CV
```python
from advanced_quant_ml.ml_predictor import (
    FinancialFeatureEngineer,
    MLReturnPredictor,
)

fe = FinancialFeatureEngineer(frac_diff_d=0.40)
X, y = fe.engineer_features(ohlc_df, include_target=True)

# Train with Purged TimeSeries CV
predictor = MLReturnPredictor(
    model_type="ridge", alpha=1.0, n_splits=5, purge_window=5, embargo_window=5
)
ml_res = predictor.fit_predict_cv(X.dropna(), y.dropna())

print(f"Out-of-Sample IC: {ml_res.information_coefficient:+.4f}")
print(f"Directional Accuracy: {ml_res.directional_hit_rate:.1%}")
```

### 5. Alternative Data Alpha & Factor Neutralization
```python
from advanced_quant_ml.alternative_data import AlternativeDataAlphaModel

alt_model = AlternativeDataAlphaModel(decay_factor=0.85, n_quantiles=5)

# Combine, smooth and neutralize signals
combined = alt_model.combine_signals(raw_signals_dict)
smoothed = alt_model.exponential_decay_smoothing(combined)
neutral = alt_model.neutralize_factors(smoothed, risk_factor_loadings)

# Evaluate IC decay & backtest long/short strategy
ic_decay = alt_model.compute_ic_decay(
    neutral, prices, horizons=[1, 2, 5, 10, 21]
)
print(ic_decay.summary_table())
```

---

## 🧪 Unit Testing

```bash
$ make test
..............................
----------------------------------------------------------------------
Ran 30 tests in 6.941s

OK
```

---

## 📜 References

1. **Bollerslev, T. (1986)**. *Generalized autoregressive conditional heteroskedasticity*. Journal of Econometrics, 31(3), 307-327.
2. **Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993)**. *On the relation between the expected value and the volatility of the nominal excess return on stocks*. Journal of Finance, 48(5), 1779-1801.
3. **Nelson, C. R., & Siegel, A. F. (1987)**. *Parsimonious modeling of yield curves*. Journal of Business, 473-489.
4. **Svensson, L. E. (1994)**. *Estimating and interpreting forward interest rates: Sweden 1992-1994*. NBER Working Paper No. 4871.
5. **Kalman, R. E. (1960)**. *A new approach to linear filtering and prediction problems*. Journal of Basic Engineering, 82(1), 35-45.
6. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. John Wiley & Sons.
