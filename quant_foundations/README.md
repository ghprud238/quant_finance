# Quant Foundations: Market Data, Risk, Factor & Regime Analytics

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-118%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"Python $\rightarrow$ pandas $\rightarrow$ NumPy $\rightarrow$ statistics $\rightarrow$ financial data $\rightarrow$ visualization.*  
> *Don't start with an exotic trading strategy if you can't analyze basic market data properly."*

A production-grade Python repository implementing the **5 Core Quantitative Foundations** for quantitative researchers, portfolio managers, and algorithmic traders.

---

## 📸 Overview: The 5 Quant Foundations

| Module | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **01. Stock Returns & Volatility Analyzer** | Asset return dynamics & range volatility | Simple/Log returns, Parkinson, Garman-Klass, Yang-Zhang, Skewness, Kurtosis, Jarque-Bera |
| **02. Portfolio Risk Dashboard** | Downside, tail risk, & benchmark analytics | Sharpe, Sortino, Calmar, Max Drawdown, Cornish-Fisher VaR, CVaR/Expected Shortfall, Beta, Alpha |
| **03. Correlation & Covariance Engine** | Cross-asset dependence & dimensional reduction | Ledoit-Wolf Shrinkage, RiskMetrics EWMA, PCA Eigen-decomposition, Hierarchical Tree Seriation |
| **04. Factor Exposure Analyzer** | Multi-factor risk & style attribution | Fama-French 3/5, Carhart 4, Custom 6-Factor, White HC1 OLS, Systematic vs. Idiosyncratic Risk Decomposition |
| **05. Market Regime Detection Model** | Macro state classification & transition dynamics | 3-State Gaussian Hidden Markov Model (Baum-Welch EM + Viterbi), Trend-Vol Moving Avg Filter, GMM |

---

## 🏛️ Repository Architecture

```text
quant_foundations/
├── data/                                 # Datasets (OHLCV prices & factor series)
│   ├── sample_prices.csv
│   └── sample_factors.csv
├── output/                               # Rendered charts & master infographic
│   ├── 01_stock_returns_and_volatility.png
│   ├── 02_portfolio_risk_dashboard.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_factor_exposure.png
│   ├── 05_market_regime_detection.png
│   └── quant_foundations_infographic.png
├── scripts/
│   └── run_all_demos.py                  # End-to-end demo execution suite
├── src/
│   └── quant_foundations/
│       ├── analyzer/                     # Module 01: Return & Volatility analytics
│       │   ├── distribution.py           # Skewness, Kurtosis, Jarque-Bera, PDF fits
│       │   ├── returns.py                # Simple, Log, Rolling, CAGR returns
│       │   └── volatility.py             # Close-to-Close, Parkinson, Garman-Klass, Yang-Zhang
│       ├── correlation/                  # Module 03: Correlation & Covariance engine
│       │   ├── clustering.py             # Correlation distance & Hierarchical Seriation (HRP)
│       │   ├── matrix.py                 # Ledoit-Wolf Shrinkage, EWMA, Higham projection
│       │   └── pca.py                    # PCA eigen-decomposition & factor scores
│       ├── data/                         # Data layer & synthetic market generators
│       │   ├── loader.py                 # CSV loader / auto-generator
│       │   └── synthetic.py              # Correlated GBM + Merton Jump Diffusion generator
│       ├── factors/                      # Module 04: Factor exposure analyzer
│       │   ├── exposure.py               # Variance decomposition & factor risk attribution
│       │   └── model.py                  # Multi-factor OLS with White HC1 robust standard errors
│       ├── portfolio/                    # Module 02: Portfolio risk dashboard
│       │   ├── dashboard.py              # Terminal risk dashboard & summary aggregator
│       │   └── risk_metrics.py           # Sharpe, Sortino, Calmar, MaxDD, VaR, CVaR, Beta, Alpha
│       ├── regimes/                      # Module 05: Market regime detection models
│       │   ├── gmm_model.py              # Gaussian Mixture Model EM clustering
│       │   ├── heuristic.py              # Trend (200d SMA) + Realized Vol regime filter
│       │   └── hmm_model.py              # Gaussian Hidden Markov Model (Baum-Welch + Viterbi)
│       └── visualization/                # Dark-theme quant visualization suite
│           └── plots.py                  # Matplotlib / Seaborn infographic renderers
├── tests/                                # 118 Unit tests covering all modules
│   ├── test_correlation.py
│   ├── test_factors.py
│   ├── test_portfolio.py
│   ├── test_regimes.py
│   └── test_returns.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/quant-foundations.git
cd quant-foundations
pip install -r requirements.txt
```

### Run All Demos & Generate Charts

```bash
python3 scripts/run_all_demos.py
```

### Run Full Test Suite

```bash
make test
# or: python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 🔬 Mathematical Formulations & Deep Dive

### 01. Stock Returns & Volatility Analyzer

#### Return Dynamics
- **Simple Return**: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
- **Log Return**: $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(1 + R_t)$
- **Compound Annual Growth Rate (CAGR)**: $\text{CAGR} = \left(\prod_{t=1}^T (1 + R_t)\right)^{\frac{252}{T}} - 1$

#### Range-Based Volatility Estimators
While classical close-to-close volatility ($\sigma_{cc} = \sqrt{\frac{252}{N-1}\sum (r_t - \bar{r})^2}$) relies only on closing snapshots, range-based estimators utilize intraday price extremes to achieve up to **8x greater statistical efficiency**:

1. **Parkinson Volatility (1980)** (High-Low Range):
   $$\sigma_P = \sqrt{\frac{252}{4 \ln 2 \cdot N} \sum_{t=1}^N \left(\ln \frac{H_t}{L_t}\right)^2}$$

2. **Garman-Klass Volatility (1980)** (OHLC Range):
   $$\sigma_{GK} = \sqrt{\frac{252}{N} \sum_{t=1}^N \left[ 0.5 \left(\ln \frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1) \left(\ln \frac{C_t}{O_t}\right)^2 \right]}$$

3. **Yang-Zhang Volatility (2000)** (Minimum-variance unbiased estimator accounting for overnight jump and drift):
   $$\sigma_{YZ}^2 = \sigma_{\text{overnight}}^2 + k \cdot \sigma_{\text{open-to-close}}^2 + (1-k) \cdot \sigma_{RS}^2$$
   $$\text{where } k = \frac{0.34}{1.34 + \frac{N+1}{N-1}}$$

#### Tail Diagnostics & Normality
- **Skewness**: $S = \frac{\frac{1}{N}\sum (R_t - \mu)^3}{\sigma^3}$ (Negative skew represents crash / left-tail risk)
- **Excess Kurtosis**: $K = \frac{\frac{1}{N}\sum (R_t - \mu)^4}{\sigma^4} - 3$ ($K > 0$ indicates fat tails / leptokurtosis)
- **Jarque-Bera Normality Test**: $JB = \frac{N}{6}\left(S^2 + \frac{K^2}{4}\right) \sim \chi^2(2)$

---

### 02. Portfolio Risk Dashboard

#### Performance & Risk-Adjusted Returns
- **Sharpe Ratio**: $\text{SR} = \frac{R_p - R_f}{\sigma_p \cdot \sqrt{252}}$
- **Sortino Ratio**: $\text{SoR} = \frac{R_p - R_f}{\sigma_{\text{downside}} \cdot \sqrt{252}}$, where $\sigma_{\text{downside}} = \sqrt{\frac{1}{N}\sum \min(R_t - \text{MAR}, 0)^2}$
- **Calmar Ratio**: $\text{CR} = \frac{\text{CAGR}}{|\text{Max Drawdown}|}$
- **Maximum Drawdown**: $\text{MDD} = \max_{t} \left( \frac{\text{HWM}_t - P_t}{\text{HWM}_t} \right)$ where $\text{HWM}_t = \max_{s \le t} P_s$

#### Value at Risk (VaR) & Expected Shortfall (CVaR)
- **Historical VaR**: $\text{VaR}_\alpha = -\text{Quantile}(R, 1-\alpha)$
- **Parametric Gaussian VaR**: $\text{VaR}_\alpha = -(\mu - z_\alpha \sigma)$
- **Cornish-Fisher Modified VaR** (Adjusts standard normal quantile $z_\alpha$ for skewness $S$ and excess kurtosis $K$):
  $$\tilde{z}_\alpha = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$
  $$\text{VaR}_{\alpha,\text{CF}} = -(\mu - \tilde{z}_\alpha \sigma)$$
- **Conditional VaR (Expected Shortfall)**: $\text{CVaR}_\alpha = -\mathbb{E}[R \mid R \le -\text{VaR}_\alpha]$

---

### 03. Correlation & Covariance Engine

#### Ledoit-Wolf Optimal Linear Shrinkage (2004)
Sample covariance estimators $S$ become ill-conditioned when $N$ (number of assets) is large relative to $T$ (sample observations). Ledoit-Wolf shrinks $S$ toward a structured target $F$ (e.g., constant correlation target):
$$\Sigma_{LW} = \delta^* F + (1 - \delta^*) S$$
where the optimal shrinkage intensity $\delta^* \in [0, 1]$ minimizes the expected quadratic Frobenius loss $\mathbb{E}[\|\Sigma_{LW} - \Sigma\|^2]$.

#### RiskMetrics EWMA Covariance
$$\Sigma_t = \lambda \Sigma_{t-1} + (1 - \lambda) r_{t-1} r_{t-1}^T \quad (\lambda = 0.94)$$

#### Principal Component Analysis (PCA) & Eigen-Portfolios
Eigen-decomposition of correlation matrix $\Sigma = V \Lambda V^T$:
- Eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_N$ define the variance explained by each orthogonal risk factor.
- Eigenvectors $v_k$ yield eigen-portfolio asset allocations.
- $\text{PC}_1$ represents the broad market factor (systematic beta).

#### Hierarchical Tree Seriation / Quasi-Diagonalization
Computes the metric correlation distance:
$$d_{i,j} = \sqrt{0.5 \cdot (1 - \rho_{i,j})}$$
Applies Ward hierarchical agglomerative clustering and reorders the correlation matrix so that closely linked clusters are placed adjacently along the diagonal.

---

### 04. Factor Exposure Analyzer

#### Multi-Factor Regression Model
$$R_{i,t} - R_{f,t} = \alpha_i + \sum_{k=1}^K \beta_{i,k} F_{k,t} + \epsilon_{i,t}$$

- **Factors Supported**:
  - **Market**: Market excess return ($Mkt - R_f$)
  - **Value**: High Minus Low book-to-market ($HML$)
  - **Size**: Small Minus Big market cap ($SMB$)
  - **Momentum**: Up Minus Down past performance ($MOM$)
  - **Quality**: Robust Minus Weak profitability ($RMW$)
  - **Low Volatility**: Betting Against Beta ($LowVol / BAB$)

#### Robust Standard Errors (White HC1)
To prevent biased $t$-statistics under heteroskedastic return errors:
$$\text{Var}(\hat{\beta}) = \frac{N}{N - K} (X^T X)^{-1} X^T \text{diag}(\hat{\epsilon}_1^2, \dots, \hat{\epsilon}_N^2) X (X^T X)^{-1}$$

#### Total Variance Decomposition
$$\text{Var}(R_i) = \underbrace{\beta_i^T \Sigma_F \beta_i}_{\text{Systematic (Factor) Risk}} + \underbrace{\sigma_{\epsilon_i}^2}_{\text{Idiosyncratic (Specific) Risk}}$$
- Individual factor risk attribution:
  $$\text{Contribution}_k = \frac{\beta_{i,k} (\Sigma_F \beta_i)_k}{\text{Var}(R_i)}$$

---

### 05. Market Regime Detection Model

#### 3-State Gaussian Hidden Markov Model (HMM)
Financial markets switch between unobservable latent states (regimes $S_t \in \{\text{Bear}, \text{Neutral}, \text{Bull}\}$):

1. **Emission Probabilities**: $R_t \mid S_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2)$
2. **Transition Matrix**: $P_{i,j} = \mathbb{P}(S_t = j \mid S_{t-1} = i)$
3. **Parameter Estimation**: Scaled Baum-Welch Expectation-Maximization (EM) algorithm.
4. **State Decoding**: Viterbi Dynamic Programming Algorithm (Maximum A Posteriori sequence).
5. **Expected Regime Duration**:
   $$\mathbb{E}[\text{Duration}_k] = \frac{1}{1 - P_{k,k}} \text{ trading days}$$

---

## 📊 Sample Code Walkthrough

### 1. Volatility & Distribution Analysis
```python
from quant_foundations.data import load_prices
from quant_foundations.analyzer import (
    simple_returns,
    garman_klass_volatility,
    yang_zhang_volatility,
    jarque_bera_test
)

prices = load_prices()
aapl_ohlc = prices["AAPL"]
returns = simple_returns(aapl_ohlc["Close"]).dropna()

# Range volatilities
gk_vol = garman_klass_volatility(aapl_ohlc, window=21).iloc[-1]
yz_vol = yang_zhang_volatility(aapl_ohlc, window=21).iloc[-1]
stat, p_val, is_normal = jarque_bera_test(returns)

print(f"Garman-Klass Vol: {gk_vol:.2%}")
print(f"Yang-Zhang Vol:   {yz_vol:.2%}")
print(f"Is Gaussian Normal? {is_normal} (p-value: {p_val:.2e})")
```

### 2. Portfolio Risk Dashboard
```python
from quant_foundations.portfolio import PortfolioRiskDashboard

dashboard = PortfolioRiskDashboard(
    returns=my_portfolio_returns,
    benchmark_returns=spy_returns,
    name="Alpha Growth Portfolio",
    risk_free_rate=0.02
)
dashboard.print_dashboard()
```

### 3. Factor Exposure & Risk Attribution
```python
from quant_foundations.factors import MultiFactorRegression, FactorExposureReport

reg = MultiFactorRegression(cov_type="hc1").fit(asset_returns, factor_returns)
report = FactorExposureReport(model=reg, factor_returns=factor_returns)

print(report.summary_table())
print(report.variance_decomposition_table())
```

### 4. Regime Detection with Gaussian HMM
```python
from quant_foundations.regimes import GaussianHMMRegimeDetector

hmm = GaussianHMMRegimeDetector(n_states=3, random_state=42).fit(spy_returns)
regimes = hmm.predict(spy_returns)

print(hmm.transition_matrix_df())
print(hmm.regime_metrics())
```

---

## 🧪 Testing

The repository includes a comprehensive unit testing suite covering edge cases, numerical stability, distribution properties, and mathematical assertions.

```bash
$ make test
......................................................................................................................
----------------------------------------------------------------------
Ran 118 tests in 10.914s

OK
```

---

## 📜 References & Recommended Reading

1. **Parkinson, M. (1980)**. *The Extreme Value Method for Estimating the Variance of the Rate of Return*. Journal of Business, 53(1), 61-65.
2. **Garman, M. B., & Klass, M. J. (1980)**. *On the Estimation of Security Price Volatilities from Historical Data*. Journal of Business, 53(1), 67-78.
3. **Yang, D., & Zhang, Q. (2000)**. *Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices*. Journal of Business, 73(3), 477-492.
4. **Ledoit, O., & Wolf, M. (2004)**. *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365-411.
5. **Fama, E. F., & French, K. R. (2015)**. *A five-factor asset pricing model*. Journal of Financial Economics, 116(1), 1-22.
6. **Rabiner, L. R. (1989)**. *A tutorial on hidden Markov models and selected applications in speech recognition*. Proceedings of the IEEE, 77(2), 257-286.
7. **López de Prado, M. (2016)**. *Building Diversified Portfolios that Outperform Out of Sample*. Journal of Portfolio Management, 42(4), 59-69.
