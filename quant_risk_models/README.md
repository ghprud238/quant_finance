# Quant Risk & Portfolio Models (Projects 06-10)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"Measure Risk $\rightarrow$ Understand Uncertainty $\rightarrow$ Optimize Portfolios $\rightarrow$ Make Better Decisions.*  
> *Master risk first. Alpha follows."*

A production-grade Python package implementing **Projects 06 through 10** of the 30 Quant Finance Projects curriculum, focusing on advanced market risk measurement, tail risk diagnostics, Monte Carlo simulations, and modern portfolio optimization.

---

## 📸 Overview: Projects 06-10

| Project | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **06. Historical VaR Calculator** | Non-parametric empirical loss distribution | Percentile quantiles, Rolling VaR, Age-weighted decay simulation ($\lambda$), Bootstrap confidence intervals |
| **07. Parametric VaR Model** | Analytical risk modeling & fat-tail corrections | Delta-Normal Gaussian, Student's $t$ with MLE degrees of freedom, Cornish-Fisher skewness/kurtosis expansion |
| **08. Monte Carlo VaR Engine** | Path-dependent stochastic simulation | Geometric Brownian Motion (GBM), Merton Jump Diffusion, Correlated Cholesky shocks (100,000 paths), Fan charts |
| **09. Expected Shortfall / CVaR Model** | Coherent tail risk & regulatory backtesting | Historical ES, Parametric Gaussian/Student-$t$ ES, Component CVaR risk budgeting, Kupiec POF & Christoffersen tests |
| **10. Mean-Variance Portfolio Optimization** | Markowitz Modern Portfolio Theory (MPT) | SLSQP Quadratic Optimization, Global Minimum Volatility, Tangency (Max Sharpe) Portfolio, Efficient Frontier, CAL |

---

## 🏛️ Repository Architecture

```text
quant_risk_models/
├── data/                                 # Multi-asset price & return datasets
│   └── sample_portfolio_data.csv
├── output/                               # Rendered dark-theme charts
│   ├── 06_distribution_and_var.png       # Portfolio return histogram + 95%/99% VaR & ES
│   ├── 08_monte_carlo_var_simulation.png # 100,000 path Monte Carlo fan chart
│   ├── 09_expected_shortfall_cvar.png    # Risk metrics summary card
│   ├── 10_efficient_frontier_optimization.png # Efficient Frontier + random cloud
│   └── quant_risk_models_infographic.png # Full composite 4-panel dashboard
├── scripts/
│   └── run_all_demos.py                  # End-to-end execution runner
├── src/
│   └── quant_risk_models/
│       ├── cvar/                         # Project 09: Expected Shortfall & Backtesting
│       │   └── expected_shortfall.py     # Historical, Gaussian, Student-$t$ ES & Kupiec/Christoffersen
│       ├── data/                         # Data layer & market simulator
│       │   └── loader.py                 # Multi-asset returns loader with realistic tail events
│       ├── optimization/                 # Project 10: Mean-Variance Portfolio Optimization
│       │   └── mean_variance.py          # Min Vol, Max Sharpe, Efficient Frontier, Random Cloud, CAL
│       ├── portfolio/                    # Portfolio risk reporting
│       │   └── risk_metrics.py           # Summary table matching the infographic card
│       ├── var/                          # Projects 06, 07, 08: Value-at-Risk Engine
│       │   ├── historical.py             # Empirical quantile, age-weighted, bootstrap VaR
│       │   ├── monte_carlo.py            # Vectorized GBM & Jump Diffusion simulation
│       │   └── parametric.py             # Delta-Normal, Student-$t$, Cornish-Fisher expansion
│       └── visualization/                # Dark-theme quant visualization suite
│           └── plots.py                  # Matplotlib / Seaborn infographic renderers
├── tests/                                # 41 Unit tests covering all 5 projects
│   ├── test_data_loader.py
│   ├── test_expected_shortfall.py
│   ├── test_historical_var.py
│   ├── test_monte_carlo_var.py
│   ├── test_optimization.py
│   └── test_parametric_var.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/quant-risk-models.git
cd quant-risk-models
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

### 06. Historical Value at Risk (VaR)
Given empirical portfolio returns $\{R_1, R_2, \dots, R_T\}$, the historical Value at Risk at confidence level $\alpha \in (0, 1)$ represents the $(1-\alpha)$ lower quantile:
$$\text{VaR}_\alpha = -\text{Quantile}_{1-\alpha}(\{R_t\}_{t=1}^T)$$

- **Age-Weighted Simulation (Boudoukh, Richardson, Whitelaw 1998)**:
  Assigns exponential decaying weights $w_t = \frac{\lambda^{T-t}(1-\lambda)}{1-\lambda^T}$ with decay factor $\lambda \in (0, 1)$, giving higher priority to recent volatility spikes.
- **Bootstrap Confidence Intervals**: Resamples returns with replacement $B=2,000$ times to construct finite-sample confidence bands for $\text{VaR}_\alpha$.

---

### 07. Parametric Value at Risk Models

1. **Delta-Normal (Gaussian) VaR**:
   $$\text{VaR}_\alpha = -(\mu - z_\alpha \cdot \sigma)$$
   where $z_\alpha = \Phi^{-1}(\alpha)$ is the standard normal critical value ($z_{0.95} \approx 1.645$, $z_{0.99} \approx 2.326$).

2. **Student's $t$-Distributed VaR**:
   Accounts for fat tails by fitting degrees of freedom $\nu$ via Maximum Likelihood Estimation:
   $$\text{VaR}_\alpha = -\left(\mu - t_{\nu, \alpha} \sqrt{\frac{\nu - 2}{\nu}} \cdot \sigma\right)$$

3. **Cornish-Fisher Modified Expansion**:
   Adjusts the normal quantile $z_\alpha$ directly using sample skewness $S$ and excess kurtosis $K$:
   $$\tilde{z}_\alpha = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$
   $$\text{VaR}_{\alpha,\text{CF}} = -(\mu - \tilde{z}_\alpha \cdot \sigma)$$

4. **Multi-Asset Portfolio Analytical VaR**:
   $$\text{VaR}_\alpha(w) = -(w^T \mu - z_\alpha \sqrt{w^T \Sigma w})$$

---

### 08. Monte Carlo VaR Engine

Simulates $N=100,000$ discrete stochastic paths under Geometric Brownian Motion (GBM):
$$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z_t \right), \quad Z_t \sim \mathcal{N}(0, 1)$$

- **Merton Jump-Diffusion**: Adds compound Poisson jump process $J_t = \sum_{j=1}^{N_t} Y_j$ where $N_t \sim \text{Poisson}(\lambda \Delta t)$ and $\ln(1+Y_j) \sim \mathcal{N}(\mu_J, \sigma_J^2)$.
- **Multi-Asset Simulation**: Employs Cholesky factor $L$ of empirical covariance matrix $\Sigma = L L^T$ to inject cross-asset correlations into Brownian shocks: $Z_{\text{corr}} = L Z_{\text{indep}}$.

---

### 09. Expected Shortfall / CVaR Model

Expected Shortfall (Conditional Value at Risk) is a **coherent risk measure** satisfying sub-additivity ($\text{ES}(X+Y) \le \text{ES}(X) + \text{ES}(Y)$), quantifying the average loss in the $(1-\alpha)$ worst-case tail:

1. **Historical Expected Shortfall**:
   $$\text{ES}_\alpha = -\frac{1}{| \{t : R_t \le -\text{VaR}_\alpha\} |} \sum_{R_t \le -\text{VaR}_\alpha} R_t$$

2. **Parametric Gaussian Expected Shortfall**:
   $$\text{ES}_\alpha = -\left(\mu - \sigma \frac{\phi(z_\alpha)}{1 - \alpha}\right)$$
   where $\phi$ is the standard normal probability density function.

3. **Parametric Student's $t$ Expected Shortfall**:
   $$\text{ES}_\alpha = -\left(\mu - \sigma \sqrt{\frac{\nu - 2}{\nu}} \frac{f_\nu(t_{\nu,\alpha})}{1 - \alpha} \left(\frac{\nu + t_{\nu,\alpha}^2}{\nu - 1}\right)\right)$$

4. **Regulatory Exception Backtesting**:
   - **Kupiec Proportion of Failures (POF) Likelihood Ratio Test**:
     $$LR_{\text{POF}} = -2 \ln \left[ \frac{(1-p)^{T-x} p^x}{(1 - \hat{p})^{T-x} \hat{p}^x} \right] \sim \chi^2(1)$$
   - **Christoffersen Independence Test**: Verifies absence of tail risk clustering over consecutive trading periods ($LR_{\text{ind}} \sim \chi^2(1)$).

---

### 10. Mean-Variance Portfolio Optimization

Markowitz Modern Portfolio Theory (MPT) solves for the pareto-optimal trade-off between expected return and variance using Sequential Least Squares Quadratic Programming (SLSQP):

1. **Global Minimum Volatility Portfolio**:
   $$\min_w w^T \Sigma w \quad \text{s.t.} \quad \sum_{i=1}^n w_i = 1, \quad l_i \le w_i \le u_i$$

2. **Maximum Sharpe Ratio (Tangency) Portfolio**:
   $$\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}} \quad \text{s.t.} \quad \sum_{i=1}^n w_i = 1, \quad l_i \le w_i \le u_i$$

3. **Target Return Efficient Portfolio**:
   $$\min_w w^T \Sigma w \quad \text{s.t.} \quad w^T \mu \ge \mu_{\text{target}}, \quad \sum_{i=1}^n w_i = 1$$

4. **Efficient Frontier & Capital Allocation Line (CAL)**:
   - Sweeps $N$ target return steps between minimum variance and maximum return.
   - Traces the linear tangent from $(0, R_f)$ to the Tangency portfolio:
     $$\mathbb{E}[R_p] = R_f + \text{Sharpe}_{\text{optimal}} \cdot \sigma_p$$

---

## 📊 Code Walkthrough

### 1. Value at Risk (Historical, Parametric, Cornish-Fisher)
```python
from quant_risk_models.var.historical import HistoricalVaRCalculator
from quant_risk_models.var.parametric import ParametricVaRModel

# Historical empirical VaR
h_calc = HistoricalVaRCalculator(portfolio_returns)
print(f"95% Historical VaR: {h_calc.compute_var(0.95):.2%}")
print(f"99% Historical VaR: {h_calc.compute_var(0.99):.2%}")

# Parametric models
p_model = ParametricVaRModel(portfolio_returns)
print(f"Gaussian VaR:       {p_model.gaussian_var(0.95):.2%}")
print(f"Cornish-Fisher VaR: {p_model.cornish_fisher_var(0.95):.2%}")
```

### 2. Monte Carlo Simulation Engine (100,000 Paths)
```python
from quant_risk_models.var.monte_carlo import MonteCarloVaREngine

mc = MonteCarloVaREngine(returns=portfolio_returns)
# Simulate 100,000 paths over 252 days
paths = mc.simulate_gbm(n_simulations=100000, horizon=252, n_steps=252, initial_value=1.0)

terminal_returns = paths[:, -1] - 1.0
print(f"1-Year 95% Simulated VaR: {mc.compute_var(terminal_returns, 0.95):.2%}")
print(f"1-Year 95% Simulated CVaR: {mc.compute_cvar(terminal_returns, 0.95):.2%}")
```

### 3. Expected Shortfall & Kupiec POF Backtesting
```python
from quant_risk_models.cvar.expected_shortfall import ExpectedShortfallModel

es_model = ExpectedShortfallModel(confidence_level=0.95)
print(f"95% Expected Shortfall: {es_model.historical_es(portfolio_returns):.2%}")

# Backtest against 95% historical cutoff
report = es_model.backtest_var(portfolio_returns, var_forecasts=0.0245)
print(report.summary())
```

### 4. Mean-Variance Portfolio Optimization & Efficient Frontier
```python
from quant_risk_models.optimization.mean_variance import MeanVarianceOptimizer

optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix, risk_free_rate=0.02)

# Optimal Tangency Portfolio (Max Sharpe)
optimal = optimizer.max_sharpe_ratio()
print(f"Optimal Return:     {optimal.expected_return:.2%}")
print(f"Optimal Volatility: {optimal.volatility:.2%}")
print(f"Optimal Sharpe:     {optimal.sharpe_ratio:.2f}")

# Generate 60 points on the Efficient Frontier
frontier = optimizer.efficient_frontier(n_points=60)
```

---

## 🧪 Unit Testing

```bash
$ make test
.........................................
----------------------------------------------------------------------
Ran 41 tests in 2.079s

OK
```

---

## 📜 References

1. **Artzner, P., Delbaen, F., Eber, J. M., & Heath, D. (1999)**. *Coherent Measures of Risk*. Mathematical Finance, 9(3), 203-228.
2. **Markowitz, H. (1952)**. *Portfolio Selection*. The Journal of Finance, 7(1), 77-91.
3. **Kupiec, P. H. (1995)**. *Techniques for Verifying the Accuracy of Risk Measurement Models*. The Journal of Derivatives, 3(2), 73-84.
4. **Christoffersen, P. F. (1998)**. *Evaluating Interval Forecasts*. International Economic Review, 39(4), 841-862.
5. **Cornish, E. A., & Fisher, R. A. (1938)**. *Moments and Cumulants in the Specification of Distributions*. Revue de l'Institut International de Statistique, 4(4), 307-320.
6. **Boudoukh, J., Richardson, M., & Whitelaw, R. (1998)**. *The Best of Both Worlds*. Risk, 11(5), 64-67.
