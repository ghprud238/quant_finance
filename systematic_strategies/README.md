# Systematic Trading Strategies (Projects 11-15)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"Every strategy should follow: Signal $\rightarrow$ Position Sizing $\rightarrow$ Transaction Costs $\rightarrow$ Backtest $\rightarrow$ Risk Metrics $\rightarrow$ Out-of-Sample Test.*  
> *A backtest with no transaction costs or out-of-sample validation isn't impressive. It's usually just curve-fitting."*

A production-grade Python package implementing **Projects 11 through 15** of the 30 Quant Finance Projects curriculum, focusing on systematic alpha generation, market-neutral statistical arbitrage, cross-sectional factor investing, and multi-asset trend following with institutional backtesting rigor.

---

## 📸 Overview: Projects 11-15

| Project | Core Philosophy | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **11. Moving Average Mean-Reversion Strategy** | Price deviates from moving average and reverts back. Buy low, sell high. | Rolling Z-score ($Z_t = \frac{P_t - \text{MA}_t}{\sigma_t}$), Bollinger Bands, RSI filtering, Mean-reversion exit |
| **12. Momentum Trading Strategy** | Ride strong trends and exit when momentum weakens. Trend is your friend. | Dual MA crossover, Time-Series Momentum (TSMOM / 12-1M), MACD trend filters, Volatility-targeted sizing |
| **13. Pairs Trading / Statistical Arbitrage** | Trade the mean reversion of the spread between two correlated assets. Market neutral. | Engle-Granger Cointegration (ADF), Kalman Filter dynamic hedge ratio ($\beta_t$), Ornstein-Uhlenbeck half-life |
| **14. Factor-Based Long/Short Strategy** | Go long stocks with strong factor exposure, short weak exposure. Data-driven edge. | Cross-sectional composite scoring (Value, Momentum, Quality, Low-Vol, Size), Quintile sorting, Dollar/Beta-neutral |
| **15. Multi-Asset Trend-Following System** | Capture macro trends across multiple asset classes with risk-managed positioning. | Equities, Bonds, FX, Commodities, Multi-horizon TSMOM, Volatility targeting, Equal Risk Contribution (Risk Parity) |

---

## 🏛️ Repository Architecture

```text
systematic_strategies/
├── data/                                 # Pre-generated datasets (2018–2024)
│   ├── cross_sectional_factors.csv       # Multi-factor score matrix
│   ├── cross_sectional_prices.csv        # 20-stock daily prices
│   ├── equities.csv                      # SPY, QQQ, AAPL, MSFT
│   ├── macro.csv                         # Equities (SPY), Bonds (TLT), FX (UUP), Commodities (GLD, USO)
│   └── pairs.csv                         # Cointegrated pairs (KO/PEP, XOM/CVX)
├── output/                               # Rendered dark-theme strategy charts
│   ├── 11_mean_reversion_strategy.png
│   ├── 12_momentum_strategy.png
│   ├── 13_pairs_trading_spread.png
│   ├── 14_factor_exposure_heatmap.png
│   ├── 15_multi_asset_trend.png
│   └── systematic_strategies_infographic.png
├── scripts/
│   └── run_all_demos.py                  # End-to-end execution runner
├── src/
│   └── systematic_strategies/
│       ├── data/                         # Data layer & synthetic generator
│       │   ├── loader.py                 # Multi-asset dataset loader
│       │   └── synthetic.py              # Correlated GBM, Merton jumps & OU spread simulator
│       ├── engine/                       # Core Backtesting & Execution Engine
│       │   ├── backtester.py             # Vectorized backtester with lagged execution
│       │   ├── costs.py                  # Linear fees, bid-ask half-spread, quadratic slippage, borrow
│       │   ├── position_sizing.py        # Volatility targeting, Kelly, Risk Parity / ERC
│       │   └── validation.py             # Train/Test split & Walk-Forward degradation diagnostics
│       ├── strategies/                   # Systematic Strategy Implementations (11-15)
│       │   ├── factor_long_short.py      # Project 14: Factor Long/Short
│       │   ├── mean_reversion.py         # Project 11: MA Mean-Reversion
│       │   ├── momentum.py               # Project 12: Momentum Trading
│       │   ├── multi_asset_trend.py      # Project 15: Multi-Asset Trend & Risk Parity
│       │   └── pairs_trading.py          # Project 13: Statistical Arbitrage
│       └── visualization/                # Dark-theme quant visualization suite
│           └── plots.py                  # Infographic and strategy chart renderers
├── tests/                                # 49 Unit tests covering all modules
│   ├── test_engine.py
│   ├── test_multi_asset_strategies.py
│   └── test_single_strategies.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/systematic-strategies.git
cd systematic-strategies
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

## 🔬 Mathematical Formulations

### 11. Moving Average Mean-Reversion Strategy
- **Z-Score Normalization**:
  $$Z_t = \frac{P_t - \text{MA}_t(N)}{\sigma_t(N)}$$
- **Bollinger Band Thresholds**:
  $$\text{Upper Band}_t = \text{MA}_t + k \sigma_t, \quad \text{Lower Band}_t = \text{MA}_t - k \sigma_t$$
- **Signal Logic**:
  $$\text{Signal}_t = \begin{cases} +1 & \text{if } Z_t \le -z_{\text{entry}} \text{ (Oversold / Buy)} \\ -1 & \text{if } Z_t \ge +z_{\text{entry}} \text{ (Overbought / Sell)} \\ 0 & \text{if } |Z_t| \le z_{\text{exit}} \text{ (Mean Reversion Exit)} \end{cases}$$

---

### 12. Momentum Trading Strategy
- **Dual Moving Average Crossover**:
  $$\text{Signal}_{\text{MA}, t} = \text{sign}\left( \text{EMA}_{\text{fast}, t} - \text{EMA}_{\text{slow}, t} \right)$$
- **Time-Series Momentum (TSMOM / 12-1M Momentum)**:
  $$R_{12-1, t} = \frac{P_{t-21}}{P_{t-252}} - 1, \quad \text{Signal}_{\text{TSMOM}, t} = \text{sign}(R_{12-1, t})$$
- **Volatility Targeting Position Sizing**:
  $$w_t = \text{Signal}_t \cdot \min\left( \frac{\sigma_{\text{target}}}{\hat{\sigma}_t}, \text{max\_leverage} \right)$$

---

### 13. Pairs Trading / Statistical Arbitrage
- **Engle-Granger Cointegration Test**:
  1. OLS cointegrating regression: $P_{1,t} = \alpha + \beta P_{2,t} + \epsilon_t$
  2. Augmented Dickey-Fuller (ADF) unit root test on residuals: $\Delta \epsilon_t = \gamma \epsilon_{t-1} + \sum_{i=1}^p \delta_i \Delta \epsilon_{t-i} + u_t$
- **Ornstein-Uhlenbeck (OU) Mean Reversion & Half-Life**:
  $$dS_t = \theta (\mu - S_t) dt + \sigma dW_t \implies t_{1/2} = \frac{\ln 2}{\theta}$$
- **Dollar-Neutral Spread Trading**:
  $$w_1 = \frac{1}{1 + |\beta_t|}, \quad w_2 = -\frac{\beta_t}{1 + |\beta_t|}$$

---

### 14. Factor-Based Long/Short Strategy
- **Composite Cross-Sectional Z-Score**:
  $$S_{i,t} = \sum_{k=1}^K w_k Z_{i,k,t}, \quad \text{where } Z_{i,k,t} = \text{clip}\left(\frac{F_{i,k,t} - \mu_{k,t}}{\sigma_{k,t}}, -3, +3\right)$$
- **Long/Short Quintile Allocation**:
  - Long Top Quintile ($Q_5$): $w_i = +\frac{0.5}{N_{Q_5}}$
  - Short Bottom Quintile ($Q_1$): $w_i = -\frac{0.5}{N_{Q_1}}$
  - Net Market Exposure: $\sum w_i = 0.0$ (Dollar-Neutral).

---

### 15. Multi-Asset Trend-Following & Risk Parity
- **4 Macro Asset Classes**: Equities (`SPY`), Bonds (`TLT`), FX (`UUP`), Commodities (`GLD`/`USO`).
- **Equal Risk Contribution (ERC) Risk Parity**:
  Solves the convex Spinu (2013) formulation to equalize risk contributions:
  $$\min_x \frac{1}{2} x^T \Sigma x - \frac{1}{K} \sum_{k=1}^K \ln(x_k), \quad w = \frac{x}{\sum x_k}$$
  ensuring $w_i (\Sigma w)_i = w_j (\Sigma w)_j$ for all asset classes.

---

### 16. Institutional Backtesting & Transaction Cost Modeling
- **Execution Cost Formula**:
  $$C(\Delta w_t) = (\text{fee\_bps} + \text{half\_spread\_bps}) \cdot |\Delta w_t| + \frac{1}{2} \gamma (\Delta w_t)^2$$
- **Walk-Forward Overfitting Degradation**:
  $$\text{Sharpe Degradation} = 1.0 - \frac{\text{Sharpe}_{\text{OOS}}}{\max(\text{Sharpe}_{\text{IS}}, 10^{-4})}$$

---

## 📊 Code Walkthrough

### 1. Moving Average Mean Reversion
```python
from systematic_strategies.strategies.mean_reversion import MovingAverageMeanReversionStrategy
from systematic_strategies.engine import BacktestEngine

strategy = MovingAverageMeanReversionStrategy(lookback_window=20, z_entry=2.0, z_exit=0.5)
signals = strategy.generate_signals(price_series)

engine = BacktestEngine()
result = engine.run(price_series, signals.position, strategy_name="MA Mean Reversion")
result.print_summary()
```

### 2. Statistical Arbitrage / Pairs Trading
```python
from systematic_strategies.strategies.pairs_trading import PairsTradingStrategy

pairs_strat = PairsTradingStrategy(lookback_window=60, hedge_method="ols")
res = pairs_strat.generate_signals(ko_prices, pep_prices)

print(f"ADF p-value:    {res.coint_test.p_value:.4f}")
print(f"OU Half-Life:   {res.ou_params.half_life_days:.1f} days")
print(f"Hedge Ratio:    {res.coint_test.hedge_ratio_static:.3f}")
```

### 3. Factor Long/Short Portfolio
```python
from systematic_strategies.strategies.factor_long_short import FactorLongShortStrategy

factor_strat = FactorLongShortStrategy(n_quantiles=5, dollar_neutral=True)
result = factor_strat.backtest(prices=stock_prices, factor_data=factor_matrices)
print(result.summary_table())
```

### 4. Multi-Asset Trend Following (Risk Parity)
```python
from systematic_strategies.strategies.multi_asset_trend import MultiAssetTrendStrategy

macro_strat = MultiAssetTrendStrategy(target_portfolio_vol=0.10, use_risk_parity=True)
trend_res = macro_strat.backtest(macro_prices_df)
print(trend_res.summary_table())
```

---

## 🧪 Unit Testing

```bash
$ make test
.................................................
----------------------------------------------------------------------
Ran 49 tests in 19.106s

OK
```

---

## 📜 References

1. **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)**. *Time series momentum*. Journal of Financial Economics, 104(2), 228-250.
2. **Engle, R. F., & Granger, C. W. (1987)**. *Co-integration and error correction: representation, estimation, and testing*. Econometrica, 251-276.
3. **Avellaneda, M., & Lee, J. H. (2010)**. *Statistical arbitrage in the US equities market*. Quantitative Finance, 10(7), 761-782.
4. **Fama, E. F., & French, K. R. (1993)**. *Common risk factors in the returns on stocks and bonds*. Journal of Financial Economics, 33(1), 3-56.
5. **Spinu, F. E. (2013)**. *An Algorithm for Computing Risk Parity Weights*. SSRN Electronic Journal.
6. **Bailey, D. H., & López de Prado, M. (2014)**. *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5), 94-107.
