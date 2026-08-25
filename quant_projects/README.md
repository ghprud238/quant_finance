# Interview-Worthy Quant Projects (Projects 26-30)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> *"Technical skills can get you an interview. These projects can get you the offer.*  
> *Solve real problems $\rightarrow$ Show depth & curiosity $\rightarrow$ Think like a quant $\rightarrow$ Stand out in interviews $\rightarrow$ Build your edge."*

A production-grade Python package implementing the capstone **Projects 26 through 30** of the 30 Quant Finance Projects curriculum, engineered to demonstrate institutional-grade quantitative depth across market microstructure, optimal execution, stress testing, automated research pipelines, and production trading systems.

---

## 📸 Overview: Projects 26-30

| Project | Core Responsibility | Key Quantitative Methodologies |
| :--- | :--- | :--- |
| **26. Limit Order Book Simulator** | Market microstructure & liquidity dynamics | Level 2 continuous double auction, FIFO Price-Time priority, Order Book Imbalance (OBI), Volume-Weighted Micro-Price |
| **27. Optimal Execution Model** | Minimize market impact & execution risk | Almgren-Chriss closed-form trajectory, Temporary & Permanent impact, Implementation Shortfall (IS) attribution, TWAP/VWAP |
| **28. Portfolio Stress Testing Engine** | Tail risk & extreme macro scenario analysis | 2008 GFC, 2020 COVID, 2022 Rate Shocks, 2D Factor Sensitivity Grids, Systemic correlation breakdown ($\Sigma_{\text{crisis}}$) |
| **29. End-to-End Research Pipeline** | Automated quantitative research lifecycle | 5-Stage architecture (`DATA ➔ FEATURES ➔ BACKTEST ➔ EVALUATE ➔ DEPLOY`), Feature Store, FFD stationarity, Model Health Monitor |
| **30. Production Systematic Trading System** | Institutional multi-asset algorithmic engine | Alpha ensemble, Volatility targeting, Drawdown circuit breakers, Almgren-Chriss execution router, Pre-trade stress gating, OOS compounding |

---

## 🏛️ Repository Architecture

```text
interview_quant_projects/
├── data/                                 # Multi-asset tick & daily bar datasets
│   └── market_data.csv                   # SPY, QQQ, AAPL, MSFT, NVDA, GOOG, AMZN, JPM, XOM, TLT
├── output/                               # Rendered dark-theme capstone charts
│   ├── 26_order_book_snapshot.png        # Level 2 LOB depth & ladder snapshot
│   ├── 27_optimal_execution_trajectory.png # Almgren-Chriss vs TWAP execution trajectory
│   ├── 28_stress_testing_scenarios.png   # Historical & macro crisis scenario P&L bars
│   ├── 29_pipeline_overview.png          # 5-Stage research pipeline & feedback loop
│   ├── 30_strategy_equity_curve_oos.png  # Out-of-sample 2020-2024 compounding equity curve
│   └── interview_quant_infographic.png   # Full composite capstone infographic
├── scripts/
│   └── run_all_demos.py                  # End-to-end execution runner
├── src/
│   └── interview_quant/
│       ├── data/                         # Data layer & synthetic tick simulator
│       │   └── loader.py                 # Multi-asset dataset generator with regime shifts
│       ├── execution/                    # Module 27: Optimal Execution Models
│       │   ├── almgren_chriss.py         # Calculus of variations optimal trajectory & frontier
│       │   └── benchmark_executors.py    # TWAP, VWAP, POV & Implementation Shortfall attribution
│       ├── microstructure/               # Module 26: Limit Order Book Simulator
│       │   ├── order_book.py             # L2 FIFO matching engine, OBI, Micro-Price
│       │   └── simulator.py              # Poisson order arrival & cancellation simulator
│       ├── pipeline/                     # Module 29: Automated Research Pipeline
│       │   └── workflow.py               # 5-Stage workflow, Data sanity, FFD feature store, Tear sheet
│       ├── stress_testing/               # Module 28: Portfolio Stress Testing Engine
│       │   └── engine.py                 # Historical crises, 2D factor grids, Correlation breakdown
│       ├── systematic_system/            # Module 30: Production Trading System
│       │   └── trading_system.py         # Multi-asset alpha, Vol targeting, Circuit breaker, Stress gate
│       └── visualization/                # Dark-theme quant visualization suite
│           └── plots.py                  # Infographic, LOB ladder, trajectory & flowchart renderers
├── tests/                                # 34 Unit tests covering all 5 modules
│   ├── test_execution.py
│   ├── test_microstructure.py
│   ├── test_pipeline.py
│   ├── test_stress_testing.py
│   └── test_systematic_system.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/interview-quant-projects.git
cd interview-quant-projects
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

### 26. Limit Order Book & Market Microstructure
- **Order Book Imbalance (OBI)**:
  $$I = \frac{V_b - V_a}{V_b + V_a} \in [-1, +1]$$
- **Volume-Weighted Micro-Price**:
  $$P_{\text{micro}} = \frac{V_b P_a + V_a P_b}{V_b + V_a} = P_b + \frac{V_b}{V_b + V_a} (P_a - P_b)$$
- **Poisson Arrival Rate**: Limit orders ($\lambda_{\text{limit}}$), Market orders ($\lambda_{\text{market}}$), Cancellations ($\lambda_{\text{cancel}}$).

---

### 27. Optimal Execution Framework (Almgren-Chriss 2000)
Liquidating $X_0$ shares over horizon $T$ across $N$ discrete intervals ($\tau = T/N$):
- **Optimal Trajectory**:
  $$x_j = \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)} X_0, \quad j = 0, \dots, N$$
  $$n_j = x_{j-1} - x_j = \frac{2 \sinh(\frac{1}{2}\kappa \tau)}{\sinh(\kappa T)} \cosh\left(\kappa \left(T - (j - \frac{1}{2})\tau\right)\right) X_0$$
  where urgency parameter $\kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}$, $\lambda$ is risk aversion, $\sigma$ is asset volatility, and $\eta$ is temporary market impact.
- **Implementation Shortfall (IS) Attribution**:
  $$\text{Total Shortfall} = \text{Delay Cost} + \text{Permanent Impact} (\frac{1}{2}\gamma X_0^2) + \text{Temporary Impact} (\eta \sum \frac{n_j^2}{\tau}) + \text{Commissions}$$

---

### 28. Portfolio Stress Testing Engine
- **Multi-Asset Factor Sensitivities**:
  - Equities: $\Delta R_i = \beta_i \Delta S_{\text{eq}} + \beta_{i, \text{vol}} \Delta \sigma$
  - Fixed Income: $\frac{\Delta P}{P} \approx -D_i \Delta y + \frac{1}{2} C_i (\Delta y)^2 - D_{s, i} \Delta s_{\text{credit}}$
  - Options: $\Delta V_i \approx \Delta_i \Delta S + \frac{1}{2} \Gamma_i (\Delta S)^2 + \nu_i \Delta \sigma$
- **Systemic Correlation Breakdown**:
  $$\Sigma_{\text{crisis}} = (1 - \alpha)\Sigma_0 + \alpha J_{\sigma}$$

---

### 29. End-to-End Quant Research Pipeline

```text
┌────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌────────┐
│  DATA  │ ──► │ FEATURES │ ──► │ BACKTEST │ ──► │ EVALUATE │ ──► │ DEPLOY │
└────────┘     └──────────┘     └──────────┘     └──────────┘     └────────┘
    ▲                                                                 │
    └───────────────────── FEEDBACK LOOP ─────────────────────────────┘
```

- **Fractional Differencing (FFD)**: $(1-B)^d = \sum_{k=0}^\infty w_k B^k$ establishing stationarity while preserving memory.
- **Deployment Readiness Scoring**: Evaluates Sharpe $\ge 1.0$, MaxDD $\le 20\%$, Capacity $\ge \$10\text{M}$, Cost Drag $\le 250\text{bps}$, and Feature Drift alerts.

---

### 30. Full Production Systematic Trading System
- **Alpha Signal Ensemble**: Time-Series Momentum (TSMOM) + Moving Average Trend + Short-term Mean Reversion.
- **Dynamic Volatility Targeting**: $w_{i,t} = \text{Signal}_{i,t} \cdot \left(\frac{\sigma_{\text{target}}}{\hat{\sigma}_{i,t} \sqrt{N} \cdot 0.5}\right) \cdot \text{Multiplier}_{\text{CB}}$.
- **Trailing Drawdown Circuit Breaker**:
  - Drawdown $\le -8\% \implies$ scales leverage down to 60%.
  - Drawdown $\le -15\% \implies$ emergency liquidation to cash (10-day cooling-off period).
- **Pre-Trade Stress Test Gate**: Failsafe halt if potential loss under worst-case historical scenario exceeds tolerance.

---

## 📊 Code Walkthrough

### 1. Level 2 Limit Order Book
```python
from interview_quant.microstructure import LimitOrderBook, Order

lob = LimitOrderBook(name="LOB_EQUITY")
lob.add_limit_order(
    Order(
        order_id="B1",
        side="buy",
        price=100.01,
        volume=1000,
        timestamp=1.0,
        order_type="limit",
    )
)
lob.add_limit_order(
    Order(
        order_id="A1",
        side="sell",
        price=100.02,
        volume=400,
        timestamp=1.0,
        order_type="limit",
    )
)

print(f"Spread:      ${lob.spread:.4f}")
print(f"Micro-Price: ${lob.micro_price:.4f}")
print(f"Imbalance:   {lob.order_book_imbalance:+.4f}")
print(lob.get_snapshot_table(depth=5))
```

### 2. Almgren-Chriss Optimal Execution Schedule
```python
from interview_quant.execution import AlmgrenChrissModel

ac = AlmgrenChrissModel(
    total_shares=1_000_000,
    horizon=1.0,
    n_intervals=20,
    volatility=0.30,
    temp_impact=2.5e-6,
    perm_impact=2.5e-7,
)
traj = ac.solve_trajectory(risk_aversion=1e-6)

print(f"Expected Impact Cost: ${traj.expected_shortfall:,.2f}")
print(f"Cost Standard Dev:    ${traj.std_shortfall:,.2f}")
print(f"Initial Trade Size:   {traj.trade_sizes[0]:,.0f} shares")
```

### 3. Portfolio Stress Testing Engine
```python
from interview_quant.stress_testing import PortfolioStressTestingEngine

engine = PortfolioStressTestingEngine(portfolio_value=10_000_000.0)
summary = engine.summary_table()
print(summary.to_string(index=False))

# Test systemic correlation breakdown
corr_stress = engine.correlation_breakdown_stress(crisis_alpha=0.70)
print(f"Volatility Surge under Crisis: {corr_stress['vol_surge_pct']:+.1f}%")
```

### 4. End-to-End Research Pipeline
```python
from interview_quant.data.loader import load_dataset
from interview_quant.pipeline import QuantResearchPipeline

data = load_dataset()
pipeline = QuantResearchPipeline()


def my_alpha_logic(clean_data):
  close = clean_data.xs('Close', level='Field', axis=1)
  sig = (close.rolling(20).mean() > close.rolling(100).mean()).astype(float)
  return sig.div(sig.sum(axis=1).replace(0, 1), axis=0)


results = pipeline.run_full_pipeline(
    data, my_alpha_logic, strategy_name='Alpha Pipeline'
)
print(results['tear_sheet'].metrics_table)
print(results['deploy_report'].deployment_checklist)
```

### 5. Full Production Systematic Trading System
```python
from interview_quant.data.loader import load_dataset
from interview_quant.systematic_system import ProductionTradingSystem

data = load_dataset()
system = ProductionTradingSystem(
    target_annual_vol=0.10, drawdown_circuit_breaker=0.15
)
sys_res = system.run_systematic_system(data)

print(f"OOS CAGR:         {sys_res.metrics['CAGR']:+.2%}")
print(f"OOS Sharpe Ratio: {sys_res.metrics['Sharpe Ratio']:.2f}")
print(f"OOS Max Drawdown: {sys_res.metrics['Max Drawdown']:.2%}")
```

---

## 🧪 Unit Testing

```bash
$ make test
..................................
----------------------------------------------------------------------
Ran 34 tests in 1.661s

OK
```

---

## 📜 References

1. **Almgren, R., & Chriss, N. (2000)**. *Optimal execution of portfolio transactions*. Journal of Risk, 3(2), 5-40.
2. **Perold, A. F. (1988)**. *The implementation shortfall: Paper versus reality*. Journal of Portfolio Management, 14(3), 4-9.
3. **Bouchaud, J. P., Gefen, Y., Potters, M., & Wyart, M. (2004)**. *Fluctuations and response in financial markets: the subtle nature of 'random' price changes*. Quantitative Finance, 4(2), 176-185.
4. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. John Wiley & Sons.
5. **Cartea, Á., Jaimungal, S., & Penalva, J. (2015)**. *Algorithmic and High-Frequency Trading*. Cambridge University Press.
