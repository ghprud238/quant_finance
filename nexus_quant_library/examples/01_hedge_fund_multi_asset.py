"""
Example 01: Hedge Fund & Multi-Asset Investment Manager Workflow
================================================================
"""
import numpy as np
import scipy.stats as stats
import nexus_quant as nq

# 1. Ingest multi-asset market data
data_engine = nq.DataEngine(seed=42)
df_equities = data_engine.generate_equity_ohlcv(tickers=["AAPL", "MSFT", "GOOG", "AMZN", "XOM", "SPY"])
closes = df_equities.xs("Close", level="Field", axis=1)
returns = nq.foundations.returns.simple_returns(closes).dropna()

print(f"Loaded {len(returns)} trading days across {len(closes.columns)} assets.")

# 2. Risk Parity Allocation (Equal Risk Contribution via Spinu log-barrier)
cov_matrix = np.cov(returns.values.T)
rp_opt = nq.portfolio.risk_parity.RiskParityOptimizer(cov_matrix, asset_names=list(closes.columns))
erc_weights, rc_pct = rp_opt.optimize()
print("\nEqual Risk Contribution (Risk Parity) Weights:")
for ticker, w, rc in zip(closes.columns, erc_weights, rc_pct):
    print(f"  - {ticker}: {w:.2%} (Variance Contribution: {rc:.2%})")

# 3. Wasserstein Distributionally Robust Optimization (DRO) under regime uncertainty
mu = returns.mean().values * 252
dro_opt = nq.portfolio.wasserstein_dro.WassersteinDROOptimizer(mu, cov_matrix * 252, risk_aversion=1.5)
dro_res = dro_opt.optimize(epsilon=0.015)

print(f"\nWasserstein DRO Robust Expected Return: {dro_res['expected_return']:.2%}")
print(f"Wasserstein DRO Worst-Case Loss Bound:  {dro_res['robust_objective']:.4f}")
print(f"Effective Number of Assets (1/HHI):     {dro_res['effective_n_assets']:.2f}")

# 4. Deflated Sharpe Ratio (DSR) Statistical Audit
dsr_calc = nq.rigor.deflated_sharpe.DeflatedSharpeRatioCalculator()
mean_ret = returns.mean(axis=1)
dsr_report = dsr_calc.compute_dsr(
    best_sharpe_ratio=1.85,
    num_trials=1000,
    sample_length=len(returns),
    skewness=float(stats.skew(mean_ret)),
    kurtosis=float(stats.kurtosis(mean_ret, fisher=False)),
)

print("\nDeflated Sharpe Ratio (DSR) Statistical Audit:")
print(f"  - Observed Backtest Sharpe: {dsr_report.observed_sharpe:.2f}")
print(f"  - Expected Max Sharpe under 1,000 noise trials: {dsr_report.expected_max_sharpe:.2f}")
print(f"  - Deflated Sharpe Ratio: {dsr_report.deflated_sharpe_ratio:.2%}")
print(f"  - Statistically Genuine Alpha (p < 0.05)? {dsr_report.is_significant_95}")
