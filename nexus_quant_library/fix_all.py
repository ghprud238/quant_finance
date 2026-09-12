import os

def w(path, content):
    full = os.path.join('/working_dir/nexus_quant_platform', path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print('Updated', path)

w('src/nexus_quant/ai_alternative_data/gnn_supply_chain.py', '''"""
Supply-Chain Knowledge Graph & Graph Neural Network (GNN) Spillover Momentum Alpha.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


class SupplyChainNetwork:
    """Represents a directed customer-supplier economic dependency network."""

    def __init__(self, tickers: List[str], adjacency_matrix: np.ndarray):
        self.tickers = tickers
        self.n_nodes = len(tickers)
        self.adj_matrix = np.array(adjacency_matrix, dtype=float)

    @classmethod
    def create_default_network(cls) -> "SupplyChainNetwork":
        tickers = ["AAPL", "NVDA", "TSMC", "ASML", "MSFT", "AMZN", "AMD", "QCOM", "AVGO", "CRUS"]
        n = len(tickers)
        A = np.zeros((n, n))
        idx = {t: i for i, t in enumerate(tickers)}

        A[idx["CRUS"], idx["AAPL"]] = 0.76
        A[idx["TSMC"], idx["AAPL"]] = 0.25
        A[idx["TSMC"], idx["NVDA"]] = 0.15
        A[idx["ASML"], idx["TSMC"]] = 0.40
        A[idx["NVDA"], idx["MSFT"]] = 0.18
        A[idx["NVDA"], idx["AMZN"]] = 0.14
        A[idx["AMD"], idx["MSFT"]] = 0.12
        A[idx["QCOM"], idx["AAPL"]] = 0.22
        A[idx["AVGO"], idx["AAPL"]] = 0.20
        return cls(tickers, A)

    def get_normalized_adjacency(self) -> np.ndarray:
        A_tilde = self.adj_matrix + np.eye(self.n_nodes)
        D_tilde = np.sum(A_tilde, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D_tilde, 1e-8)))
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def get_centrality_table(self) -> pd.DataFrame:
        A = self.adj_matrix
        out_degree = np.sum(A > 0, axis=1)
        in_degree = np.sum(A > 0, axis=0)
        hhi = np.sum(A ** 2, axis=1)
        total_dep = np.sum(A, axis=1)

        # Standard PageRank with dangling node redistribution
        col_sums = np.sum(A, axis=0)
        M = np.zeros_like(A)
        for j in range(self.n_nodes):
            if col_sums[j] > 0:
                M[:, j] = A[:, j] / col_sums[j]
            else:
                M[:, j] = 1.0 / self.n_nodes

        d = 0.85
        p = np.ones(self.n_nodes) / self.n_nodes
        for _ in range(100):
            p = (1.0 - d) / self.n_nodes + d * (M @ p)
            p = p / np.sum(p)

        return pd.DataFrame({
            "Ticker": self.tickers,
            "PageRank_Centrality": p,
            "Supplier_Customer_Count": in_degree,
            "Supplier_Out_Count": out_degree,
            "Customer_Concentration_HHI": hhi,
            "Total_Customer_Revenue_Pct": total_dep,
        }).sort_values("PageRank_Centrality", ascending=False)


@dataclass
class SupplyChainAlphaResult:
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    information_coefficient: float
    ic_information_ratio: float
    equity_curve: pd.Series
    strategy_returns: pd.Series

    def summary(self) -> str:
        return f"""Supply-Chain GNN Alpha Performance:
  - Strategy CAGR:                   {self.cagr:+.2%}
  - Annualized Volatility:           {self.volatility:.2%}
  - Sharpe Ratio (Rf=2%):            {self.sharpe_ratio:.2f}
  - Maximum Drawdown:                {self.max_drawdown:.2%}
  - Mean Information Coefficient:    {self.information_coefficient:+.4f}
  - IC Information Ratio:            {self.ic_information_ratio:.2f}"""


class SupplyChainGraphAlpha:
    """GNN-driven Customer-to-Supplier Lead-Lag Momentum Extraction."""

    def __init__(self, network: Optional[SupplyChainNetwork] = None, lead_lag_window: int = 5):
        self.network = network or SupplyChainNetwork.create_default_network()
        self.lead_lag_window = lead_lag_window

    def compute_graph_momentum_signals(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        aligned_tickers = [t for t in self.network.tickers if t in prices_df.columns]
        P = prices_df[aligned_tickers]
        returns = P.pct_change(self.lead_lag_window)

        A_norm = self.network.get_normalized_adjacency()
        signals = pd.DataFrame(index=returns.index, columns=aligned_tickers, dtype=float)

        for dt in returns.index:
            r_t = returns.loc[dt].fillna(0.0).values
            h1 = A_norm @ r_t
            h2 = A_norm @ np.maximum(0, h1)
            signals.loc[dt] = h2

        mean_sig = signals.mean(axis=1)
        std_sig = signals.std(axis=1).replace(0, 1.0)
        z_signals = signals.sub(mean_sig, axis=0).div(std_sig, axis=0)
        return z_signals

    def backtest_strategy(self, prices_df: pd.DataFrame, n_quantiles: int = 3, tx_cost_bps: float = 5.0) -> SupplyChainAlphaResult:
        signals = self.compute_graph_momentum_signals(prices_df).shift(1)
        aligned_tickers = [t for t in self.network.tickers if t in prices_df.columns]
        daily_returns = prices_df[aligned_tickers].pct_change()

        strat_returns = []
        ics = []

        for dt in daily_returns.index:
            if dt not in signals.index:
                continue
            sig_row = signals.loc[dt].dropna()
            ret_row = daily_returns.loc[dt].dropna()
            common = sig_row.index.intersection(ret_row.index)
            if len(common) < 4:
                continue

            s = sig_row[common]
            r = ret_row[common]

            ic, _ = stats.spearmanr(s, r)
            if not np.isnan(ic):
                ics.append(ic)

            q_high = s.quantile(1.0 - 1.0 / n_quantiles)
            q_low = s.quantile(1.0 / n_quantiles)

            long_mask = s >= q_high
            short_mask = s <= q_low

            w = pd.Series(0.0, index=common)
            if long_mask.sum() > 0:
                w[long_mask] = 0.5 / long_mask.sum()
            if short_mask.sum() > 0:
                w[short_mask] = -0.5 / short_mask.sum()

            gross_ret = (w * r).sum()
            strat_returns.append(gross_ret - tx_cost_bps * 1e-4 * 0.1)

        ret_series = pd.Series(strat_returns)
        cagr = float(np.prod(1.0 + ret_series) ** (252.0 / max(len(ret_series), 1)) - 1.0) if len(ret_series) > 0 else 0.0
        vol = float(ret_series.std() * np.sqrt(252.0)) if len(ret_series) > 0 else 0.0
        sharpe = float((cagr - 0.02) / vol) if vol > 0 else 0.0

        cum = (1.0 + ret_series).cumprod()
        peaks = cum.cummax()
        dd = (cum - peaks) / peaks
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0
        win_rate = float(np.mean(ret_series > 0)) if len(ret_series) > 0 else 0.0

        mean_ic = float(np.mean(ics)) if ics else 0.0
        ic_ir = float(mean_ic / (np.std(ics) + 1e-8)) if ics else 0.0

        return SupplyChainAlphaResult(
            cagr=cagr,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            information_coefficient=mean_ic,
            ic_information_ratio=ic_ir,
            equity_curve=cum,
            strategy_returns=ret_series,
        )
''')

w('src/nexus_quant/validation_rigor/deflated_sharpe.py', '''"""
The Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR) (Bailey & López de Prado 2014).
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PSRReport:
    observed_sharpe: float
    benchmark_sharpe: float
    psr_value: float
    sample_length: int
    skewness: float
    kurtosis: float
    is_significant_95: bool


@dataclass
class DSRReport:
    observed_sharpe: float
    expected_max_sharpe: float
    deflated_sharpe_ratio: float
    p_value: float
    num_trials: int
    effective_trials: float
    sample_length: int
    skewness: float
    kurtosis: float
    is_significant_95: bool
    is_significant_99: bool
    min_track_record_length_years: float

    def summary(self) -> str:
        status = "GENUINE ALPHA (Passed DSR > 0.95)" if self.is_significant_95 else "OVERFITTED / FALSE DISCOVERY (Failed DSR <= 0.95)"
        return f"""Deflated Sharpe Ratio (DSR) Audit Report:
  - Observed Strategy Sharpe:         {self.observed_sharpe:.3f}
  - Multiple Testing Trials (N):      {self.num_trials:,}
  - Expected Max Sharpe from Noise:   {self.expected_max_sharpe:.3f}
  - Deflated Sharpe Ratio (DSR):      {self.deflated_sharpe_ratio:.2%}
  - Statistical Significance:         {status}
  - Return Skewness:                  {self.skewness:+.2f}
  - Return Pearson Kurtosis:          {self.kurtosis:.2f}
  - Min Track Record Length Required: {self.min_track_record_length_years:.2f} years"""


class DeflatedSharpeRatioCalculator:
    """Computes the Deflated Sharpe Ratio (DSR) correcting for selection bias and non-normality."""

    EULER_MASCHERONI = 0.5772156649015328606

    @classmethod
    def compute_psr(
        cls,
        observed_sharpe: float,
        benchmark_sharpe: float = 0.0,
        sample_length: int = 252,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        periods_per_year: int = 252,
    ) -> PSRReport:
        sr = observed_sharpe / np.sqrt(periods_per_year)
        sr_bm = benchmark_sharpe / np.sqrt(periods_per_year)
        T = float(sample_length)

        # Standard error of Sharpe ratio under non-normality
        denom = np.sqrt(1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * (sr ** 2))
        z = ((sr - sr_bm) * np.sqrt(T - 1.0)) / max(denom, 1e-6)
        psr = float(stats.norm.cdf(z))

        return PSRReport(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            psr_value=psr,
            sample_length=sample_length,
            skewness=skewness,
            kurtosis=kurtosis,
            is_significant_95=psr >= 0.95,
        )

    @classmethod
    def expected_max_sharpe(
        cls,
        num_trials: int,
        var_sharpe_trials: float = 0.25,
        mean_sharpe_trials: float = 0.0,
        average_trial_correlation: float = 0.0,
    ) -> Tuple[float, float]:
        """Extreme value theory approximation of expected maximum Sharpe ratio under N trials."""
        n_eff = 1.0 + (num_trials - 1.0) * (1.0 - average_trial_correlation)
        n_eff = max(1.0, n_eff)
        sigma_sr = np.sqrt(var_sharpe_trials)

        if n_eff <= 1.0:
            return mean_sharpe_trials, 1.0

        z1 = stats.norm.ppf(1.0 - 1.0 / n_eff)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_eff * np.e))
        em_sr = mean_sharpe_trials + sigma_sr * ((1.0 - cls.EULER_MASCHERONI) * z1 + cls.EULER_MASCHERONI * z2)
        return float(em_sr), float(n_eff)

    @classmethod
    def min_track_record_length(
        cls,
        observed_sharpe: float,
        benchmark_sharpe: float = 0.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        confidence_level: float = 0.95,
        periods_per_year: int = 252,
    ) -> float:
        """Calculates Minimum Track Record Length (MinTRL) in years."""
        sr = observed_sharpe / np.sqrt(periods_per_year)
        sr_bm = benchmark_sharpe / np.sqrt(periods_per_year)
        if sr <= sr_bm:
            return np.inf

        z_alpha = stats.norm.ppf(confidence_level)
        var_sr = 1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * (sr ** 2)
        min_t_periods = 1.0 + var_sr * ((z_alpha / (sr - sr_bm)) ** 2)
        return float(min_t_periods / periods_per_year)

    def compute_dsr(
        self,
        best_sharpe_ratio: float,
        num_trials: int = 100,
        var_sharpe_trials: float = 0.25,
        sample_length: int = 252,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        periods_per_year: int = 252,
        average_trial_correlation: float = 0.0,
    ) -> DSRReport:
        e_max_sr, n_eff = self.expected_max_sharpe(
            num_trials=num_trials,
            var_sharpe_trials=var_sharpe_trials,
            average_trial_correlation=average_trial_correlation,
        )

        psr_rep = self.compute_psr(
            observed_sharpe=best_sharpe_ratio,
            benchmark_sharpe=e_max_sr,
            sample_length=sample_length,
            skewness=skewness,
            kurtosis=kurtosis,
            periods_per_year=periods_per_year,
        )

        min_trl = self.min_track_record_length(
            observed_sharpe=best_sharpe_ratio,
            benchmark_sharpe=0.0,
            skewness=skewness,
            kurtosis=kurtosis,
            periods_per_year=periods_per_year,
        )

        return DSRReport(
            observed_sharpe=best_sharpe_ratio,
            expected_max_sharpe=e_max_sr,
            deflated_sharpe_ratio=psr_rep.psr_value,
            p_value=1.0 - psr_rep.psr_value,
            num_trials=num_trials,
            effective_trials=n_eff,
            sample_length=sample_length,
            skewness=skewness,
            kurtosis=kurtosis,
            is_significant_95=psr_rep.psr_value >= 0.95,
            is_significant_99=psr_rep.psr_value >= 0.99,
            min_track_record_length_years=min_trl,
        )
''')

# Update test files
w('tests/test_ai_alt_data.py', '''"""
Unit Tests for AI & Alternative Data Engine.
"""

import unittest
import numpy as np
import pandas as pd

from nexus_quant.ai_alternative_data import (
    FinancialFeatureEngineer,
    PurgedTimeSeriesSplit,
    MLReturnPredictor,
    SupplyChainGraphAlpha,
    SupplyChainNetwork,
    SemanticDriftEngine,
    LazyPricesStrategy,
    MultiSourceSentimentEngine,
    MultiAgentHedgeFundSwarm,
)


class TestAIAlternativeData(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
        high = close * (1.0 + np.random.uniform(0.002, 0.015, n))
        low = close * (1.0 - np.random.uniform(0.002, 0.015, n))
        open_p = (high + low) / 2.0
        self.ohlc = pd.DataFrame({"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": np.random.uniform(1e6, 5e6, n)}, index=dates)

    def test_feature_engineering_and_fracdiff(self):
        fe = FinancialFeatureEngineer(frac_diff_d=0.35)
        X, y = fe.engineer_features(self.ohlc)
        self.assertGreaterEqual(len(X), 100)
        self.assertIn("frac_diff", X.columns)
        self.assertIn("rsi_14", X.columns)
        self.assertIn("bollinger_z", X.columns)
        self.assertEqual(len(X), len(y))

    def test_purged_cv_and_ml_predictor(self):
        fe = FinancialFeatureEngineer(frac_diff_d=0.35)
        X, y = fe.engineer_features(self.ohlc)
        predictor = MLReturnPredictor(alpha=1.0, n_splits=3, purge_window=3)
        res = predictor.fit_predict_cv(X, y)
        self.assertIsInstance(res.information_coefficient, float)
        self.assertGreaterEqual(res.directional_hit_rate, 0.0)
        self.assertLessEqual(res.directional_hit_rate, 1.0)

    def test_supply_chain_gnn_alpha(self):
        net = SupplyChainNetwork.create_default_network()
        cent = net.get_centrality_table()
        self.assertEqual(len(cent), len(net.tickers))
        self.assertAlmostEqual(cent["PageRank_Centrality"].sum(), 1.0, places=3)

        p_dict = {t: 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 200))) for t in net.tickers}
        p_df = pd.DataFrame(p_dict)
        g_alpha = SupplyChainGraphAlpha(net)
        res = g_alpha.backtest_strategy(p_df)
        self.assertIsInstance(res.sharpe_ratio, float)
        self.assertGreater(len(res.equity_curve), 50)

    def test_sec_semantic_drift_and_lazy_prices(self):
        engine = SemanticDriftEngine()
        doc_prior = {"mda": "The company had record revenue growth and stable operational margins.", "risk_factors": "General market risks apply to our retail products."}
        doc_curr_lazy = {"mda": "The company had record revenue growth and stable operational margins.", "risk_factors": "General market risks apply to our retail products."}
        doc_curr_drift = {"mda": "The company faces substantial litigation, regulatory breach, and margin impairment.", "risk_factors": "Severe material weakness in controls and cybersecurity breach."}

        rep_lazy = engine.analyze_filing_pair("MSFT", 2023, doc_prior, doc_curr_lazy)
        rep_drift = engine.analyze_filing_pair("HIGH_RISK_CO", 2023, doc_prior, doc_curr_drift)

        self.assertAlmostEqual(rep_lazy.cosine_drift_total, 0.0, places=2)
        self.assertGreater(rep_drift.cosine_drift_total, 0.15)
        self.assertEqual(rep_lazy.category, "LAZY_DISCLOSURE")
        self.assertEqual(rep_drift.category, "HIGH_DRIFT")

        df = pd.DataFrame([rep_lazy.__dict__, rep_drift.__dict__])
        strat = LazyPricesStrategy(quantile_cutoff=0.5)
        pos = strat.generate_positions(df)
        self.assertIn("Weight", pos.columns)

    def test_sentiment_fear_greed_and_swarm(self):
        engine = MultiSourceSentimentEngine()
        fgi = engine.compute_fear_greed_index(self.ohlc["Close"], self.ohlc["Volume"])
        self.assertGreaterEqual(fgi.composite_index.min(), 0.0)
        self.assertLessEqual(fgi.composite_index.max(), 100.0)

        swarm = MultiAgentHedgeFundSwarm()
        memo = swarm.run_investment_committee(macro_signal=0.8, crypto_onchain=0.7, sentiment_fgi=65.0)
        self.assertAlmostEqual(sum(memo.optimal_weights.values()), 1.0, places=4)
        self.assertGreater(memo.expected_portfolio_return, 0.0)


if __name__ == "__main__":
    unittest.main()
''')

w('tests/test_validation_orchestrator.py', '''"""
Unit Tests for Validation Rigor, Deflated Sharpe & Master Orchestrator.
"""

import unittest
import numpy as np
import pandas as pd

from nexus_quant.validation_rigor import (
    DeflatedSharpeRatioCalculator,
    QuantResearchValidationPipeline,
)
from nexus_quant.orchestrator import NexusMasterOrchestrator


class TestValidationOrchestrator(unittest.TestCase):

    def test_deflated_sharpe_calculations(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        
        psr = dsr_calc.compute_psr(observed_sharpe=2.0, benchmark_sharpe=0.0, sample_length=252)
        self.assertGreater(psr.psr_value, 0.95)
        self.assertTrue(psr.is_significant_95)

        em_sr, n_eff = dsr_calc.expected_max_sharpe(num_trials=1000, var_sharpe_trials=0.25)
        self.assertGreater(em_sr, 1.2)
        self.assertEqual(n_eff, 1000.0)

        # Deflated Sharpe under 5-year sample (1260 days)
        dsr = dsr_calc.compute_dsr(best_sharpe_ratio=2.5, num_trials=1000, sample_length=1260)
        self.assertGreater(dsr.deflated_sharpe_ratio, 0.95)
        self.assertTrue(dsr.is_significant_95)

        min_trl = dsr_calc.min_track_record_length(observed_sharpe=2.0, benchmark_sharpe=0.0)
        self.assertGreater(min_trl, 0.0)
        self.assertLess(min_trl, 10.0)

    def test_validation_pipeline_and_readiness(self):
        np.random.seed(42)
        pipe = QuantResearchValidationPipeline()
        p = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0015, 0.010, 500))))
        w = pd.Series(1.0, index=p.index)

        tear_sheet = pipe.evaluate_strategy(p, w)
        self.assertGreater(tear_sheet.cagr, 0.0)
        self.assertGreater(tear_sheet.sharpe_ratio, 0.5)

        readiness = pipe.score_deployment_readiness(tear_sheet, dsr_p_value=0.98)
        self.assertGreaterEqual(readiness.readiness_score, 80)
        self.assertTrue(readiness.is_deployable)

    def test_nexus_master_orchestrator(self):
        orchestrator = NexusMasterOrchestrator(initial_capital_usd=10_000_000.0)
        res = orchestrator.build_cross_asset_portfolio(n_days=500, seed=42)
        self.assertGreater(res.sharpe_ratio, 1.0)
        self.assertGreater(res.cagr, 0.05)
        self.assertEqual(len(res.asset_allocations), 10)
        self.assertGreater(len(res.stress_test_results), 2)
        self.assertIn("CAGR (Annual Compound Growth)", res.summary_table["Metric"].values)


if __name__ == "__main__":
    unittest.main()
''')

print("Applied fixes!")
