"""
Comprehensive Unit Tests for AI & Alternative Data Engine (Tracks 5, 7, 8, 10).
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
from nexus_quant.ai_alternative_data.ml_predictor import get_weights_ffd, frac_diff_fixed_width


class TestAIAlternativeData(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
        high = close * (1.0 + np.random.uniform(0.002, 0.015, n))
        low = close * (1.0 - np.random.uniform(0.002, 0.015, n))
        open_p = (high + low) / 2.0
        self.ohlc = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.uniform(1e6, 5e6, n),
        }, index=dates)

    def test_ffd_weights_and_convergence(self):
        w = get_weights_ffd(d=0.4, threshold=1e-4)
        self.assertGreater(len(w), 5)
        self.assertAlmostEqual(w[-1], 1.0, places=4)
        self.assertTrue(np.all(np.abs(w) < 1.01))

    def test_feature_engineering_and_fracdiff(self):
        fe = FinancialFeatureEngineer(frac_diff_d=0.35)
        X, y = fe.engineer_features(self.ohlc)
        self.assertGreaterEqual(len(X), 100)
        self.assertIn("frac_diff", X.columns)
        self.assertIn("rsi_14", X.columns)
        self.assertIn("bollinger_z", X.columns)
        self.assertIn("vol_21d", X.columns)
        self.assertIn("parkinson_vol", X.columns)
        self.assertIn("garman_klass_vol", X.columns)
        self.assertEqual(len(X), len(y))

    def test_purged_cv_splits(self):
        fe = FinancialFeatureEngineer(frac_diff_d=0.35)
        X, y = fe.engineer_features(self.ohlc)
        cv = PurgedTimeSeriesSplit(n_splits=4, purge_window=4, embargo_window=4)
        splits = cv.split(X)
        self.assertEqual(len(splits), 4)
        for tr_idx, te_idx in splits:
            self.assertGreater(len(tr_idx), 0)
            self.assertGreater(len(te_idx), 0)
            self.assertEqual(len(set(tr_idx).intersection(set(te_idx))), 0)

    def test_ml_predictor_performance(self):
        fe = FinancialFeatureEngineer(frac_diff_d=0.35)
        X, y = fe.engineer_features(self.ohlc)
        predictor = MLReturnPredictor(alpha=1.0, n_splits=3, purge_window=3)
        res = predictor.fit_predict_cv(X, y)
        self.assertIsInstance(res.information_coefficient, float)
        self.assertGreaterEqual(res.directional_hit_rate, 0.0)
        self.assertLessEqual(res.directional_hit_rate, 1.0)
        self.assertGreater(len(res.feature_importances), 5)
        self.assertIn("Financial ML Performance", res.summary())

    def test_supply_chain_network_and_pagerank(self):
        net = SupplyChainNetwork.create_default_network()
        cent = net.get_centrality_table()
        self.assertEqual(len(cent), len(net.tickers))
        self.assertAlmostEqual(cent["PageRank_Centrality"].sum(), 1.0, places=3)
        self.assertIn("AAPL", cent["Ticker"].values)

    def test_supply_chain_gnn_strategy(self):
        net = SupplyChainNetwork.create_default_network()
        p_dict = {t: 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 200))) for t in net.tickers}
        p_df = pd.DataFrame(p_dict)
        g_alpha = SupplyChainGraphAlpha(net, lead_lag_window=3)
        res = g_alpha.backtest_strategy(p_df)
        self.assertIsInstance(res.sharpe_ratio, float)
        self.assertGreater(len(res.equity_curve), 50)
        self.assertIn("Supply-Chain GNN Alpha", res.summary())

    def test_sec_semantic_drift_and_sentiment(self):
        engine = SemanticDriftEngine(high_drift_threshold=0.15, lazy_threshold=0.04)
        doc_prior = {"mda": "The company had record revenue growth and stable operational margins.", "risk_factors": "General market risks apply to our retail products."}
        doc_curr_lazy = {"mda": "The company had record revenue growth and stable operational margins.", "risk_factors": "General market risks apply to our retail products."}
        doc_curr_drift = {"mda": "The company faces substantial litigation, regulatory breach, and margin impairment.", "risk_factors": "Severe material weakness in controls and cybersecurity breach."}

        rep_lazy = engine.analyze_filing_pair("MSFT", 2023, doc_prior, doc_curr_lazy)
        rep_drift = engine.analyze_filing_pair("HIGH_RISK_CO", 2023, doc_prior, doc_curr_drift)

        self.assertAlmostEqual(rep_lazy.cosine_drift_total, 0.0, places=2)
        self.assertGreater(rep_drift.cosine_drift_total, 0.15)
        self.assertEqual(rep_lazy.category, "LAZY_DISCLOSURE")
        self.assertEqual(rep_drift.category, "HIGH_DRIFT")
        self.assertLess(rep_drift.sentiment_score, 0.0)

    def test_lazy_prices_dollar_neutrality(self):
        engine = SemanticDriftEngine()
        doc_prior = {"mda": "Stable earnings.", "risk_factors": "Normal competition."}
        doc_lazy = {"mda": "Stable earnings.", "risk_factors": "Normal competition."}
        doc_drift = {"mda": "Catastrophic loss and litigation.", "risk_factors": "Breach."}

        r1 = engine.analyze_filing_pair("CO_A", 2023, doc_prior, doc_lazy)
        r2 = engine.analyze_filing_pair("CO_B", 2023, doc_prior, doc_drift)
        df = pd.DataFrame([r1.__dict__, r2.__dict__])

        strat = LazyPricesStrategy(quantile_cutoff=0.5)
        pos = strat.generate_positions(df)
        self.assertAlmostEqual(pos["Weight"].sum(), 0.0, places=4)
        self.assertIn("LONG (LAZY_ALPHA)", pos["Recommendation"].values)
        self.assertIn("SHORT (HIGH_DRIFT_RISK)", pos["Recommendation"].values)

    def test_sentiment_fear_greed_and_regimes(self):
        engine = MultiSourceSentimentEngine()
        fgi = engine.compute_fear_greed_index(self.ohlc["Close"], self.ohlc["Volume"])
        self.assertGreaterEqual(fgi.composite_index.min(), 0.0)
        self.assertLessEqual(fgi.composite_index.max(), 100.0)
        self.assertIn(fgi.regimes.iloc[-1], ["EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"])
        self.assertIn("Fear & Greed Index Snapshot", fgi.summary())

    def test_multi_agent_hedge_fund_swarm(self):
        swarm = MultiAgentHedgeFundSwarm(target_vol_annual=0.12)
        memo = swarm.run_investment_committee(macro_signal=0.8, crypto_onchain=0.7, sentiment_fgi=65.0)
        self.assertAlmostEqual(sum(memo.optimal_weights.values()), 1.0, places=4)
        self.assertGreater(memo.expected_portfolio_return, 0.0)
        self.assertGreater(memo.expected_portfolio_volatility, 0.0)
        self.assertEqual(len(memo.agent_views), 3)
        self.assertIn("# Investment Committee Memorandum", memo.to_markdown())


if __name__ == "__main__":
    unittest.main()
