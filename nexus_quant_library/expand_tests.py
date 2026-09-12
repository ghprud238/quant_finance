import os

def w(path, content):
    full = os.path.join('/working_dir/nexus_quant_platform', path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print('Expanded', path)

w('tests/test_ai_alt_data.py', '''"""
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
''')

w('tests/test_defi_prediction.py', '''"""
Comprehensive Unit Tests for DeFi AMMs, LVR, Perpetual Basis & Prediction Market Arbitrage (Tracks 9, 11).
"""

import unittest
import numpy as np
import pandas as pd

from nexus_quant.defi_prediction_markets import (
    ConcentratedLiquidityAMM,
    ConstantProductAMM,
    ImpermanentLossCalculator,
    LossVersusRebalancingEngine,
    PerpetualFundingEngine,
    CashAndCarryBasisTrader,
    PredictionMarketArbitrageEngine,
    BinaryOrderBook,
    OrderBookLevel,
)


class TestDeFiPrediction(unittest.TestCase):

    def test_uniswap_v2_constant_product(self):
        v2 = ConstantProductAMM(reserve_x=1000.0, reserve_y=3_000_000.0, fee_rate=0.003)
        self.assertAlmostEqual(v2.spot_price, 3000.0, places=2)
        swap = v2.swap_exact_in(amount_in_x=10.0)
        self.assertGreater(swap.amount_out, 0.0)
        self.assertLess(v2.spot_price, 3000.0)
        self.assertGreater(swap.price_impact_pct, 0.0)

    def test_uniswap_v3_concentrated_liquidity(self):
        v3 = ConcentratedLiquidityAMM(current_price=3000.0, fee_tier=0.003)
        pos = v3.mint_position(owner="Alice", price_lower=2500.0, price_upper=3500.0, amount_x=10.0, amount_y=30_000.0)
        self.assertGreater(pos.liquidity, 0.0)
        
        eff = v3.capital_efficiency_multiplier(2500.0, 3500.0)
        self.assertGreater(eff, 5.0)

        swap_x = v3.swap(amount_in=2.0, token_in="ETH")
        self.assertGreater(swap_x.amount_out, 0.0)
        self.assertLess(v3.current_price, 3000.0)

        swap_y = v3.swap(amount_in=10_000.0, token_in="USDC")
        self.assertGreater(swap_y.amount_out, 0.0)
        self.assertGreater(v3.current_price, 2900.0)

    def test_impermanent_loss_formulas(self):
        # 25% price increase -> IL ~ -0.6%
        il_v2 = ImpermanentLossCalculator.standard_cfmm_il(1.25)
        self.assertLess(il_v2, 0.0)
        self.assertAlmostEqual(il_v2, -0.006, places=2)

        # Concentrated IL within range
        il_v3 = ImpermanentLossCalculator.concentrated_liquidity_il(3000.0, 3500.0, 2500.0, 4000.0)
        self.assertLess(il_v3, 0.0)
        self.assertLess(il_v3, il_v2)  # Concentrated IL is magnified

    def test_lvr_engine_and_breakeven_vol(self):
        prices = pd.Series(3000.0 * np.exp(np.cumsum(np.random.normal(0, 0.005, 500))))
        volumes = pd.Series(np.random.uniform(500_000, 2_000_000, 500))
        lvr_engine = LossVersusRebalancingEngine(fee_rate=0.0030)
        sim = lvr_engine.simulate_lp_performance(prices, volumes, initial_capital_usd=100_000.0)
        self.assertGreater(sim.total_lvr_usd, 0.0)
        self.assertGreater(sim.total_fee_revenue_usd, 0.0)
        self.assertGreater(sim.breakeven_volatility_annual, 0.0)
        self.assertIn("Initial LP Capital", sim.summary_table["Metric"].values)

    def test_perpetual_funding_rate_clamps(self):
        fr_normal = PerpetualFundingEngine.calculate_funding_rate(0.0008, 0.0003)
        self.assertLessEqual(fr_normal, 0.0075)
        self.assertGreaterEqual(fr_normal, -0.0075)

        fr_extreme = PerpetualFundingEngine.calculate_funding_rate(0.025, 0.0003)
        self.assertAlmostEqual(fr_extreme, 0.0075, places=4)

        fr_neg = PerpetualFundingEngine.calculate_funding_rate(-0.025, 0.0003)
        self.assertAlmostEqual(fr_neg, -0.0075, places=4)

    def test_cash_and_carry_basis_trading(self):
        n = 300
        df = pd.DataFrame({
            "spot_price": 3000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))),
            "funding_rate": np.random.uniform(0.0001, 0.0005, n),
        })
        trader = CashAndCarryBasisTrader(initial_capital_usd=100_000.0)
        res = trader.backtest(df)
        self.assertGreater(res.final_equity_usd, res.initial_capital_usd)
        self.assertGreater(res.cagr, 0.0)
        self.assertGreater(res.sharpe_ratio, 0.0)
        self.assertLessEqual(res.max_drawdown_pct, 0.05)
        self.assertIn("Cash-and-Carry Delta-Neutral Performance", res.summary())

    def test_prediction_market_intra_venue_arbitrage(self):
        engine = PredictionMarketArbitrageEngine()
        book = BinaryOrderBook(
            venue="Polymarket",
            contract_id="ELECTION_YES",
            yes_bids=[OrderBookLevel(0.44, 5000)],
            yes_asks=[OrderBookLevel(0.45, 5000)],
            no_bids=[OrderBookLevel(0.47, 5000)],
            no_asks=[OrderBookLevel(0.48, 5000)],
        )
        intra = engine.check_intra_venue_arbitrage(book, target_size=1000.0)
        self.assertIsNotNone(intra)
        self.assertTrue(intra.is_executable)
        self.assertAlmostEqual(intra.gross_edge_pct, 7.0, places=1)
        self.assertGreater(intra.net_profit_usd, 50.0)

    def test_prediction_market_cross_venue_arbitrage(self):
        engine = PredictionMarketArbitrageEngine()
        book_a = BinaryOrderBook(
            venue="Polymarket",
            contract_id="FED_RATE",
            yes_bids=[OrderBookLevel(0.50, 5000)],
            yes_asks=[OrderBookLevel(0.52, 5000)],
            no_bids=[OrderBookLevel(0.46, 5000)],
            no_asks=[OrderBookLevel(0.48, 5000)],
        )
        book_b = BinaryOrderBook(
            venue="Kalshi",
            contract_id="FED_RATE",
            yes_bids=[OrderBookLevel(0.58, 5000)],
            yes_asks=[OrderBookLevel(0.60, 5000)],
            no_bids=[OrderBookLevel(0.38, 5000)],
            no_asks=[OrderBookLevel(0.40, 5000)],
        )
        arbs = engine.check_cross_venue_arbitrage(book_a, book_b, target_size=1000.0)
        self.assertGreater(len(arbs), 0)
        exec_arbs = [a for a in arbs if a.is_executable]
        self.assertGreater(len(exec_arbs), 0)
        self.assertAlmostEqual(exec_arbs[0].gross_edge_pct, 8.0, places=1)

    def test_kelly_criterion_sizing(self):
        engine = PredictionMarketArbitrageEngine()
        kelly = engine.calculate_kelly_fraction(true_win_prob=0.999, market_price=0.93, bankroll_usd=100_000.0)
        self.assertGreater(kelly.recommended_fraction, 0.0)
        self.assertLessEqual(kelly.recommended_fraction, 0.40)
        self.assertGreater(kelly.recommended_stake_usd, 0.0)
        self.assertGreater(kelly.expected_growth_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
''')

w('tests/test_validation_orchestrator.py', '''"""
Comprehensive Unit Tests for Validation Rigor, Deflated Sharpe & Master Orchestrator (Tracks 6, 11).
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

    def test_psr_and_non_normality(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        
        # High Sharpe over 1 year
        psr = dsr_calc.compute_psr(observed_sharpe=2.5, benchmark_sharpe=0.0, sample_length=252, skewness=-0.5, kurtosis=4.5)
        self.assertGreater(psr.psr_value, 0.95)
        self.assertTrue(psr.is_significant_95)

        # Low Sharpe
        psr_low = dsr_calc.compute_psr(observed_sharpe=0.4, benchmark_sharpe=0.0, sample_length=252)
        self.assertLess(psr_low.psr_value, 0.90)
        self.assertFalse(psr_low.is_significant_95)

    def test_expected_max_sharpe_multiple_testing(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        
        # 1 trial vs 1,000 trials
        em_1, _ = dsr_calc.expected_max_sharpe(num_trials=1, var_sharpe_trials=0.25)
        em_1000, n_eff = dsr_calc.expected_max_sharpe(num_trials=1000, var_sharpe_trials=0.25)
        
        self.assertEqual(em_1, 0.0)
        self.assertGreater(em_1000, 1.2)
        self.assertEqual(n_eff, 1000.0)

    def test_deflated_sharpe_ratio_audit(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        
        # Genuine Strategy: Sharpe 2.8 under 1,000 trials across 5 years (1260 days)
        dsr_pass = dsr_calc.compute_dsr(best_sharpe_ratio=2.8, num_trials=1000, sample_length=1260, skewness=-0.5, kurtosis=4.0)
        self.assertGreater(dsr_pass.deflated_sharpe_ratio, 0.95)
        self.assertTrue(dsr_pass.is_significant_95)
        self.assertIn("GENUINE ALPHA", dsr_pass.summary())

        # Overfitted Strategy: Sharpe 1.2 under 10,000 trials across 1 year (252 days)
        dsr_fail = dsr_calc.compute_dsr(best_sharpe_ratio=1.2, num_trials=10000, sample_length=252)
        self.assertLess(dsr_fail.deflated_sharpe_ratio, 0.95)
        self.assertFalse(dsr_fail.is_significant_95)
        self.assertIn("OVERFITTED", dsr_fail.summary())

    def test_min_track_record_length(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        min_trl = dsr_calc.min_track_record_length(observed_sharpe=2.0, benchmark_sharpe=0.0)
        self.assertGreater(min_trl, 0.0)
        self.assertLess(min_trl, 5.0)

    def test_validation_pipeline_and_readiness_checklist(self):
        np.random.seed(42)
        pipe = QuantResearchValidationPipeline(risk_free_rate=0.02, tx_cost_bps=5.0)
        p = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0015, 0.010, 500))))
        w = pd.Series(1.0, index=p.index)

        tear_sheet = pipe.evaluate_strategy(p, w)
        self.assertGreater(tear_sheet.cagr, 0.0)
        self.assertGreater(tear_sheet.sharpe_ratio, 0.5)
        self.assertGreater(tear_sheet.estimated_capacity_usd, 1_000_000.0)
        self.assertIn("CAGR (Annual Return)", tear_sheet.metrics_table["Metric"].values)

        readiness = pipe.score_deployment_readiness(tear_sheet, dsr_p_value=0.98)
        self.assertGreaterEqual(readiness.readiness_score, 80)
        self.assertTrue(readiness.is_deployable)
        self.assertEqual(len(readiness.failure_reasons), 0)

    def test_nexus_master_orchestrator_integration(self):
        orchestrator = NexusMasterOrchestrator(initial_capital_usd=10_000_000.0)
        res = orchestrator.build_cross_asset_portfolio(n_days=1000, seed=42)
        
        self.assertGreater(res.sharpe_ratio, 1.5)
        self.assertGreater(res.cagr, 0.08)
        self.assertLess(res.max_drawdown, 0.0)
        self.assertEqual(len(res.asset_allocations), 10)
        self.assertAlmostEqual(sum(res.asset_allocations.values()), 1.0, places=4)
        self.assertEqual(len(res.stress_test_results), 4)
        self.assertGreater(res.deflated_sharpe_ratio, 0.95)
        self.assertIn("Nexus 11-Track Master Portfolio", res.summary())


if __name__ == "__main__":
    unittest.main()
''')

print("Expanded all test suites!")
