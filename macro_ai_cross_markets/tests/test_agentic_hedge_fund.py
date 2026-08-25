"""Unit tests for Project 50: Autonomous Multi-Agent Macroeconomic & Crypto Hedge Fund Swarm."""

import unittest
import numpy as np
import pandas as pd

from macro_ai_cross_markets.agentic_hedge_fund.hedge_fund_swarm import (
    MacroData,
    CryptoData,
    SentimentData,
    AgentView,
    MacroEconomistAgent,
    CryptoMicrostructureAgent,
    SentimentAlphaAgent,
    RiskAndExecutionPMAgent,
    InvestmentCommitteeMemo,
    MultiAgentHedgeFundSwarm,
    SwarmBacktestResult,
)


class TestAgenticHedgeFundSwarm(unittest.TestCase):
    """Validates multi-agent personas, Black-Litterman optimization, Committee memos, and backtesting."""

    def setUp(self):
        self.macro_data_growth = MacroData(
            gdp_growth_pct=2.8,
            cpi_inflation_pct=2.4,
            central_bank_rate_pct=4.5,
            yield_curve_slope_bps=35.0,
            dxy_index=102.5,
            vix_index=14.0,
        )
        self.macro_data_stagflation = MacroData(
            gdp_growth_pct=0.8,
            cpi_inflation_pct=4.2,
            central_bank_rate_pct=5.5,
            yield_curve_slope_bps=-15.0,
            dxy_index=106.0,
            vix_index=24.0,
        )
        self.crypto_data_bottom = CryptoData(
            btc_price=58000.0,
            eth_price=2800.0,
            mvrv_z_score=0.35,
            funding_rate_8h_pct=0.005,
            exchange_reserve_flow_usd=-2.5e8,
            defi_tvl_change_pct=6.5,
        )
        self.crypto_data_overheated = CryptoData(
            btc_price=98000.0,
            eth_price=4800.0,
            mvrv_z_score=3.85,
            funding_rate_8h_pct=0.045,
            exchange_reserve_flow_usd=3.2e8,
            defi_tvl_change_pct=-2.1,
        )
        self.sentiment_data_fear = SentimentData(
            fear_and_greed_index=18.0,
            news_sentiment_score=-0.45,
            social_media_bull_bear_ratio=0.65,
            retail_put_call_ratio=1.25,
        )
        self.sentiment_data_greed = SentimentData(
            fear_and_greed_index=84.0,
            news_sentiment_score=0.75,
            social_media_bull_bear_ratio=2.40,
            retail_put_call_ratio=0.52,
        )
        self.swarm = MultiAgentHedgeFundSwarm()

    def test_macro_economist_agent_growth(self):
        agent = MacroEconomistAgent()
        regime, views = agent.evaluate(self.macro_data_growth)
        self.assertEqual(regime, "DISINFLATIONARY_GROWTH")
        self.assertGreater(len(views), 0)
        eq_views = [v for v in views if v.asset == "Global_Equities"]
        self.assertEqual(len(eq_views), 1)
        self.assertEqual(eq_views[0].direction, "BULLISH")
        self.assertGreater(eq_views[0].conviction, 0.70)

    def test_macro_economist_agent_stagflation(self):
        agent = MacroEconomistAgent()
        regime, views = agent.evaluate(self.macro_data_stagflation)
        self.assertEqual(regime, "STAGFLATION")
        comm_views = [v for v in views if v.asset == "Commodities"]
        self.assertEqual(comm_views[0].direction, "BULLISH")
        eq_views = [v for v in views if v.asset == "Global_Equities"]
        self.assertEqual(eq_views[0].direction, "BEARISH")

    def test_crypto_microstructure_agent_bottom_and_top(self):
        agent = CryptoMicrostructureAgent()
        regime_bot, views_bot = agent.evaluate(self.crypto_data_bottom)
        self.assertEqual(regime_bot, "ACCUMULATION_BOTTOM")
        self.assertEqual(views_bot[0].direction, "BULLISH")
        self.assertGreater(views_bot[0].conviction, 0.80)

        regime_top, views_top = agent.evaluate(self.crypto_data_overheated)
        self.assertEqual(regime_top, "EUPHORIC_OVERHEATED")
        self.assertEqual(views_top[0].direction, "BEARISH")

    def test_sentiment_alpha_agent_contrarian(self):
        agent = SentimentAlphaAgent()
        regime_fear, views_fear = agent.evaluate(self.sentiment_data_fear)
        self.assertEqual(regime_fear, "EXTREME_FEAR_CONTRARIAN_BULLISH")
        self.assertTrue(any(v.direction == "BULLISH" for v in views_fear))

        regime_greed, views_greed = agent.evaluate(self.sentiment_data_greed)
        self.assertEqual(regime_greed, "EXTREME_GREED_CONTRARIAN_BEARISH")
        self.assertTrue(any(v.direction == "BEARISH" for v in views_greed))

    def test_pm_agent_black_litterman_optimization(self):
        pm = RiskAndExecutionPMAgent()
        all_views = [
            AgentView(
                asset="Global_Equities",
                direction="BULLISH",
                expected_return_annual=0.15,
                conviction=0.85,
                thesis="Strong earnings momentum",
                agent_name="MacroEconomist"
            ),
            AgentView(
                asset="Crypto_Assets",
                direction="BULLISH",
                expected_return_annual=0.30,
                conviction=0.80,
                thesis="On-chain accumulation bottom",
                agent_name="CryptoSpecialist"
            ),
            AgentView(
                asset="Sovereign_Bonds",
                direction="NEUTRAL",
                expected_return_annual=0.04,
                conviction=0.50,
                thesis="Neutral duration",
                agent_name="MacroEconomist"
            ),
        ]
        weights, post_returns, risk_metrics, debates = pm.reconcile_and_optimize(
            agent_views=all_views,
            date="2026-08-25"
        )
        
        # Verify budget constraint: sum(w_i) == 1.0
        total_w = sum(weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=5)
        
        # Verify long-only constraint and max asset weight bounds
        for asset, w in weights.items():
            self.assertGreaterEqual(w, -1e-6)
            self.assertLessEqual(w, pm.DEFAULT_MAX_WEIGHTS[asset] + 1e-4)
            
        # Verify risk metrics
        self.assertIn("portfolio_volatility_ann", risk_metrics)
        self.assertIn("var_95_daily", risk_metrics)
        self.assertIn("effective_n_assets", risk_metrics)
        self.assertGreater(risk_metrics["portfolio_volatility_ann"], 0.0)
        self.assertGreater(risk_metrics["effective_n_assets"], 1.0)

    def test_investment_committee_memo_generation(self):
        memo = self.swarm.conduct_investment_committee(
            macro_data=self.macro_data_growth,
            crypto_data=self.crypto_data_bottom,
            sentiment_data=self.sentiment_data_fear,
            date="2026-08-25",
        )
        self.assertIsInstance(memo, InvestmentCommitteeMemo)
        self.assertEqual(memo.date, "2026-08-25")
        self.assertIn("Global_Equities", memo.recommended_weights)
        self.assertIn("Crypto_Assets", memo.recommended_weights)
        
        # Test markdown generation
        md_text = memo.to_markdown()
        self.assertIn("# 🏛️ INVESTMENT COMMITTEE MEMORANDUM", md_text)
        self.assertIn("Global_Equities", md_text)
        self.assertIn("Target Annualized Volatility", md_text)
        
        # Test dataframe format
        df = memo.to_dataframe()
        self.assertFalse(df.empty)
        self.assertIn("Asset_Class", df.columns)
        self.assertIn("Target_Weight", df.columns)

    def test_multi_agent_swarm_backtest(self):
        dates = pd.date_range("2022-01-01", "2024-12-31", freq="B")
        np.random.seed(42)
        n_days = len(dates)
        
        returns_df = pd.DataFrame({
            "Global_Equities": np.random.normal(0.0004, 0.010, n_days),
            "Sovereign_Bonds": np.random.normal(0.0001, 0.005, n_days),
            "Commodities": np.random.normal(0.0003, 0.012, n_days),
            "Crypto_Assets": np.random.normal(0.0010, 0.035, n_days),
            "Cash_and_FX": np.full(n_days, 0.045 / 252.0),
        }, index=dates)
        
        res = self.swarm.backtest(
            multi_asset_returns_df=returns_df,
            rebalance_freq_days=21,
            initial_capital=10_000_000.0,
        )
        
        self.assertIsInstance(res, SwarmBacktestResult)
        self.assertEqual(len(res.equity_curve), n_days)
        self.assertEqual(res.weights_df.shape[0], n_days)
        self.assertIn("CAGR", res.metrics)
        self.assertIn("Sharpe_Ratio", res.metrics)
        self.assertIn("Max_Drawdown", res.metrics)
        
        summary = res.summary_table()
        self.assertFalse(summary.empty)
        self.assertIn("Metric", summary.columns)
        self.assertIn("Value", summary.columns)


if __name__ == "__main__":
    unittest.main()
