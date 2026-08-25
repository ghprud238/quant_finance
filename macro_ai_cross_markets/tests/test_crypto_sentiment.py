"""Unit tests for Module 48: Social Media, News & Crypto Fear/Greed LLM Market Sentiment Engine."""

import unittest
import numpy as np
import pandas as pd

from macro_ai_cross_markets.crypto_sentiment.sentiment_engine import (
    MultiSourceSentimentEngine,
    SentimentAspect,
    FearGreedRegime,
    SentimentRecord,
    FearGreedComponents,
    LeadLagResult,
    SentimentStrategyResult,
)


class TestCryptoSentimentEngine(unittest.TestCase):
    """Validates multi-source sentiment scoring, ABSA, Fear & Greed Index, and lead-lag correlations."""

    def setUp(self):
        self.engine = MultiSourceSentimentEngine()

    def test_tokenization_and_aspect_classification(self):
        text_macro = "Inflation cooled down and GDP growth exceeded forecasts despite bond yield pressures."
        tokens_macro = self.engine.tokenize(text_macro)
        self.assertIn("inflation", tokens_macro)
        self.assertIn("gdp", tokens_macro)
        aspect_macro = self.engine.classify_aspect(tokens_macro)
        self.assertEqual(aspect_macro, SentimentAspect.MACRO)

        text_cb = "Fed chair Powell indicated a potential rate hike pause following the upcoming FOMC meeting."
        tokens_cb = self.engine.tokenize(text_cb)
        aspect_cb = self.engine.classify_aspect(tokens_cb)
        self.assertEqual(aspect_cb, SentimentAspect.CENTRAL_BANK)

        text_crypto = "Bitcoin hash rate touched an all time high as whale wallet accumulation surged onchain."
        tokens_crypto = self.engine.tokenize(text_crypto)
        aspect_crypto = self.engine.classify_aspect(tokens_crypto)
        self.assertEqual(aspect_crypto, SentimentAspect.CRYPTO)

        text_reg = "The SEC filed an enforcement lawsuit and issued a subpoena regarding compliance violations."
        tokens_reg = self.engine.tokenize(text_reg)
        aspect_reg = self.engine.classify_aspect(tokens_reg)
        self.assertEqual(aspect_reg, SentimentAspect.REGULATORY)

    def test_sentiment_polarity_and_negation(self):
        # Bullish sentiment
        res_pos = self.engine.analyze_text("Bullish breakout rally as institutional inflows surge to record high.")
        self.assertGreater(res_pos.polarity, 0.3)
        self.assertGreater(res_pos.confidence, 0.0)

        # Bearish sentiment
        res_neg = self.engine.analyze_text("Bearish market crash as massive liquidation cascade triggers panic dump.")
        self.assertLess(res_neg.polarity, -0.3)

        # Negation handling: "not bullish" should flip to negative
        res_negated = self.engine.analyze_text("Market indicators are not bullish after the recent breakdown.")
        self.assertLess(res_negated.polarity, 0.0)

    def test_ingest_documents(self):
        docs = [
            {"text": "Bitcoin rally surges past resistance with heavy buying momentum.", "source": "twitter", "timestamp": "2024-01-01"},
            {"text": "SEC regulatory lawsuit triggers market drop and concern.", "source": "news", "timestamp": "2024-01-02"},
            {"text": "Federal Reserve maintains interest rate pause amid inflation easing.", "source": "macro", "timestamp": "2024-01-03"}
        ]
        df = self.engine.ingest_documents(docs)
        self.assertEqual(len(df), 3)
        self.assertIn("Polarity", df.columns)
        self.assertIn("Aspect", df.columns)
        self.assertTrue(df.iloc[0]["Polarity"] > 0)
        self.assertTrue(df.iloc[1]["Polarity"] < 0)

    def test_fear_greed_index_reconstruction(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        # Bullish price series with rising momentum and low vol
        prices = pd.Series(40000.0 * np.cumprod(1.0 + np.random.normal(0.003, 0.02, 100)), index=dates)
        volumes = pd.Series(np.random.uniform(1e9, 5e9, 100), index=dates)
        vols = prices.pct_change().rolling(30).std() * np.sqrt(365)
        
        fgi_df = self.engine.compute_fear_greed_index(
            volatility_series=vols,
            price_series=prices,
            volume_series=volumes,
        )
        
        self.assertEqual(len(fgi_df), 100)
        self.assertIn("Composite_FGI", fgi_df.columns)
        self.assertIn("Regime", fgi_df.columns)
        self.assertTrue((fgi_df["Composite_FGI"] >= 0.0).all())
        self.assertTrue((fgi_df["Composite_FGI"] <= 100.0).all())

    def test_lead_lag_cross_correlation(self):
        dates = pd.date_range("2024-01-01", periods=150, freq="D")
        np.random.seed(42)
        
        # Create leading sentiment series
        sentiment = pd.Series(np.random.normal(0, 1, 150), index=dates)
        
        # Returns lag sentiment by 2 days + noise
        returns = pd.Series(0.0, index=dates)
        returns.iloc[2:] = 0.6 * sentiment.iloc[:-2].values + np.random.normal(0, 0.4, 148)

        res = self.engine.compute_lead_lag_correlation(sentiment, returns, max_lag=5)
        self.assertIsInstance(res, LeadLagResult)
        self.assertEqual(len(res.lags), 11)
        self.assertEqual(res.peak_lag, 2)
        self.assertTrue(res.is_sentiment_leading)
        self.assertGreater(res.peak_correlation, 0.3)

    def test_sentiment_trading_strategy_backtest(self):
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        np.random.seed(42)
        
        prices_df = pd.DataFrame({
            "BTC": 20000.0 * np.cumprod(1.0 + np.random.normal(0.001, 0.02, 200)),
            "ETH": 1500.0 * np.cumprod(1.0 + np.random.normal(0.0012, 0.025, 200)),
        }, index=dates)
        
        sentiment_df = pd.DataFrame({
            "BTC": np.random.uniform(20, 80, 200),
            "ETH": np.random.uniform(20, 80, 200),
        }, index=dates)

        res = self.engine.backtest_sentiment_strategy(
            prices_df=prices_df,
            sentiment_df=sentiment_df,
            threshold_long=55.0,
            threshold_short=45.0,
            rebalance_freq=3,
            transaction_cost_bps=5.0,
        )
        
        self.assertIsInstance(res, SentimentStrategyResult)
        self.assertEqual(len(res.cumulative_returns), 200)
        self.assertIsInstance(res.sharpe_ratio, float)
        self.assertIsInstance(res.max_drawdown, float)
        self.assertLessEqual(res.max_drawdown, 0.0)
        self.assertFalse(res.metrics_table.empty)


if __name__ == "__main__":
    unittest.main()
