"""Unit tests for Project 31: Financial LLM & SEC 10-K Semantic Drift Alpha Engine."""

import unittest
import numpy as np
import pandas as pd

from genai_advanced_quant.data.loader import generate_synthetic_sec_filings
from genai_advanced_quant.llm_alpha.semantic_drift import (
    SemanticDriftEngine,
    SimpleTfidfVectorizer,
    LazyPricesStrategy,
    clean_and_tokenize,
)


class TestSemanticDrift(unittest.TestCase):
    """Validates text tokenization, TF-IDF vectorization, Cosine dissimilarity, and Lazy Prices alpha."""
    
    def setUp(self):
        self.engine = SemanticDriftEngine()
        self.filings = generate_synthetic_sec_filings(seed=42)
        
    def test_tokenization(self):
        text = "The company's revenue increased by 15% due to robust AI demand, despite supply chain disruptions!"
        tokens = clean_and_tokenize(text)
        self.assertIn("revenue", tokens)
        self.assertIn("increased", tokens)
        self.assertIn("robust", tokens)
        self.assertNotIn("the", tokens)
        
    def test_tfidf_vectorizer(self):
        docs = [
            "artificial intelligence GPU computing hyperscale datacenter revenue growth",
            "commercial bank consumer lending credit loss provision net interest margin",
            "electric vehicle automotive manufacturing gross margin battery capacity"
        ]
        vec = SimpleTfidfVectorizer()
        matrix = vec.fit_transform(docs)
        self.assertEqual(matrix.shape[0], 3)
        self.assertGreater(matrix.shape[1], 5)
        # Verify unit L2 norm
        for row in matrix:
            norm_val = np.linalg.norm(row)
            self.assertAlmostEqual(norm_val, 1.0, places=5)
            
    def test_cosine_dissimilarity_bounds(self):
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([1.0, 0.0, 0.0])
        vec_c = np.array([0.0, 1.0, 0.0])
        
        # Identical vectors -> dissimilarity = 0.0
        self.assertAlmostEqual(self.engine.cosine_dissimilarity(vec_a, vec_b), 0.0)
        # Orthogonal vectors -> dissimilarity = 1.0
        self.assertAlmostEqual(self.engine.cosine_dissimilarity(vec_a, vec_c), 1.0)
        
    def test_loughran_mcdonald_sentiment(self):
        pos_tokens = ["growth", "profit", "record", "innovative", "resilient"]
        neg_tokens = ["loss", "impairment", "lawsuit", "default", "crisis", "adverse"]
        
        pos_res = self.engine.loughran_mcdonald_sentiment(pos_tokens)
        neg_res = self.engine.loughran_mcdonald_sentiment(neg_tokens)
        
        self.assertGreater(pos_res["sentiment"], 0.5)
        self.assertLess(neg_res["sentiment"], -0.5)
        self.assertGreater(neg_res["negative_pct"], 50.0)
        
    def test_cross_sectional_universe_drift(self):
        universe_df = self.engine.analyze_universe(self.filings, target_year=2023)
        self.assertFalse(universe_df.empty)
        self.assertIn("Ticker", universe_df.columns)
        self.assertIn("Cosine_Drift_Total", universe_df.columns)
        self.assertIn("Sentiment_Score", universe_df.columns)
        self.assertIn("Category", universe_df.columns)
        
        # Ensure NVDA or META have significant drift in 2023
        high_drift_tickers = universe_df[universe_df["Category"] == "HIGH_DRIFT"]["Ticker"].tolist()
        self.assertTrue(len(high_drift_tickers) >= 1)
        
    def test_lazy_prices_strategy(self):
        universe_df = self.engine.analyze_universe(self.filings, target_year=2023)
        strat = LazyPricesStrategy(quantile_cutoff=0.30)
        positions = strat.generate_positions(universe_df)
        
        self.assertFalse(positions.empty)
        self.assertIn("Weight", positions.columns)
        # Dollar neutrality check
        total_net_weight = positions["Weight"].sum()
        self.assertAlmostEqual(total_net_weight, 0.0, places=5)
        
        # Ensure Long positions have lower drift than Short positions
        long_drifts = positions[positions["Weight"] > 0]["Cosine_Drift_Total"].mean()
        short_drifts = positions[positions["Weight"] < 0]["Cosine_Drift_Total"].mean()
        self.assertLess(long_drifts, short_drifts)


if __name__ == '__main__':
    unittest.main()
