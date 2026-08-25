"""Unit tests for Project 46: Multilingual Central Bank LLM & Hawk/Dove Monetary Policy Indexer."""

import unittest
import numpy as np
import pandas as pd

from macro_ai_cross_markets.data.loader import generate_central_bank_statements, PolicyStatement
from macro_ai_cross_markets.central_bank_nlp.hawk_dove import (
    CentralBankStanceIndexer,
    TaylorRuleModel,
    clean_and_tokenize,
)


class TestCentralBankNLP(unittest.TestCase):
    """Validates Hawk/Dove score bounds, topic decomposition, multilingual handling, and Taylor rule residualization."""

    def setUp(self):
        self.indexer = CentralBankStanceIndexer()
        self.statements = generate_central_bank_statements(seed=42)
        self.taylor_model = TaylorRuleModel()

    def test_tokenization(self):
        text = "The FOMC decided to raise the target range for the federal funds rate!"
        tokens = clean_and_tokenize(text)
        self.assertIn("fomc", tokens)
        self.assertIn("decided", tokens)
        self.assertIn("raise", tokens)

    def test_hawkish_statement_scoring(self):
        # 2022 Fed rate hike statement
        fed_hike = next(s for s in self.statements if s.central_bank == "FED" and s.rate_decision == "HIKE" and "2022" in s.date)
        res = self.indexer.score_statement(fed_hike)

        self.assertGreater(res.hawk_dove_score, 0.20)
        self.assertGreater(res.hawkish_count, res.dovish_count)
        self.assertGreater(res.predicted_2y_yield_move_bps, 0.0)
        self.assertGreater(res.predicted_fx_move_pct, 0.0)

    def test_dovish_statement_scoring(self):
        # 2024 Fed rate cut statement
        fed_cut = next(s for s in self.statements if s.central_bank == "FED" and s.rate_decision == "CUT")
        res = self.indexer.score_statement(fed_cut)

        self.assertLess(res.hawk_dove_score, -0.10)
        self.assertLess(res.predicted_2y_yield_move_bps, 0.0)
        self.assertLess(res.predicted_fx_move_pct, 0.0)

    def test_multilingual_scoring(self):
        # Japanese BOJ, Portuguese BCB, Chinese PBOC
        boj_stmt = next(s for s in self.statements if s.central_bank == "BOJ" and s.rate_decision == "HOLD")
        res_boj = self.indexer.score_statement(boj_stmt)
        self.assertEqual(res_boj.language, "ja")

        bcb_stmt = next(s for s in self.statements if s.central_bank == "BCB" and s.rate_decision == "HIKE")
        res_bcb = self.indexer.score_statement(bcb_stmt)
        self.assertGreater(res_bcb.hawk_dove_score, 0.15)

    def test_corpus_analysis_dataframe(self):
        df = self.indexer.analyze_corpus(self.statements)
        self.assertFalse(df.empty)
        self.assertIn("Central_Bank", df.columns)
        self.assertIn("Hawk_Dove_Score", df.columns)
        self.assertIn("Stance", df.columns)
        self.assertTrue((df["Hawk_Dove_Score"].values >= -1.0).all())
        self.assertTrue((df["Hawk_Dove_Score"].values <= 1.0).all())

    def test_taylor_rule_model(self):
        res = self.taylor_model.calculate_taylor_rule(
            country="US",
            current_policy_rate=5.25,
            inflation_rate=3.2,
            inflation_target=2.0,
            r_star=0.5,
            output_gap_pct=0.5,
            hawk_dove_nlp_score=0.45,
        )
        self.assertEqual(res.country, "US")
        self.assertGreater(res.taylor_rule_rate, 2.0)
        self.assertIsInstance(res.residual_hawk_dove_score, float)
        self.assertTrue(-1.0 <= res.residual_hawk_dove_score <= 1.0)


if __name__ == '__main__':
    unittest.main()
