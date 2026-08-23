"""
Unit tests for Market Regime Detection Models:
- GaussianHMMRegimeDetector
- TrendVolRegimeFilter
- GMMRegimeDetector
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure package is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from quant_foundations.regimes.hmm_model import (
    GaussianHMMRegimeDetector,
    REGIME_NAMES_2STATE,
    REGIME_NAMES_3STATE,
)
from quant_foundations.regimes.heuristic import TrendVolRegimeFilter
from quant_foundations.regimes.gmm_model import GMMRegimeDetector


class TestGaussianHMMRegimeDetector(unittest.TestCase):
    """Test suite for Gaussian Hidden Markov Model Regime Detector."""

    def setUp(self):
        """Generate synthetic return series with distinct regime switching dynamics."""
        np.random.seed(101)
        self.T = 1500
        self.dates = pd.date_range(start="2018-01-01", periods=self.T, freq="B")

        # 3 regimes:
        # State 0: Bear (mean = -0.0015, vol = 0.025)
        # State 1: Neutral (mean = 0.0002, vol = 0.012)
        # State 2: Bull (mean = 0.0012, vol = 0.007)
        self.true_means = [-0.0015, 0.0002, 0.0012]
        self.true_vols = [0.025, 0.012, 0.007]

        # Transition matrix (sticky regimes)
        self.A = np.array([
            [0.92, 0.06, 0.02],
            [0.05, 0.90, 0.05],
            [0.02, 0.06, 0.92],
        ])

        states = [0]
        returns = []
        for t in range(self.T):
            cur_state = states[-1]
            ret = np.random.normal(self.true_means[cur_state], self.true_vols[cur_state])
            returns.append(ret)
            if t < self.T - 1:
                next_state = np.random.choice(3, p=self.A[cur_state])
                states.append(next_state)

        self.true_states = np.array(states)
        self.returns_s = pd.Series(returns, index=self.dates, name="Returns")

    def test_fit_2state_hmm(self):
        """Test 2-state HMM fitting and consistent state ordering (Bear=0, Bull=1)."""
        hmm = GaussianHMMRegimeDetector(n_states=2, max_iter=100, random_state=42)
        hmm.fit(self.returns_s)

        self.assertTrue(hmm.is_fitted)
        self.assertEqual(hmm.n_states, 2)
        self.assertEqual(hmm.regime_names_, REGIME_NAMES_2STATE)

        # State 0 (Bear) mean must be less than State 1 (Bull) mean
        self.assertLess(hmm.means_[0, 0], hmm.means_[1, 0])

        # Check transition matrix shape and row sums = 1
        trans_mat = hmm.trans_mat_
        self.assertEqual(trans_mat.shape, (2, 2))
        np.testing.assert_allclose(np.sum(trans_mat, axis=1), np.ones(2), rtol=1e-5)

    def test_fit_3state_hmm(self):
        """Test 3-state HMM fitting and consistent state ordering (Bear=0, Neutral=1, Bull=2)."""
        hmm = GaussianHMMRegimeDetector(n_states=3, max_iter=120, random_state=42)
        hmm.fit(self.returns_s)

        self.assertTrue(hmm.is_fitted)
        self.assertEqual(hmm.n_states, 3)
        self.assertEqual(hmm.regime_names_, REGIME_NAMES_3STATE)

        # State means should be strictly monotonic: Bear < Neutral < Bull
        self.assertLess(hmm.means_[0, 0], hmm.means_[1, 0])
        self.assertLess(hmm.means_[1, 0], hmm.means_[2, 0])

        # Check posterior probabilities shape and row sums = 1
        probs = hmm.predict_proba()
        self.assertEqual(probs.shape, (self.T, 3))
        np.testing.assert_allclose(probs.sum(axis=1).values, np.ones(self.T), rtol=1e-4)

    def test_viterbi_decoding_and_prediction(self):
        """Test Viterbi decoding sequence."""
        hmm = GaussianHMMRegimeDetector(n_states=3, max_iter=100, random_state=42).fit(self.returns_s)
        regimes = hmm.predict()

        self.assertEqual(len(regimes), self.T)
        self.assertTrue(set(regimes.unique()).issubset(set(REGIME_NAMES_3STATE)))

        # Predict on new slice
        slice_preds = hmm.predict(self.returns_s.iloc[:50])
        self.assertEqual(len(slice_preds), 50)

    def test_stationary_distribution_and_durations(self):
        """Test stationary distribution satisfies pi * P = pi and expected duration calculation."""
        hmm = GaussianHMMRegimeDetector(n_states=3, max_iter=100, random_state=42).fit(self.returns_s)
        pi = hmm.stationary_dist_
        P = hmm.trans_mat_

        # pi * P = pi
        np.testing.assert_allclose(pi @ P, pi, atol=1e-4)
        self.assertAlmostEqual(np.sum(pi), 1.0, places=5)

        # Expected duration = 1 / (1 - P_ii)
        for i, name in enumerate(hmm.regime_names_):
            p_stay = P[i, i]
            expected = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf")
            self.assertAlmostEqual(hmm.expected_durations_[name], expected, places=5)
            self.assertGreater(hmm.expected_durations_[name], 1.0)

    def test_regime_conditional_metrics(self):
        """Test regime conditional statistics: Annualized Return, Volatility, Sharpe."""
        hmm = GaussianHMMRegimeDetector(n_states=3, max_iter=100, random_state=42).fit(self.returns_s)
        metrics_df = hmm.regime_metrics(risk_free_rate=0.02)

        self.assertEqual(len(metrics_df), 3)
        expected_cols = [
            "Observations", "Frequency_Pct", "Stationary_Prob",
            "Expected_Duration_Days", "Annualized_Return",
            "Annualized_Volatility", "Sharpe_Ratio"
        ]
        for col in expected_cols:
            self.assertIn(col, metrics_df.columns)

        # Bear regime should have lower return and higher vol than Bull
        bear_ret = metrics_df.loc["Bear", "Annualized_Return"]
        bull_ret = metrics_df.loc["Bull", "Annualized_Return"]
        self.assertLess(bear_ret, bull_ret)

        bear_vol = metrics_df.loc["Bear", "Annualized_Volatility"]
        bull_vol = metrics_df.loc["Bull", "Annualized_Volatility"]
        self.assertGreater(bear_vol, bull_vol)


class TestTrendVolRegimeFilter(unittest.TestCase):
    """Test suite for TrendVolRegimeFilter."""

    def setUp(self):
        """Generate synthetic price series with upward trend, downward trend, and high/low vol."""
        np.random.seed(202)
        n_days = 600
        dates = pd.date_range("2021-01-01", periods=n_days, freq="B")

        # Create price path: 200 days up (low vol), 200 days down (high vol), 200 days flat
        ret1 = np.random.normal(0.001, 0.008, 200)
        ret2 = np.random.normal(-0.0015, 0.025, 200)
        ret3 = np.random.normal(0.0001, 0.012, 200)
        rets = np.concatenate([ret1, ret2, ret3])

        prices = 100.0 * np.exp(np.cumsum(rets))
        self.prices_s = pd.Series(prices, index=dates, name="Price")

        self.filter = TrendVolRegimeFilter(
            sma_window=50,  # smaller window for test
            vol_window=20,
            vol_threshold="median",
        )

    def test_classification_logic(self):
        """Test rule-based classification into Bull, Bear, and Neutral."""
        result_df = self.filter.classify(self.prices_s)

        self.assertIn("Regime", result_df.columns)
        self.assertIn("Regime_Code", result_df.columns)
        self.assertIn("SMA_50", result_df.columns)
        self.assertIn("Rolling_Vol_20", result_df.columns)
        self.assertIn("Vol_Threshold", result_df.columns)

        # After warmup (index >= 50):
        warmup_df = result_df.dropna(subset=["SMA_50", "Rolling_Vol_20"])

        # Check Bull rules: Price > SMA and Vol <= Threshold
        bull_rows = warmup_df[warmup_df["Regime"] == "Bull"]
        if len(bull_rows) > 0:
            self.assertTrue((bull_rows["Price"] > bull_rows["SMA_50"]).all())
            self.assertTrue((bull_rows["Rolling_Vol_20"] <= bull_rows["Vol_Threshold"]).all())
            self.assertTrue((bull_rows["Regime_Code"] == 2).all())

        # Check Bear rules: Price < SMA and Vol > Threshold
        bear_rows = warmup_df[warmup_df["Regime"] == "Bear"]
        if len(bear_rows) > 0:
            self.assertTrue((bear_rows["Price"] < bear_rows["SMA_50"]).all())
            self.assertTrue((bear_rows["Rolling_Vol_20"] > bear_rows["Vol_Threshold"]).all())
            self.assertTrue((bear_rows["Regime_Code"] == 0).all())

    def test_regime_metrics(self):
        """Test regime performance metrics calculation."""
        metrics = self.filter.regime_metrics(self.prices_s)
        self.assertEqual(len(metrics), 3)
        self.assertIn("Bear", metrics.index)
        self.assertIn("Bull", metrics.index)
        self.assertIn("Neutral", metrics.index)
        self.assertIn("Annualized_Return", metrics.columns)
        self.assertIn("Annualized_Volatility", metrics.columns)
        self.assertIn("Avg_Duration_Days", metrics.columns)


class TestGMMRegimeDetector(unittest.TestCase):
    """Test suite for GMMRegimeDetector."""

    def setUp(self):
        """Generate synthetic return series."""
        np.random.seed(303)
        n = 800
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        # 3 clusters
        c1 = np.random.normal(0.0015, 0.008, 300)   # Bull
        c2 = np.random.normal(-0.002, 0.024, 250)   # Bear
        c3 = np.random.normal(0.0001, 0.012, 250)   # Neutral
        rets = np.concatenate([c1, c2, c3])
        self.returns_s = pd.Series(rets, index=dates, name="Return")

    def test_gmm_fit_and_clustering(self):
        """Test GMM fitting on (Return, Volatility) features."""
        gmm = GMMRegimeDetector(n_components=3, vol_window=15, max_iter=100, random_state=42)
        gmm.fit(self.returns_s)

        self.assertTrue(gmm.is_fitted)
        self.assertEqual(len(gmm.weights_), 3)
        self.assertAlmostEqual(float(np.sum(gmm.weights_)), 1.0, places=5)

        # Means should be sorted: Bear < Neutral < Bull
        self.assertLess(gmm.means_[0, 0], gmm.means_[1, 0])
        self.assertLess(gmm.means_[1, 0], gmm.means_[2, 0])

        # Check soft probabilities
        probs = gmm.predict_proba()
        self.assertEqual(probs.shape[1], 3)
        np.testing.assert_allclose(probs.sum(axis=1).values, np.ones(len(probs)), rtol=1e-4)

        # Check regime metrics
        metrics = gmm.regime_metrics()
        self.assertEqual(len(metrics), 3)
        self.assertIn("Annualized_Return", metrics.columns)
        self.assertIn("Annualized_Volatility", metrics.columns)


if __name__ == "__main__":
    unittest.main()
