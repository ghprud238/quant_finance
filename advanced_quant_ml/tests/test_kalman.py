"""Unit tests for Module 23 (Kalman Filter for Pairs Trading)."""

import unittest
import numpy as np
import pandas as pd
from advanced_quant_ml.kalman import (
    KalmanFilterPairs,
    KalmanFilterResult,
    KalmanPairsStrategy,
    KalmanStrategyResult,
)


class TestKalmanFilterPairs(unittest.TestCase):
    """Tests for KalmanFilterPairs and state-space recursive regression."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        self.dates = pd.date_range("2020-01-01", periods=n, freq="B")
        
        # Synthetic cointegrated pair: y_t = 10.0 + 2.5 * x_t + spread_t
        self.x = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, n)), index=self.dates)
        # Mean-reverting Ornstein-Uhlenbeck spread
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = 0.85 * spread[t - 1] + np.random.normal(0, 0.5)
        self.y = pd.Series(10.0 + 2.5 * self.x.values + spread, index=self.dates)

    def test_filter_output_structure(self):
        kf = KalmanFilterPairs(delta=1e-4, observation_cov="auto")
        res = kf.filter(self.y, self.x)

        self.assertIsInstance(res, KalmanFilterResult)
        self.assertEqual(len(res.beta), len(self.y))
        self.assertEqual(len(res.alpha), len(self.y))
        self.assertEqual(len(res.z_score), len(self.y))
        self.assertTrue(np.all(np.isfinite(res.beta.values)))
        self.assertTrue(np.all(np.isfinite(res.z_score.values)))
        
        df = res.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.y))

    def test_hedge_ratio_convergence(self):
        kf = KalmanFilterPairs(delta=1e-4, observation_cov="auto")
        res = kf.filter(self.y, self.x)

        # After initial warm-up, beta should converge close to true value 2.5
        tail_beta = res.beta.iloc[100:].mean()
        self.assertAlmostEqual(tail_beta, 2.5, delta=0.2)

    def test_z_score_properties(self):
        kf = KalmanFilterPairs(delta=1e-4, observation_cov="auto")
        res = kf.filter(self.y, self.x)

        # Standardized z-score should have approximately zero mean
        tail_z = res.z_score.iloc[100:]
        self.assertAlmostEqual(tail_z.mean(), 0.0, delta=0.5)
        self.assertGreater(tail_z.std(), 0.5)

    def test_kalman_pairs_strategy_backtest(self):
        strat = KalmanPairsStrategy(z_entry=1.5, z_exit=0.3, delta=1e-4, observation_cov="auto")
        res = strat.backtest(self.y, self.x)

        self.assertIsInstance(res, KalmanStrategyResult)
        self.assertEqual(len(res.positions), len(self.y))
        self.assertEqual(len(res.net_returns), len(self.y))
        self.assertTrue("CAGR" in res.metrics)
        self.assertTrue("Sharpe Ratio" in res.metrics)
        self.assertTrue("Max Drawdown" in res.metrics)
        self.assertLessEqual(res.metrics["Max Drawdown"], 0.0)

        tbl = res.summary_table()
        self.assertIsInstance(tbl, pd.DataFrame)
        self.assertGreater(len(tbl), 5)

    def test_input_validation(self):
        kf = KalmanFilterPairs()
        with self.assertRaises(ValueError):
            kf.filter(self.y, self.x.iloc[:-10])


if __name__ == "__main__":
    unittest.main()
