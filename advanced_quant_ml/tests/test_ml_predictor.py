"""Unit tests for Module 24 (Machine Learning Return Predictor)."""

import unittest
import numpy as np
import pandas as pd
from advanced_quant_ml.ml_predictor import (
    FinancialFeatureEngineer,
    MLReturnPredictor,
    PurgedTimeSeriesSplit,
    MLModelResult,
)
from advanced_quant_ml.ml_predictor.features import frac_diff_ffd, get_ffd_weights, find_min_d


class TestFinancialFeatureEngineer(unittest.TestCase):
    """Tests for FinancialFeatureEngineer and Fractional Differencing."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        
        # Simulated OHLC
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
        high = close * (1.0 + np.abs(np.random.normal(0, 0.008, n)))
        low = close * (1.0 - np.abs(np.random.normal(0, 0.008, n)))
        open_p = close * (1.0 + np.random.normal(0, 0.004, n))
        volume = np.random.uniform(1e6, 5e6, n)

        self.df_ohlc = pd.DataFrame(
            {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

    def test_ffd_weights_properties(self):
        w = get_ffd_weights(d=0.5, thres=1e-4)
        self.assertEqual(w[0], 1.0)
        self.assertLess(w[1], 0.0)
        self.assertTrue(np.all(np.abs(w[1:]) < 1.0))

    def test_fractional_differentiation_shape(self):
        close = self.df_ohlc["Close"]
        fd = frac_diff_ffd(close, d=0.4)
        self.assertEqual(len(fd), len(close))
        self.assertTrue(np.isnan(fd.iloc[0]))
        self.assertFalse(np.isnan(fd.iloc[-1]))

    def test_find_min_d(self):
        close = self.df_ohlc["Close"]
        min_d = find_min_d(close, p_threshold=0.10, d_step=0.10)
        self.assertGreaterEqual(min_d, 0.0)
        self.assertLessEqual(min_d, 1.0)

    def test_feature_engineering_pipeline(self):
        fe = FinancialFeatureEngineer(
            momentum_windows=[1, 5, 21],
            volatility_windows=[10, 21],
            frac_diff_d=0.4,
            target_horizon=1,
        )
        X, y = fe.engineer_features(self.df_ohlc, include_target=True)

        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertTrue("return_1d" in X.columns)
        self.assertTrue("vol_cc_21d" in X.columns)
        self.assertTrue("rsi_14" in X.columns)
        self.assertTrue("bollinger_zscore_20" in X.columns)
        self.assertTrue("frac_diff_d0.40" in X.columns)
        self.assertFalse(X.isnull().any().any())
        self.assertFalse(y.isnull().any())


class TestMLReturnPredictor(unittest.TestCase):
    """Tests for MLReturnPredictor and PurgedTimeSeriesSplit."""

    def setUp(self):
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        
        # Synthetic features and target with positive true signal
        f1 = np.random.normal(0, 1, n)
        f2 = np.random.normal(0, 1, n)
        f3 = np.random.normal(0, 1, n)
        
        y = 0.08 * f1 - 0.05 * f2 + np.random.normal(0, 0.02, n)

        self.X = pd.DataFrame({"signal_alpha": f1, "signal_beta": f2, "noise_gamma": f3}, index=dates)
        self.y = pd.Series(y, index=dates, name="target_return")

    def test_purged_time_series_split(self):
        splitter = PurgedTimeSeriesSplit(n_splits=4, purge_window=3, embargo_window=3)
        splits = list(splitter.split(self.X, self.y))

        self.assertEqual(len(splits), 4)
        for train_idx, test_idx in splits:
            # Verify strict train/test separation
            self.assertEqual(len(set(train_idx).intersection(set(test_idx))), 0)
            self.assertGreater(len(train_idx), 50)
            self.assertGreater(len(test_idx), 50)

    def test_ml_return_predictor_ridge_cv(self):
        predictor = MLReturnPredictor(model_type="ridge", alpha=1.0, n_splits=4)
        res = predictor.fit_predict_cv(self.X, self.y)

        self.assertIsInstance(res, MLModelResult)
        self.assertEqual(len(res.predictions), len(self.X))
        self.assertTrue(np.isfinite(res.information_coefficient))
        self.assertTrue(np.isfinite(res.rank_ic))
        self.assertGreater(res.information_coefficient, 0.0)  # Positive true signal
        self.assertGreater(res.directional_hit_rate, 0.50)
        self.assertTrue("signal_alpha" in res.feature_importance)
        
        tbl = res.summary_table()
        self.assertIsInstance(tbl, pd.DataFrame)

    def test_ml_return_predictor_lasso_cv(self):
        predictor = MLReturnPredictor(model_type="lasso", alpha=0.5, n_splits=4)
        res = predictor.fit_predict_cv(self.X, self.y)
        self.assertIsInstance(res, MLModelResult)
        self.assertTrue(np.isfinite(res.information_coefficient))


if __name__ == "__main__":
    unittest.main()
