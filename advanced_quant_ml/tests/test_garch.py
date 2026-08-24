"""Unit tests for GARCH and GJR-GARCH Volatility Modeling (Module 21)."""

import unittest
import numpy as np
import pandas as pd
from advanced_quant_ml.garch import GARCHModel, GARCHFitResult, GARCHForecastResult
from advanced_quant_ml.data import load_equity_returns


class TestGARCHModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # Synthetic ARCH/GARCH returns
        n = 1000
        cls.returns = np.random.normal(0.0005, 0.015, n)
        cls.dates = pd.date_range("2020-01-01", periods=n, freq="B")
        cls.series = pd.Series(cls.returns, index=cls.dates)

    def test_garch_fit_basic(self):
        model = GARCHModel(model_type="GARCH")
        res = model.fit(self.series)

        self.assertIsInstance(res, GARCHFitResult)
        self.assertIn("omega", res.params)
        self.assertIn("alpha", res.params)
        self.assertIn("beta", res.params)
        self.assertGreater(res.params["omega"], 0.0)
        self.assertGreaterEqual(res.params["alpha"], 0.0)
        self.assertGreaterEqual(res.params["beta"], 0.0)
        self.assertLess(res.persistence, 1.0)
        self.assertGreater(res.unconditional_volatility_ann, 0.0)

    def test_gjr_garch_asymmetry(self):
        model = GARCHModel(model_type="GJR-GARCH")
        res = model.fit(self.series)

        self.assertIn("gamma", res.params)
        self.assertLess(res.persistence, 1.0)
        self.assertGreater(res.half_life_days, 0.0)
        self.assertEqual(len(res.conditional_volatility), len(self.series))
        self.assertEqual(len(res.standardized_residuals), len(self.series))

    def test_garch_multi_step_forecast(self):
        model = GARCHModel(model_type="GJR-GARCH")
        model.fit(self.series)
        fc = model.forecast(horizon=30)

        self.assertIsInstance(fc, GARCHForecastResult)
        self.assertEqual(len(fc.daily_variance_forecast), 30)
        self.assertEqual(len(fc.annualized_volatility_forecast), 30)
        self.assertEqual(len(fc.cumulative_annualized_volatility), 30)
        self.assertTrue(np.all(fc.daily_variance_forecast > 0))

        # Check long-term convergence to unconditional variance
        # As h -> infty, forecast variance should converge to unconditional variance
        long_fc = model.forecast(horizon=500)
        diff_end = abs(long_fc.daily_variance_forecast[-1] - model.fit_result.unconditional_variance)
        self.assertLess(diff_end, 1e-4)

    def test_forecast_dataframe_format(self):
        model = GARCHModel(model_type="GARCH")
        model.fit(self.series)
        fc = model.forecast(horizon=10)
        df_fc = fc.to_dataframe()

        self.assertEqual(len(df_fc), 10)
        self.assertIn("Daily_Variance", df_fc.columns)
        self.assertIn("Daily_Vol_Ann", df_fc.columns)
        self.assertIn("Term_Vol_Ann", df_fc.columns)

    def test_fit_on_loaded_data(self):
        spy_ret = load_equity_returns(ticker="SPY")
        model = GARCHModel(model_type="GJR-GARCH")
        res = model.fit(spy_ret)

        self.assertGreater(res.params["alpha"], 0.0)
        self.assertGreater(res.params["beta"], 0.5)
        self.assertGreater(res.persistence, 0.70)
        self.assertLess(res.persistence, 1.0)


if __name__ == "__main__":
    unittest.main()
