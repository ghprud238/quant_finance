"""Unit tests for Yield Curve Term Structure & Bootstrapping (Module 22)."""

import unittest
import numpy as np
import pandas as pd
from advanced_quant_ml.yield_curve import (
    NelsonSiegelModel,
    NelsonSiegelSvenssonModel,
    YieldCurveBootstrapper,
    YieldCurvePCA,
    NelsonSiegelFitResult,
    NSSFitResult,
    BootstrapResult,
)
from advanced_quant_ml.data import load_yield_curve_data, YIELD_CURVE_MATURITIES


class TestYieldCurveModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample upward-sloping Treasury curve
        cls.maturities = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        cls.par_yields = np.array([1.50, 1.75, 2.05, 2.40, 2.85, 3.15, 3.55, 3.80, 4.05, 4.35, 4.45])

    def test_nelson_siegel_calibration(self):
        ns = NelsonSiegelModel()
        fit_res = ns.fit(self.maturities, self.par_yields)

        self.assertIsInstance(fit_res, NelsonSiegelFitResult)
        self.assertGreater(fit_res.beta0, 0.0)
        self.assertGreater(fit_res.lambda_decay, 0.0)
        self.assertLess(fit_res.rmse, 0.15)  # Fitting error under 15 bps
        self.assertGreater(fit_res.r_squared, 0.95)

        # Test yield and forward predictions
        y_5y = fit_res.predict_yield(5.0)
        f_5y = fit_res.predict_forward(5.0)
        self.assertGreater(y_5y, 0.0)
        self.assertGreater(f_5y, 0.0)

    def test_nelson_siegel_fixed_lambda(self):
        ns = NelsonSiegelModel(lambda_decay=1.5)
        fit_res = ns.fit(self.maturities, self.par_yields)

        self.assertEqual(fit_res.lambda_decay, 1.5)
        self.assertGreater(fit_res.beta0, 0.0)
        self.assertLess(fit_res.rmse, 0.20)

    def test_nss_calibration(self):
        nss = NelsonSiegelSvenssonModel()
        fit_res = nss.fit(self.maturities, self.par_yields)

        self.assertIsInstance(fit_res, NSSFitResult)
        self.assertGreater(fit_res.beta0, 0.0)
        self.assertLess(fit_res.rmse, 0.10)
        self.assertGreater(fit_res.r_squared, 0.98)

    def test_yield_curve_bootstrapping(self):
        res = YieldCurveBootstrapper.bootstrap_par_yields(self.maturities, self.par_yields)

        self.assertIsInstance(res, BootstrapResult)
        self.assertEqual(len(res.zero_rates), len(self.maturities))
        self.assertEqual(len(res.discount_factors), len(self.maturities))
        self.assertTrue(np.all(res.discount_factors > 0.0))
        self.assertTrue(np.all(res.discount_factors <= 1.0))

        # Test discount factor monotonicity and interpolation
        df_1y = res.get_discount_factor(1.0)
        df_10y = res.get_discount_factor(10.0)
        self.assertGreater(df_1y, df_10y)

        zero_5y = res.get_zero_rate(5.0)
        self.assertGreater(zero_5y, 0.0)

    def test_yield_curve_pca(self):
        df_yields = load_yield_curve_data()
        pca = YieldCurvePCA(n_components=3).fit(df_yields)

        summary = pca.summary()
        self.assertIn("PC1 (Level Shift)", summary.index)
        self.assertIn("PC2 (Slope Tilt)", summary.index)
        self.assertIn("PC3 (Curvature Twist)", summary.index)

        # Top 3 PCs explain > 90% of total curve variance
        self.assertGreater(summary["Cumulative_Pct"].iloc[-1], 90.0)


if __name__ == "__main__":
    unittest.main()
