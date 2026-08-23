"""
Unit tests for Factor Exposure Analyzer (MultiFactorRegression & FactorExposureReport).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats

# Ensure package is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from quant_foundations.factors.model import MultiFactorRegression, FACTOR_MODEL_PRESETS
from quant_foundations.factors.exposure import FactorExposureReport


class TestMultiFactorRegression(unittest.TestCase):
    """Test suite for MultiFactorRegression."""

    def setUp(self):
        """Generate reproducible synthetic asset and factor return datasets."""
        np.random.seed(42)
        self.n_obs = 1000
        self.dates = pd.date_range(start="2020-01-01", periods=self.n_obs, freq="B")

        # 6 Custom Factors: Market, Value, Size, Momentum, Quality, Low_Vol
        factor_data = np.random.normal(0.0004, 0.01, size=(self.n_obs, 6))
        self.factor_names = ["Market", "Value", "Size", "Momentum", "Quality", "Low_Vol"]
        self.factor_df = pd.DataFrame(factor_data, index=self.dates, columns=self.factor_names)

        # True parameters
        self.true_alpha = 0.0002  # daily alpha ~ 5% annualized
        self.true_betas = np.array([1.10, 0.35, -0.20, 0.45, 0.15, -0.30])
        self.noise = np.random.normal(0.0, 0.005, size=self.n_obs)

        # Asset excess returns
        asset_excess = self.true_alpha + factor_data @ self.true_betas + self.noise
        self.asset_series = pd.Series(asset_excess, index=self.dates, name="Asset")
        self.rf_series = pd.Series(0.0001, index=self.dates, name="RF")
        self.asset_total = self.asset_series + self.rf_series

    def test_capm_regression(self):
        """Test single-factor CAPM regression."""
        mkt = self.factor_df[["Market"]]
        model = MultiFactorRegression(model_type="capm", cov_type="hc1")
        model.fit(self.asset_total, mkt, risk_free_rate=self.rf_series)

        self.assertTrue(model.is_fitted)
        self.assertIn("Market", model.betas)
        # Beta should be reasonably positive
        self.assertGreater(model.betas["Market"], 0.5)
        self.assertGreater(model.r_squared, 0.0)
        self.assertLessEqual(model.r_squared, 1.0)
        self.assertEqual(model.n_observations, self.n_obs)
        self.assertEqual(model.df_residuals, self.n_obs - 2)

    def test_multifactor_custom_preset_recovery(self):
        """Test multi-factor parameter estimation recovery on 6 custom factors."""
        model = MultiFactorRegression(model_type="custom", cov_type="hc1")
        model.fit(self.asset_series, self.factor_df, risk_free_rate=0.0)

        self.assertTrue(model.is_fitted)
        # Check estimated parameters close to true parameters
        self.assertAlmostEqual(model.alpha, self.true_alpha, delta=0.0005)
        for i, name in enumerate(self.factor_names):
            self.assertAlmostEqual(model.betas[name], self.true_betas[i], delta=0.05)

        # Annualized alpha check
        self.assertAlmostEqual(model.annualized_alpha, self.true_alpha * 252, delta=0.15)
        self.assertGreater(model.r_squared, 0.70)

    def test_presets_selection(self):
        """Test standard presets: FF3, Carhart4, FF5."""
        ff5_factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        ff5_df = pd.DataFrame(
            np.random.normal(0, 0.01, size=(self.n_obs, 5)),
            index=self.dates,
            columns=ff5_factors,
        )
        y = 0.0001 + ff5_df.values @ np.array([1.0, 0.2, -0.1, 0.3, -0.2]) + np.random.normal(0, 0.005, self.n_obs)
        y_s = pd.Series(y, index=self.dates)

        # Test FF3
        m_ff3 = MultiFactorRegression(model_type="ff3").fit(y_s, ff5_df)
        self.assertEqual(m_ff3.factor_names, ["Mkt-RF", "SMB", "HML"])

        # Test FF5
        m_ff5 = MultiFactorRegression(model_type="ff5").fit(y_s, ff5_df)
        self.assertEqual(m_ff5.factor_names, ff5_factors)

    def test_standard_errors_homoskedastic_vs_hc0_hc1(self):
        """Test standard errors across homoskedastic, White HC0, and White HC1."""
        # Introduce heteroskedasticity: variance proportional to market factor squared
        het_noise = np.random.normal(0, 1, self.n_obs) * (np.abs(self.factor_df["Market"].values) * 5.0 + 0.002)
        y_het = self.true_alpha + self.factor_df.values @ self.true_betas + het_noise

        m_homo = MultiFactorRegression(cov_type="homoskedastic").fit(y_het, self.factor_df)
        m_hc0 = MultiFactorRegression(cov_type="hc0").fit(y_het, self.factor_df)
        m_hc1 = MultiFactorRegression(cov_type="hc1").fit(y_het, self.factor_df)

        # HC1 should equal HC0 * (N / (N - p))
        N = self.n_obs
        p = len(self.factor_names) + 1
        scale = np.sqrt(N / (N - p))

        for name in self.factor_names:
            se_hc0 = m_hc0.standard_errors[name]
            se_hc1 = m_hc1.standard_errors[name]
            self.assertAlmostEqual(se_hc1, se_hc0 * scale, places=7)

        # p-values must be in [0, 1]
        for p_val in m_hc1.p_values.values():
            self.assertGreaterEqual(p_val, 0.0)
            self.assertLessEqual(p_val, 1.0)

    def test_ridge_regularization_multicollinear(self):
        """Test Ridge regularized regression on collinear factor portfolios."""
        # Create collinear factors
        f1 = np.random.normal(0, 0.01, self.n_obs)
        f2 = f1 + np.random.normal(0, 1e-4, self.n_obs)  # 0.999+ correlation
        collinear_factors = pd.DataFrame({"F1": f1, "F2": f2})
        y = 2.0 * f1 + np.random.normal(0, 0.005, self.n_obs)

        # OLS without regularization
        m_ols = MultiFactorRegression(regularization=None).fit(y, collinear_factors)
        # Ridge regularized
        m_ridge = MultiFactorRegression(regularization="ridge", alpha_ridge=10.0).fit(y, collinear_factors)

        self.assertTrue(m_ridge.is_fitted)
        # Ridge should stabilize coefficients
        self.assertLess(abs(m_ridge.betas["F1"]) + abs(m_ridge.betas["F2"]),
                        abs(m_ols.betas["F1"]) + abs(m_ols.betas["F2"]) + 5.0)

    def test_predictions(self):
        """Test predict method."""
        model = MultiFactorRegression().fit(self.asset_series, self.factor_df)
        preds = model.predict(self.factor_df)
        self.assertEqual(len(preds), self.n_obs)
        np.testing.assert_allclose(preds, model.fitted_values, rtol=1e-5)

    def test_missing_data_alignment(self):
        """Test handling of misaligned indices and NaN values."""
        asset_with_nan = self.asset_series.copy()
        asset_with_nan.iloc[10:15] = np.nan

        factors_with_nan = self.factor_df.copy()
        factors_with_nan.iloc[20:25, 0] = np.nan

        model = MultiFactorRegression().fit(asset_with_nan, factors_with_nan)
        self.assertEqual(model.n_observations, self.n_obs - 10)


class TestFactorExposureReport(unittest.TestCase):
    """Test suite for FactorExposureReport."""

    def setUp(self):
        """Set up test environment."""
        np.random.seed(123)
        self.n_obs = 800
        self.dates = pd.date_range(start="2021-01-01", periods=self.n_obs, freq="B")

        self.factor_names = ["Market", "Value", "Size", "Momentum", "Quality", "Low_Vol"]
        factor_data = np.random.normal(0.0003, 0.012, size=(self.n_obs, 6))
        self.factor_df = pd.DataFrame(factor_data, index=self.dates, columns=self.factor_names)

        true_betas = np.array([1.05, 0.40, -0.15, 0.30, 0.20, -0.25])
        asset_excess = 0.0001 + factor_data @ true_betas + np.random.normal(0, 0.006, self.n_obs)
        self.asset_series = pd.Series(asset_excess, index=self.dates, name="Fund_Excess")

        self.report = FactorExposureReport(
            asset_returns=self.asset_series,
            factor_returns=self.factor_df,
            annualization_factor=252,
        )

    def test_variance_decomposition_sum(self):
        """Test that Systematic Risk + Idiosyncratic Risk decomposes total excess variance."""
        sys_var = self.report.systematic_variance
        idio_var = self.report.idiosyncratic_variance
        tot_var = self.report.total_variance

        # Check decomp variance is close to sample variance
        self.assertAlmostEqual(sys_var + idio_var, tot_var, delta=tot_var * 0.05)

        # Risk percentages should sum to 100%
        sys_pct = self.report.systematic_risk_pct
        idio_pct = self.report.idiosyncratic_risk_pct
        self.assertAlmostEqual(sys_pct + idio_pct, 100.0, places=4)

    def test_factor_risk_attributions_sum(self):
        """Test that individual factor variance contributions sum to total systematic variance."""
        vc_dict = self.report.factor_variance_contributions
        sum_vc = sum(vc_dict.values())
        self.assertAlmostEqual(sum_vc, self.report.systematic_variance, places=7)

        # Systematic risk percentages per factor sum to 100%
        sys_pct_dict = self.report.factor_systematic_risk_pct
        sum_sys_pct = sum(sys_pct_dict.values())
        self.assertAlmostEqual(sum_sys_pct, 100.0, places=4)

    def test_summary_table_structure(self):
        """Test summary table columns and content matching the infographic."""
        table = self.report.summary_table()
        expected_cols = ["Beta", "Std_Error", "t_Stat", "p_Value", "Risk_Attribution_Pct", "Significance"]
        for col in expected_cols:
            self.assertIn(col, table.columns)

        self.assertEqual(len(table), len(self.factor_names))
        for factor in self.factor_names:
            self.assertIn(factor, table.index)

    def test_model_metrics(self):
        """Test model metrics dictionary."""
        metrics = self.report.model_metrics()
        self.assertIn("R_Squared", metrics)
        self.assertIn("Adjusted_R_Squared", metrics)
        self.assertIn("Annualized_Alpha", metrics)
        self.assertIn("Systematic_Risk_Pct", metrics)
        self.assertIn("Idiosyncratic_Risk_Pct", metrics)
        self.assertIn("Total_Volatility_Ann", metrics)

    def test_string_representation(self):
        """Test string formatted report output."""
        rep_str = str(self.report)
        self.assertIn("FACTOR EXPOSURE & RISK ATTRIBUTION REPORT", rep_str)
        self.assertIn("Market", rep_str)
        self.assertIn("Value", rep_str)


if __name__ == "__main__":
    unittest.main()
