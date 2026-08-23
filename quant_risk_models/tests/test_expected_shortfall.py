"""Unit tests for Expected Shortfall (CVaR) and Portfolio Risk Metrics."""

import unittest
import numpy as np
import pandas as pd
from scipy import stats

from quant_risk_models.cvar.expected_shortfall import (
    ExpectedShortfallModel,
    KupiecBacktestResult,
    ChristoffersenBacktestResult,
    RiskBacktestReport,
    ComponentCVaRReport,
)
from quant_risk_models.portfolio.risk_metrics import PortfolioRiskReport


class TestExpectedShortfallModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.model = ExpectedShortfallModel(confidence_level=0.95)
        # 10,000 observations from standard normal
        self.normal_returns = pd.Series(np.random.normal(loc=0.0005, scale=0.015, size=10_000))
        # Heavy-tailed Student-t returns (df=4)
        self.t_returns = pd.Series(stats.t.rvs(df=4, loc=0.0005, scale=0.015 * np.sqrt(2.0 / 4.0), size=10_000, random_state=42))

    def test_coherence_property_cvar_ge_var(self):
        """Test that CVaR >= VaR (coherence property) for all confidence levels and estimators."""
        for cl in [0.90, 0.95, 0.99]:
            # 1. Historical
            h_var = self.model.historical_var(self.normal_returns, confidence_level=cl, as_loss=True)
            h_es = self.model.historical_es(self.normal_returns, confidence_level=cl, as_loss=True)
            self.assertGreaterEqual(h_es, h_var - 1e-9)

            # 2. Parametric Gaussian
            g_var = self.model.parametric_gaussian_var(self.normal_returns, confidence_level=cl, as_loss=True)
            g_es = self.model.parametric_gaussian_es(self.normal_returns, confidence_level=cl, as_loss=True)
            self.assertGreaterEqual(g_es, g_var - 1e-9)

            # 3. Parametric Student-t
            t_var = self.model.parametric_student_t_var(self.t_returns, confidence_level=cl, df=4, as_loss=True)
            t_es = self.model.parametric_student_t_es(self.t_returns, confidence_level=cl, df=4, as_loss=True)
            self.assertGreaterEqual(t_es, t_var - 1e-9)

            # 4. Monte Carlo
            mc_var, mc_es = self.model.monte_carlo_es(self.t_returns, confidence_level=cl, n_simulations=50_000, as_loss=True)
            self.assertGreaterEqual(mc_es, mc_var - 1e-9)

    def test_confidence_level_monotonicity(self):
        """Test that higher confidence levels produce strictly higher VaR and CVaR."""
        var_90 = self.model.historical_var(self.normal_returns, confidence_level=0.90)
        var_95 = self.model.historical_var(self.normal_returns, confidence_level=0.95)
        var_99 = self.model.historical_var(self.normal_returns, confidence_level=0.99)
        self.assertLess(var_90, var_95)
        self.assertLess(var_95, var_99)

        es_90 = self.model.historical_es(self.normal_returns, confidence_level=0.90)
        es_95 = self.model.historical_es(self.normal_returns, confidence_level=0.95)
        es_99 = self.model.historical_es(self.normal_returns, confidence_level=0.99)
        self.assertLess(es_90, es_95)
        self.assertLess(es_95, es_99)

    def test_heavy_tails_student_t_vs_gaussian(self):
        """Heavy-tailed distribution must exhibit higher tail risk at 99% than Gaussian."""
        g_es_99 = self.model.parametric_gaussian_es(self.t_returns, confidence_level=0.99)
        t_es_99 = self.model.parametric_student_t_es(self.t_returns, confidence_level=0.99, df=4)
        self.assertGreater(t_es_99, g_es_99)

    def test_component_cvar_euler_additivity(self):
        """Test that sum of Component CVaR equals total portfolio CVaR (Euler additivity)."""
        # Create 4 asset returns matrix
        cov = np.array([
            [0.0004, 0.0002, 0.0001, 0.0000],
            [0.0002, 0.0005, 0.0002, 0.0001],
            [0.0001, 0.0002, 0.0006, 0.0002],
            [0.0000, 0.0001, 0.0002, 0.0003]
        ])
        raw_ret = np.random.multivariate_normal([0.0005, 0.0006, 0.0004, 0.0003], cov, size=5000)
        df_ret = pd.DataFrame(raw_ret, columns=["AAPL", "MSFT", "GOOG", "TLT"])
        weights = {"AAPL": 0.35, "MSFT": 0.35, "GOOG": 0.15, "TLT": 0.15}

        # 1. Historical Component CVaR
        h_comp = self.model.component_cvar(df_ret, weights=weights, confidence_level=0.95, method="historical")
        self.assertAlmostEqual(h_comp.component_cvar.sum(), h_comp.portfolio_cvar, places=6)
        self.assertAlmostEqual(h_comp.percentage_cvar.sum(), 1.0, places=6)

        # 2. Parametric Component CVaR
        p_comp = self.model.component_cvar(df_ret, weights=weights, confidence_level=0.95, method="parametric")
        self.assertAlmostEqual(p_comp.component_cvar.sum(), p_comp.portfolio_cvar, places=6)
        self.assertAlmostEqual(p_comp.percentage_cvar.sum(), 1.0, places=6)

    def test_kupiec_and_christoffersen_backtest(self):
        """Test Kupiec POF and Christoffersen independence tests."""
        # Simulated well-calibrated VaR series
        n_obs = 1000
        returns = pd.Series(np.random.normal(0, 0.01, size=n_obs))
        # 95% parametric VaR forecast
        var_fc = float(1.644853 * 0.01)

        report = self.model.backtest_var(returns, var_forecasts=var_fc, confidence_level=0.95)
        self.assertEqual(report.confidence_level, 0.95)
        self.assertEqual(report.kupiec.total_observations, n_obs)
        # Expected exceptions ~ 50 out of 1000
        self.assertAlmostEqual(report.kupiec.expected_exceptions, 50.0)
        self.assertFalse(report.kupiec.is_rejected)

        summary_df = report.summary()
        self.assertIn("Overall Backtest Verdict", summary_df["Metric"].values)


class TestPortfolioRiskReport(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        dates = pd.date_range(start="2020-01-01", periods=756, freq="B")
        self.returns = pd.Series(np.random.normal(0.0005, 0.012, size=756), index=dates)
        self.report = PortfolioRiskReport(self.returns, portfolio_name="Alpha Tech Fund", risk_free_rate=0.02)

    def test_risk_metrics_computation(self):
        """Verify all metrics calculate without error and fall in plausible bounds."""
        ann_ret = self.report.annualized_return()
        ann_vol = self.report.annualized_volatility()
        sr = self.report.sharpe_ratio()
        sor = self.report.sortino_ratio()
        mdd = self.report.max_drawdown()
        v95 = self.report.var_95()
        es95 = self.report.cvar_95()

        self.assertIsInstance(ann_ret, float)
        self.assertGreater(ann_vol, 0.0)
        self.assertIsInstance(sr, float)
        self.assertIsInstance(sor, float)
        self.assertLessEqual(mdd, 0.0)  # Max drawdown is negative or zero
        self.assertGreaterEqual(es95, v95)

    def test_summary_and_export(self):
        """Test dictionary and dataframe formatting."""
        d = self.report.to_dict()
        self.assertEqual(d["portfolio_name"], "Alpha Tech Fund")
        self.assertEqual(d["observations"], 756)

        df = self.report.summary_dataframe()
        self.assertEqual(len(df), 10)
        self.assertIn("Metric", df.columns)
        self.assertIn("Value", df.columns)


if __name__ == "__main__":
    unittest.main()
