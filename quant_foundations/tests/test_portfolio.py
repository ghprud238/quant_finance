"""Comprehensive Unit Tests for Portfolio Risk Dashboard and Risk Metrics Engine."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Ensure quant_foundations package in src/ is in sys.path
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import scipy.stats as stats

from quant_foundations.portfolio.risk_metrics import (
    _to_series,
    _align_series,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    downside_deviation,
    drawdown_series,
    excess_kurtosis,
    gain_loss_ratio,
    information_ratio,
    jensens_alpha,
    max_drawdown,
    max_drawdown_duration,
    omega_ratio,
    realized_beta,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tail_ratio,
    tracking_error,
    value_at_risk,
    win_rate,
)
from quant_foundations.portfolio.dashboard import PortfolioRiskDashboard


class TestReturnAndVolatility(unittest.TestCase):
    """Test annualized returns, volatility, and basic properties."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.daily_returns = pd.Series(np.random.normal(0.0008, 0.012, 252))

    def test_annualized_return_geometric(self) -> None:
        # Constant 1% daily return over 252 days
        const_returns = pd.Series([0.01] * 252)
        expected_cagr = (1.01 ** 252) - 1.0
        cagr = annualized_return(const_returns, periods_per_year=252, geometric=True)
        self.assertAlmostEqual(cagr, expected_cagr, places=5)

    def test_annualized_return_arithmetic(self) -> None:
        const_returns = pd.Series([0.01] * 252)
        expected_arithmetic = 0.01 * 252
        arithmetic = annualized_return(const_returns, periods_per_year=252, geometric=False)
        self.assertAlmostEqual(arithmetic, expected_arithmetic, places=5)

    def test_annualized_return_empty_and_loss(self) -> None:
        self.assertEqual(annualized_return([], periods_per_year=252), 0.0)
        # Total loss -100%
        total_loss = pd.Series([-1.0, 0.05])
        self.assertEqual(annualized_return(total_loss, periods_per_year=252, geometric=True), -1.0)

    def test_annualized_volatility(self) -> None:
        const_returns = pd.Series([0.02] * 100)
        self.assertEqual(annualized_volatility(const_returns), 0.0)

        # Standard normal random returns
        sample_std = self.daily_returns.std(ddof=1)
        expected_vol = sample_std * np.sqrt(252)
        vol = annualized_volatility(self.daily_returns, periods_per_year=252)
        self.assertAlmostEqual(vol, expected_vol, places=6)

    def test_annualized_volatility_edge_cases(self) -> None:
        self.assertEqual(annualized_volatility([]), 0.0)
        self.assertEqual(annualized_volatility([0.05]), 0.0)


class TestRiskAdjustedRatios(unittest.TestCase):
    """Test Sharpe, Sortino, Calmar, Omega, and Tail ratios."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.0005, 0.015, 500))

    def test_sharpe_ratio(self) -> None:
        rf = 0.03
        rf_periodic = rf / 252
        excess = self.returns - rf_periodic
        expected_sharpe = (excess.mean() / self.returns.std(ddof=1)) * np.sqrt(252)
        sharpe = sharpe_ratio(self.returns, risk_free_rate=rf, periods_per_year=252)
        self.assertAlmostEqual(sharpe, expected_sharpe, places=5)

    def test_sharpe_ratio_zero_vol(self) -> None:
        flat = pd.Series([0.001] * 50)
        self.assertEqual(sharpe_ratio(flat), 0.0)

    def test_sortino_ratio(self) -> None:
        rf = 0.02
        target = 0.0
        r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.04])
        sortino = sortino_ratio(r, risk_free_rate=rf, target_return=target, periods_per_year=252)
        
        # Manual verification
        rf_p = rf / 252
        excess_mean_ann = (r.mean() - rf_p) * 252
        downside_diff = np.minimum(r.values, 0.0)
        downside_dev = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(252)
        expected_sortino = excess_mean_ann / downside_dev
        self.assertAlmostEqual(sortino, expected_sortino, places=5)

    def test_sortino_ratio_no_downside(self) -> None:
        all_positive = pd.Series([0.01, 0.02, 0.03, 0.01])
        sortino = sortino_ratio(all_positive, target_return=0.0)
        self.assertEqual(sortino, float("inf"))

    def test_calmar_ratio(self) -> None:
        r = pd.Series([0.05, -0.02, -0.03, 0.04, 0.01])
        ann_ret = annualized_return(r, periods_per_year=252)
        mdd = abs(max_drawdown(r))
        expected_calmar = ann_ret / mdd
        calmar = calmar_ratio(r, periods_per_year=252)
        self.assertAlmostEqual(calmar, expected_calmar, places=5)

    def test_calmar_ratio_zero_drawdown(self) -> None:
        all_pos = pd.Series([0.01, 0.02, 0.03])
        self.assertEqual(calmar_ratio(all_pos), float("inf"))

    def test_omega_ratio(self) -> None:
        r = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04])
        gains = np.sum([0.05, 0.03, 0.04])
        losses = np.sum([0.02, 0.01])
        expected_omega = gains / losses
        self.assertAlmostEqual(omega_ratio(r, threshold=0.0), expected_omega, places=5)

    def test_omega_ratio_no_losses(self) -> None:
        all_pos = pd.Series([0.01, 0.02, 0.03])
        self.assertEqual(omega_ratio(all_pos, threshold=0.0), float("inf"))

    def test_tail_ratio(self) -> None:
        r = pd.Series(np.linspace(-0.10, 0.15, 100))
        q95 = np.percentile(r, 95)
        q5 = abs(np.percentile(r, 5))
        expected_tail = q95 / q5
        self.assertAlmostEqual(tail_ratio(r, upper_p=95, lower_p=5), expected_tail, places=5)


class TestDrawdownSeries(unittest.TestCase):
    """Test drawdown series, high-water mark, and duration calculations."""

    def test_drawdown_calculation_precision(self) -> None:
        # Day 0: +10% -> Wealth = 1.10, HWM = 1.10, DD = 0
        # Day 1: -10% -> Wealth = 1.10 * 0.9 = 0.99, HWM = 1.10, DD = (0.99 - 1.10)/1.10 = -0.10 (-10%)
        # Day 2: -10% -> Wealth = 0.99 * 0.9 = 0.891, HWM = 1.10, DD = (0.891 - 1.10)/1.10 = -0.19 (-19%)
        # Day 3: +25% -> Wealth = 0.891 * 1.25 = 1.11375, HWM = 1.11375, DD = 0
        returns = pd.Series([0.10, -0.10, -0.10, 0.25])
        dd_df = drawdown_series(returns)

        self.assertListEqual(
            list(dd_df.columns),
            [
                "cumulative_returns",
                "high_water_mark",
                "drawdown_pct",
                "drawdown_duration",
                "max_drawdown",
                "max_drawdown_duration",
            ],
        )
        self.assertAlmostEqual(dd_df.loc[0, "cumulative_returns"], 0.10, places=5)
        self.assertAlmostEqual(dd_df.loc[1, "drawdown_pct"], -0.10, places=5)
        self.assertAlmostEqual(dd_df.loc[2, "drawdown_pct"], -0.19, places=5)
        self.assertAlmostEqual(dd_df.loc[3, "drawdown_pct"], 0.0, places=5)

        self.assertAlmostEqual(dd_df.loc[2, "drawdown_duration"], 2)
        self.assertAlmostEqual(dd_df.loc[3, "drawdown_duration"], 0)

        # Max drawdown should be -19%
        self.assertAlmostEqual(max_drawdown(returns), -0.19, places=5)
        self.assertEqual(max_drawdown_duration(returns), 2)

    def test_empty_drawdown(self) -> None:
        empty_df = drawdown_series([])
        self.assertTrue(empty_df.empty)
        self.assertEqual(max_drawdown([]), 0.0)
        self.assertEqual(max_drawdown_duration([]), 0)


class TestValueAtRiskAndCVaR(unittest.TestCase):
    """Test all VaR methods (Historical, Parametric, Cornish-Fisher, Monte Carlo) and CVaR."""

    def setUp(self) -> None:
        np.random.seed(123)
        self.returns = pd.Series(np.random.normal(0.0004, 0.015, 1000))

    def test_historical_var(self) -> None:
        cl = 0.95
        expected_var = float(np.percentile(self.returns, 5.0))
        var = value_at_risk(self.returns, confidence_level=cl, method="historical")
        self.assertAlmostEqual(var, expected_var, places=6)

    def test_parametric_var(self) -> None:
        cl = 0.95
        mu = float(self.returns.mean())
        sigma = float(self.returns.std(ddof=1))
        z = float(stats.norm.ppf(cl))
        expected_var = mu - z * sigma
        var = value_at_risk(self.returns, confidence_level=cl, method="parametric")
        self.assertAlmostEqual(var, expected_var, places=6)

    def test_cornish_fisher_var(self) -> None:
        cl = 0.95
        mu = float(self.returns.mean())
        sigma = float(self.returns.std(ddof=1))
        s = float(stats.skew(self.returns.values, bias=False))
        k = float(stats.kurtosis(self.returns.values, fisher=True, bias=False))
        z = float(stats.norm.ppf(cl))
        z_tilde = z + (1/6)*(z**2 - 1)*s + (1/24)*(z**3 - 3*z)*k - (1/36)*(2*z**3 - 5*z)*(s**2)
        expected_var = mu - z_tilde * sigma
        var = value_at_risk(self.returns, confidence_level=cl, method="cornish_fisher")
        self.assertAlmostEqual(var, expected_var, places=6)

    def test_cornish_fisher_matches_parametric_for_normal(self) -> None:
        # For a large standard normal sample where skewness and excess kurtosis are near 0
        np.random.seed(42)
        large_normal = np.random.normal(0.0005, 0.02, 500000)
        var_param = value_at_risk(large_normal, confidence_level=0.95, method="parametric")
        var_cf = value_at_risk(large_normal, confidence_level=0.95, method="cornish_fisher")
        # Should be within 0.0005 of each other
        self.assertAlmostEqual(var_param, var_cf, delta=0.0005)

    def test_monte_carlo_var(self) -> None:
        var_mc_norm = value_at_risk(
            self.returns,
            confidence_level=0.95,
            method="monte_carlo",
            dist="normal",
            n_simulations=100000,
            random_state=42,
        )
        var_param = value_at_risk(self.returns, confidence_level=0.95, method="parametric")
        # Monte carlo normal should be very close to analytical parametric
        self.assertAlmostEqual(var_mc_norm, var_param, places=2)

        var_mc_t = value_at_risk(
            self.returns,
            confidence_level=0.95,
            method="monte_carlo",
            dist="t",
            n_simulations=50000,
            random_state=42,
        )
        self.assertIsInstance(var_mc_t, float)
        self.assertTrue(var_mc_t < 0.0)

    def test_invalid_var_method(self) -> None:
        with self.assertRaises(ValueError):
            value_at_risk(self.returns, method="invalid_method_name")

    def test_conditional_value_at_risk_historical(self) -> None:
        cl = 0.95
        var_hist = value_at_risk(self.returns, confidence_level=cl, method="historical")
        tail = self.returns[self.returns <= var_hist]
        expected_cvar = float(tail.mean())
        cvar = conditional_value_at_risk(self.returns, confidence_level=cl, method="historical")
        self.assertAlmostEqual(cvar, expected_cvar, places=6)
        # CVaR is more severe loss than VaR
        self.assertLessEqual(cvar, var_hist)

    def test_conditional_value_at_risk_parametric(self) -> None:
        cl = 0.95
        mu = float(self.returns.mean())
        sigma = float(self.returns.std(ddof=1))
        z = float(stats.norm.ppf(cl))
        alpha = 1.0 - cl
        pdf_z = float(stats.norm.pdf(z))
        expected_cvar = mu - sigma * (pdf_z / alpha)
        cvar = conditional_value_at_risk(self.returns, confidence_level=cl, method="parametric")
        self.assertAlmostEqual(cvar, expected_cvar, places=6)

    def test_invalid_cvar_method(self) -> None:
        with self.assertRaises(ValueError):
            conditional_value_at_risk(self.returns, method="invalid_method")


class TestBenchmarkRelativeMetrics(unittest.TestCase):
    """Test Beta, Jensen's Alpha, Tracking Error, and Information Ratio."""

    def setUp(self) -> None:
        np.random.seed(42)
        dates = pd.date_range("2025-01-01", periods=252, freq="B")
        self.rb = pd.Series(np.random.normal(0.0004, 0.012, 252), index=dates)
        noise = np.random.normal(0.0, 0.004, 252)
        self.rp = 1.2 * self.rb + 0.0003 + noise

    def test_realized_beta(self) -> None:
        beta = realized_beta(self.rp, self.rb)
        cov = np.cov(self.rp, self.rb, ddof=1)[0, 1]
        var_b = np.var(self.rb, ddof=1)
        self.assertAlmostEqual(beta, cov / var_b, places=6)
        self.assertAlmostEqual(beta, 1.2, delta=0.1)

    def test_beta_self_and_flat(self) -> None:
        self.assertAlmostEqual(realized_beta(self.rb, self.rb), 1.0, places=6)
        flat = pd.Series([0.0] * 50)
        self.assertEqual(realized_beta(self.rp.iloc[:50], flat), 0.0)

    def test_jensens_alpha(self) -> None:
        rf = 0.02
        alpha = jensens_alpha(self.rp, self.rb, risk_free_rate=rf, periods_per_year=252)
        beta = realized_beta(self.rp, self.rb)
        rf_p = rf / 252
        expected_alpha = ((self.rp.mean() - rf_p) - beta * (self.rb.mean() - rf_p)) * 252
        self.assertAlmostEqual(alpha, expected_alpha, places=6)

    def test_tracking_error(self) -> None:
        te = tracking_error(self.rp, self.rb, periods_per_year=252)
        diff = self.rp - self.rb
        expected_te = diff.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(te, expected_te, places=6)
        # Self tracking error is zero
        self.assertAlmostEqual(tracking_error(self.rb, self.rb), 0.0, places=6)

    def test_information_ratio(self) -> None:
        ir = information_ratio(self.rp, self.rb, periods_per_year=252)
        te = tracking_error(self.rp, self.rb, periods_per_year=252)
        active_ret = (self.rp.mean() - self.rb.mean()) * 252
        self.assertAlmostEqual(ir, active_ret / te, places=5)

    def test_information_ratio_zero_te(self) -> None:
        self.assertEqual(information_ratio(self.rb, self.rb), 0.0)


class TestDistributionMetrics(unittest.TestCase):
    """Test skewness, excess kurtosis, win rate, and gain-to-loss ratio."""

    def test_skewness_and_kurtosis(self) -> None:
        np.random.seed(42)
        normal_samples = np.random.normal(0, 1, 10000)
        self.assertAlmostEqual(skewness(normal_samples), 0.0, delta=0.1)
        self.assertAlmostEqual(excess_kurtosis(normal_samples), 0.0, delta=0.15)

    def test_win_rate_and_gain_loss(self) -> None:
        r = pd.Series([0.02, 0.04, -0.01, -0.02])
        self.assertAlmostEqual(win_rate(r), 0.5, places=5)
        # Average gain = 0.03, Average loss = 0.015 -> GL ratio = 2.0
        self.assertAlmostEqual(gain_loss_ratio(r), 2.0, places=5)


class TestPortfolioRiskDashboard(unittest.TestCase):
    """Test PortfolioRiskDashboard aggregation, summaries, and dashboard rendering."""

    def setUp(self) -> None:
        np.random.seed(42)
        dates = pd.date_range("2025-01-01", periods=252, freq="B")
        self.asset_a = pd.Series(np.random.normal(0.0006, 0.012, 252), index=dates, name="Asset_A")
        self.asset_b = pd.Series(np.random.normal(0.0004, 0.015, 252), index=dates, name="Asset_B")
        self.asset_c = pd.Series(np.random.normal(0.0008, 0.018, 252), index=dates, name="Asset_C")
        self.df_assets = pd.DataFrame({"Asset_A": self.asset_a, "Asset_B": self.asset_b, "Asset_C": self.asset_c})
        self.benchmark = pd.Series(np.random.normal(0.0004, 0.011, 252), index=dates, name="SP500")

    def test_single_asset_dashboard(self) -> None:
        dash = PortfolioRiskDashboard(
            returns=self.asset_a,
            benchmark_returns=self.benchmark,
            risk_free_rate=0.02,
            name="Tech Strategy",
            benchmark_name="S&P 500",
        )
        self.assertTrue(dash.has_benchmark)
        self.assertEqual(dash.name, "Tech Strategy")
        self.assertEqual(dash.benchmark_name, "S&P 500")

        metrics = dash.metrics()
        self.assertIn("annualized_return", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("calmar_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("var_historical", metrics)
        self.assertIn("cvar_historical", metrics)
        self.assertIn("realized_beta", metrics)
        self.assertIn("jensens_alpha", metrics)

    def test_multi_asset_dashboard_with_weights(self) -> None:
        weights = {"Asset_A": 0.5, "Asset_B": 0.3, "Asset_C": 0.2}
        dash = PortfolioRiskDashboard(
            returns=self.df_assets,
            weights=weights,
            benchmark_returns=self.benchmark,
            risk_free_rate=0.02,
            name="MultiAsset_Fund",
        )
        self.assertIsNotNone(dash.asset_returns)
        self.assertIsNotNone(dash.weights)
        self.assertAlmostEqual(dash.weights.sum(), 1.0)
        self.assertEqual(len(dash.portfolio_returns), 252)

        # Verify portfolio returns match weighted sum
        expected_port_ret = 0.5 * self.asset_a + 0.3 * self.asset_b + 0.2 * self.asset_c
        np.testing.assert_allclose(dash.portfolio_returns.values, expected_port_ret.values)

    def test_multi_asset_dashboard_default_equal_weights(self) -> None:
        dash = PortfolioRiskDashboard(
            returns=self.df_assets,
            weights=None,
        )
        self.assertAlmostEqual(dash.weights["Asset_A"], 1.0 / 3.0)
        self.assertAlmostEqual(dash.weights["Asset_B"], 1.0 / 3.0)
        self.assertAlmostEqual(dash.weights["Asset_C"], 1.0 / 3.0)

    def test_mismatched_weights_raises(self) -> None:
        with self.assertRaises(ValueError):
            PortfolioRiskDashboard(returns=self.df_assets, weights=[0.5, 0.5])  # 2 weights for 3 assets

    def test_summary_methods(self) -> None:
        dash = PortfolioRiskDashboard(
            returns=self.df_assets,
            benchmark_returns=self.benchmark,
            risk_free_rate=0.02,
        )
        df_summary = dash.summary(as_dataframe=True)
        self.assertIsInstance(df_summary, pd.DataFrame)
        self.assertIn("Category", df_summary.columns)
        self.assertIn("Metric", df_summary.columns)
        self.assertIn("Value", df_summary.columns)
        self.assertIn("Formatted", df_summary.columns)
        self.assertIn("Description", df_summary.columns)

        dict_summary = dash.summary(as_dataframe=False)
        self.assertIsInstance(dict_summary, dict)
        self.assertIn("sharpe_ratio", dict_summary)

    def test_print_dashboard_rendering(self) -> None:
        dash = PortfolioRiskDashboard(
            returns=self.df_assets,
            weights={"Asset_A": 0.4, "Asset_B": 0.4, "Asset_C": 0.2},
            benchmark_returns=self.benchmark,
            risk_free_rate=0.02,
            name="Alpha Growth Portfolio",
            benchmark_name="S&P 500 Index",
        )
        rendered = dash.print_dashboard(width=78)
        self.assertIsInstance(rendered, str)
        self.assertIn("PORTFOLIO RISK & PERFORMANCE DASHBOARD", rendered)
        self.assertIn("ALPHA GROWTH PORTFOLIO", rendered)
        self.assertIn("Sharpe Ratio", rendered)
        self.assertIn("Sortino Ratio", rendered)
        self.assertIn("Calmar Ratio", rendered)
        self.assertIn("Max Drawdown", rendered)
        self.assertIn("Cornish-Fisher VaR", rendered)
        self.assertIn("S&P 500 Index", rendered)
        self.assertIn("ASSET ALLOCATION BREAKDOWN", rendered)

    def test_drawdown_table_method(self) -> None:
        dash = PortfolioRiskDashboard(returns=self.asset_a)
        dd_table = dash.drawdown_table()
        self.assertIsInstance(dd_table, pd.DataFrame)
        self.assertEqual(len(dd_table), len(self.asset_a))
        self.assertIn("high_water_mark", dd_table.columns)



class TestInputValidationAndHelpers(unittest.TestCase):
    """Test helper functions and edge case input conversions."""

    def test_to_series_valid_inputs(self) -> None:
        # From list
        s_list = _to_series([0.01, -0.02, 0.03])
        self.assertEqual(len(s_list), 3)
        self.assertIsInstance(s_list, pd.Series)

        # From 1D array
        s_arr = _to_series(np.array([0.01, 0.02]))
        self.assertEqual(len(s_arr), 2)

        # From single-col DataFrame
        df = pd.DataFrame({"AAPL": [0.01, 0.02, 0.03]})
        s_df = _to_series(df)
        self.assertEqual(len(s_df), 3)

    def test_to_series_invalid_inputs(self) -> None:
        # Multi-column dataframe should raise ValueError
        df_multi = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with self.assertRaises(ValueError):
            _to_series(df_multi)

        # 2D array should raise ValueError
        arr_2d = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            _to_series(arr_2d)

    def test_align_series_with_dates(self) -> None:
        idx1 = pd.date_range("2025-01-01", periods=5, freq="D")
        idx2 = pd.date_range("2025-01-03", periods=5, freq="D")
        s1 = pd.Series([1, 2, 3, 4, 5], index=idx1)
        s2 = pd.Series([10, 20, 30, 40, 50], index=idx2)
        a1, a2 = _align_series(s1, s2)
        self.assertEqual(len(a1), 3)
        self.assertEqual(len(a2), 3)
        self.assertTrue((a1.index == a2.index).all())

    def test_align_series_empty(self) -> None:
        a1, a2 = _align_series([], [])
        self.assertEqual(len(a1), 0)
        self.assertEqual(len(a2), 0)

    def test_dashboard_with_dict_and_array(self) -> None:
        # Dict of asset returns
        data_dict = {"StockA": [0.01, 0.02, -0.01], "StockB": [0.02, -0.01, 0.03]}
        dash_dict = PortfolioRiskDashboard(data_dict, weights={"StockA": 0.5, "StockB": 0.5})
        self.assertEqual(len(dash_dict.portfolio_returns), 3)

        # 2D array of asset returns
        data_arr = np.array([[0.01, 0.02], [0.02, -0.01], [-0.01, 0.03]])
        dash_arr = PortfolioRiskDashboard(data_arr, weights=[0.6, 0.4])
        self.assertEqual(len(dash_arr.portfolio_returns), 3)


if __name__ == "__main__":
    unittest.main()
