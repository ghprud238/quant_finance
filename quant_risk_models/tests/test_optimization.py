"""Comprehensive Unit Tests for Mean-Variance Portfolio Optimization & Efficient Frontier Engine."""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from quant_risk_models.optimization import (
    MeanVarianceOptimizer,
    OptimizationResult,
    EfficientFrontierResult,
    SimulatedPortfoliosResult,
)


class TestMeanVarianceOptimizer(unittest.TestCase):
    """Unit test suite for MeanVarianceOptimizer."""

    def setUp(self):
        np.random.seed(42)
        self.asset_names = ["AAPL", "MSFT", "GOOG", "AMZN", "JNJ", "TLT"]
        self.n_assets = len(self.asset_names)

        # Realistic annualized expected returns (CAGR)
        self.mu = pd.Series(
            [0.18, 0.16, 0.14, 0.15, 0.08, 0.04],
            index=self.asset_names,
            name="expected_returns",
        )

        # Realistic correlation and covariance matrix
        corr = np.array([
            [1.00, 0.65, 0.60, 0.58, 0.20, -0.15],
            [0.65, 1.00, 0.68, 0.62, 0.22, -0.18],
            [0.60, 0.68, 1.00, 0.55, 0.18, -0.12],
            [0.58, 0.62, 0.55, 1.00, 0.15, -0.10],
            [0.20, 0.22, 0.18, 0.15, 1.00,  0.25],
            [-0.15, -0.18, -0.12, -0.10, 0.25, 1.00],
        ])
        vols = np.array([0.24, 0.22, 0.25, 0.26, 0.14, 0.10])
        cov = np.outer(vols, vols) * corr
        self.cov_df = pd.DataFrame(cov, index=self.asset_names, columns=self.asset_names)

        self.risk_free_rate = 0.02
        self.optimizer = MeanVarianceOptimizer(
            expected_returns=self.mu,
            cov_matrix=self.cov_df,
            risk_free_rate=self.risk_free_rate,
        )

    def test_initialization_with_returns_df(self):
        """Tests initializing optimizer with historical daily returns DataFrame."""
        n_days = 500
        rets = np.random.multivariate_normal(
            mean=self.mu.values / 252,
            cov=self.cov_df.values / 252,
            size=n_days,
        )
        returns_df = pd.DataFrame(rets, columns=self.asset_names)

        opt = MeanVarianceOptimizer(returns_df=returns_df, risk_free_rate=0.02)
        self.assertEqual(opt.n_assets, self.n_assets)
        self.assertEqual(opt.asset_names, self.asset_names)
        self.assertEqual(opt.cov_matrix.shape, (self.n_assets, self.n_assets))

    def test_dimension_mismatch_raises(self):
        """Tests that mismatched dimensions raise ValueError."""
        bad_cov = self.cov_df.iloc[:3, :3]
        with self.assertRaises(ValueError):
            MeanVarianceOptimizer(expected_returns=self.mu, cov_matrix=bad_cov)

    def test_invalid_bounds_raises(self):
        """Tests that invalid bounds specification raises ValueError."""
        with self.assertRaises(ValueError):
            MeanVarianceOptimizer(
                expected_returns=self.mu,
                cov_matrix=self.cov_df,
                weight_bounds=[(0, 1), (0, 1)],  # Wrong length
            )

    def test_min_volatility_constraints(self):
        """Tests Global Minimum Volatility portfolio constraint satisfaction."""
        res = self.optimizer.min_volatility()

        self.assertTrue(res.success)
        # 1. Weights sum to 1
        self.assertAlmostEqual(res.weights.sum(), 1.0, places=5)
        # 2. Long only bounds [0, 1]
        self.assertTrue(np.all(res.weights.values >= -1e-6))
        self.assertTrue(np.all(res.weights.values <= 1.0 + 1e-6))

        # 3. Min vol must be <= min individual asset volatility
        min_asset_vol = np.min(np.sqrt(np.diag(self.cov_df.values)))
        self.assertLessEqual(res.volatility, min_asset_vol + 1e-4)

    def test_two_asset_analytical_solution(self):
        """Verifies SLSQP numerical minimum variance against analytical two-asset formula."""
        sub_mu = self.mu.iloc[:2]
        sub_cov = self.cov_df.iloc[:2, :2]
        opt2 = MeanVarianceOptimizer(expected_returns=sub_mu, cov_matrix=sub_cov)

        res = opt2.min_volatility()

        # Analytical unconstrained weights: w1 = (var2 - cov12) / (var1 + var2 - 2*cov12)
        var1 = sub_cov.iloc[0, 0]
        var2 = sub_cov.iloc[1, 1]
        cov12 = sub_cov.iloc[0, 1]
        analytical_w1 = (var2 - cov12) / (var1 + var2 - 2 * cov12)
        analytical_w2 = 1.0 - analytical_w1

        self.assertAlmostEqual(res.weights.iloc[0], analytical_w1, places=4)
        self.assertAlmostEqual(res.weights.iloc[1], analytical_w2, places=4)

    def test_max_sharpe_ratio(self):
        """Tests Tangency / Maximum Sharpe Ratio portfolio."""
        res = self.optimizer.max_sharpe_ratio(risk_free_rate=self.risk_free_rate)

        self.assertTrue(res.success)
        self.assertAlmostEqual(res.weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(res.weights.values >= -1e-6))

        # Compare with individual assets Sharpe ratios
        diag_vols = np.sqrt(np.diag(self.cov_df.values))
        single_sharpes = (self.mu.values - self.risk_free_rate) / diag_vols
        max_single_sharpe = np.max(single_sharpes)

        self.assertGreaterEqual(res.sharpe_ratio, max_single_sharpe - 1e-4)

        # Compare with equal weight portfolio Sharpe
        eq_w = np.ones(self.n_assets) / self.n_assets
        _, _, eq_sr = self.optimizer.portfolio_performance(eq_w)
        self.assertGreaterEqual(res.sharpe_ratio, eq_sr)

    def test_tangency_beats_random_portfolios(self):
        """Tests that Tangency portfolio Sharpe exceeds all simulated random portfolios."""
        tangency = self.optimizer.max_sharpe_ratio()
        sim = self.optimizer.simulate_random_portfolios(n_portfolios=2000, seed=123)

        max_sim_sharpe = np.max(sim.sharpe_ratios)
        self.assertGreaterEqual(tangency.sharpe_ratio, max_sim_sharpe - 1e-3)

    def test_target_return_portfolio(self):
        """Tests optimization subject to target expected return."""
        target_r = 0.12
        res = self.optimizer.efficient_return(target_return=target_r)

        self.assertTrue(res.success)
        self.assertAlmostEqual(res.weights.sum(), 1.0, places=5)
        self.assertGreaterEqual(res.expected_return, target_r - 1e-4)

    def test_target_volatility_portfolio(self):
        """Tests optimization subject to target volatility cap."""
        target_vol = 0.15
        res = self.optimizer.efficient_risk(target_volatility=target_vol)

        self.assertTrue(res.success)
        self.assertAlmostEqual(res.weights.sum(), 1.0, places=5)
        self.assertLessEqual(res.volatility, target_vol + 1e-4)

    def test_efficient_frontier_curve(self):
        """Tests continuous Efficient Frontier generation."""
        n_points = 30
        ef = self.optimizer.efficient_frontier(n_points=n_points)

        self.assertEqual(len(ef.returns), n_points)
        self.assertEqual(len(ef.volatilities), n_points)
        self.assertEqual(ef.weights.shape, (n_points, self.n_assets))

        # Verify monotonic non-decreasing returns
        self.assertTrue(np.all(np.diff(ef.returns) >= -1e-5))

        # Verify weights sum to 1 for all frontier portfolios
        weight_sums = ef.weights.sum(axis=1)
        np.testing.assert_allclose(weight_sums.values, 1.0, atol=1e-4)

        # Verify min vol point matches min_vol_portfolio
        self.assertAlmostEqual(ef.volatilities[0], ef.min_vol_portfolio.volatility, places=3)

    def test_capital_allocation_line(self):
        """Tests Capital Allocation Line generation."""
        cal = self.optimizer.capital_allocation_line(n_points=20)

        # 1. Starts at (0, Rf)
        self.assertAlmostEqual(cal["volatilities"][0], 0.0, places=5)
        self.assertAlmostEqual(cal["returns"][0], self.risk_free_rate, places=5)

        # 2. Linear slope equals tangency Sharpe ratio
        slope = (cal["returns"][-1] - cal["returns"][0]) / cal["volatilities"][-1]
        self.assertAlmostEqual(slope, cal["sharpe_ratio"], places=4)

    def test_custom_box_constraints(self):
        """Tests optimizer with max weight constraint (e.g. max 30% per asset)."""
        max_weight = 0.30
        custom_bounds = [(0.0, max_weight) for _ in range(self.n_assets)]

        res = self.optimizer.min_volatility(custom_bounds=custom_bounds)
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(res.weights.values <= max_weight + 1e-5))

    def test_short_selling_bounds(self):
        """Tests optimizer allowing short selling (-50% to +150%)."""
        short_bounds = [(-0.5, 1.5) for _ in range(self.n_assets)]
        opt_short = MeanVarianceOptimizer(
            expected_returns=self.mu,
            cov_matrix=self.cov_df,
            weight_bounds=short_bounds,
            risk_free_rate=0.02,
        )
        res = opt_short.min_volatility()
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(res.weights.values >= -0.5 - 1e-5))
        self.assertTrue(np.all(res.weights.values <= 1.5 + 1e-5))

    def test_zero_risk_free_rate(self):
        """Tests optimizer with risk_free_rate = 0.0."""
        opt0 = MeanVarianceOptimizer(
            expected_returns=self.mu,
            cov_matrix=self.cov_df,
            risk_free_rate=0.0,
        )
        res = opt0.max_sharpe_ratio()
        self.assertTrue(res.success)
        self.assertGreater(res.sharpe_ratio, 0.0)

    def test_simulated_portfolios_dataframe(self):
        """Tests random portfolio simulation export to DataFrame."""
        sim = self.optimizer.simulate_random_portfolios(n_portfolios=100, seed=42)
        df = sim.to_dataframe()

        self.assertEqual(len(df), 100 + self.n_assets + 1)
        self.assertIn("Return", df.columns)
        self.assertIn("Volatility", df.columns)
        self.assertIn("Sharpe_Ratio", df.columns)
        for name in self.asset_names:
            self.assertIn(name, df.columns)

    def test_result_summary_and_dict(self):
        """Tests result formatting methods."""
        res = self.optimizer.max_sharpe_ratio()
        d = res.to_dict()
        self.assertIn("expected_return", d)
        self.assertIn("volatility", d)
        self.assertIn("weights", d)

        summary_text = res.summary()
        self.assertIn("Portfolio Optimization Result", summary_text)

        ef = self.optimizer.efficient_frontier(n_points=10)
        ef_summary = ef.summary()
        self.assertIn("Efficient Frontier Summary", ef_summary)
        ef_df = ef.to_dataframe()
        self.assertEqual(len(ef_df), 10)


if __name__ == "__main__":
    unittest.main()
