"""Unit tests for Wasserstein Distributionally Robust Portfolio Optimization (DRO)."""

import unittest
import numpy as np
import pandas as pd
from genai_advanced_quant.robust_dro.dro_optimizer import (
    WassersteinDROOptimizer,
    DROResult,
)


class TestWassersteinDRO(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_assets = 5
        self.n_obs = 500
        self.asset_names = [f"Stock_{i+1}" for i in range(self.n_assets)]

        # Ground truth mean and covariance
        self.true_mu = np.array([0.12, 0.15, 0.08, 0.10, 0.14])
        cov_raw = np.random.normal(0, 0.05, (self.n_assets, self.n_assets))
        self.true_cov = cov_raw @ cov_raw.T + np.diag([0.04, 0.05, 0.03, 0.035, 0.045])

        # Generate sample returns
        daily_mu = self.true_mu / 252.0
        daily_cov = self.true_cov / 252.0
        returns_array = np.random.multivariate_normal(daily_mu, daily_cov, size=self.n_obs)
        self.returns_df = pd.DataFrame(returns_array, columns=self.asset_names)

        self.optimizer = WassersteinDROOptimizer(
            returns_data=self.returns_df,
            risk_aversion=1.0,
            risk_free_rate=0.02,
        )

    def test_nominal_equivalence(self):
        """Test that eps = 0 produces the exact nominal Markowitz Mean-Variance solution."""
        res_zero = self.optimizer.optimize(epsilon=0.0, norm_p=2)
        self.assertTrue(res_zero.converged)
        self.assertAlmostEqual(np.sum(res_zero.weights), 1.0, places=5)
        self.assertTrue(np.all(res_zero.weights >= -1e-6))
        self.assertAlmostEqual(res_zero.nominal_objective, res_zero.robust_objective, places=6)

    def test_robust_objective_monotonicity(self):
        """Test that worst-case robust objective is strictly non-decreasing in epsilon."""
        epsilons = [0.0, 0.005, 0.01, 0.02, 0.05]
        robust_objs = []
        for eps in epsilons:
            res = self.optimizer.optimize(epsilon=eps, norm_p=2)
            robust_objs.append(res.robust_objective)

        # Check monotonic increase
        for i in range(len(robust_objs) - 1):
            self.assertGreaterEqual(robust_objs[i+1] + 1e-7, robust_objs[i],
                                    msg=f"Failed monotonicity at eps={epsilons[i+1]}: {robust_objs[i+1]} < {robust_objs[i]}")

    def test_budget_and_long_only_constraints(self):
        """Verify sum(w) = 1 and w >= 0 for all ambiguity radii."""
        for eps in [0.001, 0.01, 0.03]:
            res = self.optimizer.optimize(epsilon=eps, allow_short=False)
            self.assertTrue(res.converged)
            self.assertAlmostEqual(float(np.sum(res.weights)), 1.0, places=5)
            self.assertTrue(np.all(res.weights.values >= -1e-7))

    def test_diversification_shrinkage_with_l2_norm(self):
        """Verify that increasing L2 Wasserstein radius shrinks weights towards equal-weight."""
        res_nom = self.optimizer.optimize(epsilon=0.0, norm_p=2)
        res_rob = self.optimizer.optimize(epsilon=0.08, norm_p=2)

        # Higher epsilon should increase effective N assets (diversification)
        self.assertGreaterEqual(res_rob.effective_n_assets + 1e-4, res_nom.effective_n_assets)
        # Higher epsilon should reduce peak concentration (max weight)
        self.assertLessEqual(res_rob.weights.max() - 1e-4, res_nom.weights.max())

    def test_norm_options(self):
        """Test optimization under L1, L2, L_inf, and Mahalanobis dual norms."""
        for norm_type in [1, 2, "inf", "mahalanobis"]:
            res = self.optimizer.optimize(epsilon=0.01, norm_p=norm_type)
            self.assertTrue(res.converged, msg=f"Failed convergence for norm {norm_type}")
            self.assertAlmostEqual(np.sum(res.weights), 1.0, places=5)

    def test_robust_efficient_frontier(self):
        """Verify continuous frontier generation across risk-aversion levels."""
        frontier = self.optimizer.robust_efficient_frontier(epsilon=0.01, n_points=15)
        self.assertEqual(len(frontier["returns"]), 15)
        self.assertEqual(len(frontier["volatilities"]), 15)
        self.assertEqual(frontier["weights"].shape, (15, self.n_assets))
        # Check all frontier weights satisfy budget
        for w in frontier["weights"]:
            self.assertAlmostEqual(np.sum(w), 1.0, places=5)

    def test_ambiguity_sweep_table(self):
        """Test dataframe output from evaluate_ambiguity_sweep."""
        df_sweep = self.optimizer.evaluate_ambiguity_sweep(epsilons=np.linspace(0, 0.02, 5))
        self.assertEqual(len(df_sweep), 5)
        self.assertIn("Robust_Obj", df_sweep.columns)
        self.assertIn("Effective_N", df_sweep.columns)

    def test_bootstrap_radius_estimation(self):
        """Test empirical bootstrap radius estimation."""
        eps_est = WassersteinDROOptimizer.estimate_wasserstein_radius(
            self.returns_df, confidence_level=0.95, n_bootstrap=100
        )
        self.assertGreater(eps_est, 0.0)
        self.assertLess(eps_est, 1.0)

    def test_out_of_sample_comparison(self):
        """Test out-of-sample benchmark comparison table across strategies."""
        train_df = self.returns_df.iloc[:350]
        test_df = self.returns_df.iloc[350:]
        df_comp = WassersteinDROOptimizer.out_of_sample_comparison(
            train_returns=train_df,
            test_returns=test_df,
            epsilon=0.015,
        )
        self.assertEqual(len(df_comp), 4)
        self.assertIn("Wasserstein DRO (Robust)", df_comp["Strategy"].values)
        self.assertIn("OOS_Sharpe", df_comp.columns)


if __name__ == "__main__":
    unittest.main()
