"""Unit tests for Project 47: Global Macro Cross-Asset & Emerging Markets Sovereign Risk Contagion Model."""

import unittest
import numpy as np
import pandas as pd

from macro_ai_cross_markets.data.loader import generate_macro_market_data
from macro_ai_cross_markets.sovereign_contagion.spillover import SovereignContagionEngine


class TestSovereignContagion(unittest.TestCase):
    """Validates VAR estimation, Diebold-Yilmaz GFEVD decomposition, spillover index bounds, and Copula tail dependence."""

    def setUp(self):
        self.engine = SovereignContagionEngine(var_lags=2, forecast_horizon=10)
        self.market_data = generate_macro_market_data(n_days=400, seed=42)
        self.cds_df = self.market_data["cds_spreads"][["US", "Germany", "Italy", "Greece", "Brazil", "Turkey"]]

    def test_var_parameter_estimation(self):
        diff_df = self.cds_df.diff().dropna()
        A_mats, Sigma = self.engine.estimate_var_parameters(diff_df)

        self.assertEqual(len(A_mats), 2)
        self.assertEqual(A_mats[0].shape, (6, 6))
        self.assertEqual(Sigma.shape, (6, 6))
        # Covariance Sigma must be symmetric positive semi-definite
        self.assertTrue(np.allclose(Sigma, Sigma.T, atol=1e-5))

    def test_diebold_yilmaz_spillovers(self):
        dy_res = self.engine.compute_diebold_yilmaz_spillovers(self.cds_df)

        self.assertGreater(dy_res.total_spillover_index, 0.0)
        self.assertLessEqual(dy_res.total_spillover_index, 100.0)

        # Row sums of spillover matrix must equal 100%
        row_sums = dy_res.spillover_matrix.sum(axis=1).values
        for s in row_sums:
            self.assertAlmostEqual(s, 100.0, places=1)

        self.assertEqual(len(dy_res.net_spillover), 6)
        self.assertTrue(len(dy_res.net_transmitters) > 0)
        self.assertTrue(len(dy_res.net_receivers) > 0)

    def test_clayton_copula_tail_dependence(self):
        diff_df = self.cds_df.diff().dropna()
        u = diff_df["Turkey"].values
        v = diff_df["Brazil"].values

        theta, lambda_L = self.engine.fit_bivariate_clayton_copula(u, v)
        self.assertGreater(theta, 0.0)
        self.assertTrue(0.0 <= lambda_L <= 1.0)

    def test_gumbel_copula_tail_dependence(self):
        diff_df = self.cds_df.diff().dropna()
        u = diff_df["Italy"].values
        v = diff_df["Greece"].values

        theta, lambda_U = self.engine.fit_bivariate_gumbel_copula(u, v)
        self.assertGreaterEqual(theta, 1.0)
        self.assertTrue(0.0 <= lambda_U <= 1.0)

    def test_full_sovereign_report(self):
        report = self.engine.generate_full_sovereign_report(self.cds_df)

        self.assertGreater(report.total_spillover_index, 0.0)
        self.assertIsNotNone(report.top_systemic_transmitter)
        self.assertIsNotNone(report.top_vulnerable_receiver)
        self.assertFalse(report.spillover_table.empty)
        self.assertFalse(report.copula_tail_table.empty)


if __name__ == '__main__':
    unittest.main()
