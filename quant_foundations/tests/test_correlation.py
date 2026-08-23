"""Comprehensive Unit Test Suite for Module 3: Correlation & Covariance Engine, Shrinkage, PCA, Clustering."""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from quant_foundations.correlation import (
    compute_correlation_matrix,
    compute_covariance_matrix,
    ledoit_wolf_shrinkage,
    ewma_covariance,
    rolling_correlation,
    is_positive_definite,
    nearest_correlation_matrix,
    PCAFactorEngine,
    correlation_distance,
    hierarchical_correlation_clustering,
    quasi_diagonalize,
)


class TestCorrelationMatrix(unittest.TestCase):
    """Test suite for correlation and covariance matrix estimation."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='B')
        self.assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']
        returns = np.random.normal(loc=0.0005, scale=0.015, size=(200, 5))
        self.returns_df = pd.DataFrame(returns, index=dates, columns=self.assets)

    def test_compute_correlation_matrix_methods(self):
        for method in ['pearson', 'spearman', 'kendall']:
            corr = compute_correlation_matrix(self.returns_df, method=method)
            self.assertIsInstance(corr, pd.DataFrame)
            self.assertEqual(corr.shape, (5, 5))
            self.assertEqual(list(corr.index), self.assets)
            self.assertEqual(list(corr.columns), self.assets)
            self.assertTrue(np.allclose(corr.values, corr.values.T, atol=1e-7))
            self.assertTrue(np.allclose(np.diag(corr.values), 1.0, atol=1e-7))
            self.assertTrue(np.all(corr.values >= -1.0 - 1e-7) and np.all(corr.values <= 1.0 + 1e-7))

    def test_compute_correlation_matrix_perfect_correlation(self):
        x = np.random.randn(50)
        df_perfect = pd.DataFrame({'A': x, 'B': 2.0 * x, 'C': -3.0 * x})
        corr = compute_correlation_matrix(df_perfect, method='pearson')
        self.assertAlmostEqual(corr.loc['A', 'B'], 1.0, places=6)
        self.assertAlmostEqual(corr.loc['A', 'C'], -1.0, places=6)
        self.assertAlmostEqual(corr.loc['B', 'C'], -1.0, places=6)

    def test_compute_correlation_matrix_errors(self):
        with self.assertRaises(TypeError):
            compute_correlation_matrix(self.returns_df.values)
        with self.assertRaises(ValueError):
            compute_correlation_matrix(pd.DataFrame())
        with self.assertRaises(ValueError):
            compute_correlation_matrix(self.returns_df.iloc[:1])
        with self.assertRaises(ValueError):
            compute_correlation_matrix(self.returns_df, method='unsupported_method')

    def test_compute_covariance_matrix_annualization(self):
        cov_daily = compute_covariance_matrix(self.returns_df, annualized=False)
        cov_annual = compute_covariance_matrix(self.returns_df, annualized=True, periods_per_year=252)
        cov_monthly = compute_covariance_matrix(self.returns_df, annualized=True, periods_per_year=12)
        
        self.assertIsInstance(cov_daily, pd.DataFrame)
        self.assertIsInstance(cov_annual, pd.DataFrame)
        self.assertTrue(np.allclose(cov_annual.values, cov_daily.values * 252))
        self.assertTrue(np.allclose(cov_monthly.values, cov_daily.values * 12))
        self.assertTrue(is_positive_definite(cov_daily))
        self.assertTrue(is_positive_definite(cov_annual))

    def test_compute_covariance_matrix_errors(self):
        with self.assertRaises(TypeError):
            compute_covariance_matrix([1, 2, 3])
        with self.assertRaises(ValueError):
            compute_covariance_matrix(pd.DataFrame())
        with self.assertRaises(ValueError):
            compute_covariance_matrix(self.returns_df, periods_per_year=0)
        with self.assertRaises(ValueError):
            compute_covariance_matrix(self.returns_df, periods_per_year=-10)


class TestLedoitWolfShrinkage(unittest.TestCase):
    """Test suite for Ledoit-Wolf optimal shrinkage estimator."""

    def setUp(self):
        np.random.seed(123)
        self.T, self.N = 150, 8
        returns = np.random.randn(self.T, self.N) * 0.02
        self.returns_df = pd.DataFrame(
            returns,
            columns=[f"Asset_{i}" for i in range(self.N)]
        )

    def test_constant_correlation_shrinkage(self):
        shrunk_cov, delta = ledoit_wolf_shrinkage(
            self.returns_df,
            shrinkage_target='constant_correlation',
            annualized=True,
            periods_per_year=252
        )
        self.assertIsInstance(shrunk_cov, pd.DataFrame)
        self.assertEqual(shrunk_cov.shape, (self.N, self.N))
        self.assertEqual(list(shrunk_cov.columns), list(self.returns_df.columns))
        self.assertEqual(list(shrunk_cov.index), list(self.returns_df.columns))
        self.assertGreaterEqual(delta, 0.0)
        self.assertLessEqual(delta, 1.0)
        self.assertTrue(is_positive_definite(shrunk_cov))
        self.assertTrue(np.allclose(shrunk_cov.values, shrunk_cov.values.T))

    def test_identity_shrinkage(self):
        shrunk_cov, delta = ledoit_wolf_shrinkage(
            self.returns_df,
            shrinkage_target='identity',
            annualized=False
        )
        self.assertIsInstance(shrunk_cov, pd.DataFrame)
        self.assertEqual(shrunk_cov.shape, (self.N, self.N))
        self.assertGreaterEqual(delta, 0.0)
        self.assertLessEqual(delta, 1.0)
        self.assertTrue(is_positive_definite(shrunk_cov))
        self.assertTrue(np.allclose(shrunk_cov.values, shrunk_cov.values.T))

    def test_shrinkage_extreme_singular(self):
        # Short sample where T < N (singular sample covariance)
        T_small = 6
        N_large = 12
        singular_returns = pd.DataFrame(
            np.random.randn(T_small, N_large) * 0.01,
            columns=[f"Asset_{i}" for i in range(N_large)]
        )
        sample_cov = singular_returns.cov()
        # Sample cov is singular (rank <= 5 < 12)
        self.assertFalse(is_positive_definite(sample_cov))
        
        # Ledoit-Wolf shrinkage restores full rank and positive definiteness
        shrunk_cov, delta = ledoit_wolf_shrinkage(
            singular_returns,
            shrinkage_target='constant_correlation'
        )
        self.assertTrue(is_positive_definite(shrunk_cov))
        self.assertGreater(delta, 0.0)

    def test_invalid_shrinkage_inputs(self):
        with self.assertRaises(TypeError):
            ledoit_wolf_shrinkage(self.returns_df.values)
        with self.assertRaises(ValueError):
            ledoit_wolf_shrinkage(self.returns_df, shrinkage_target='invalid_target')
        with self.assertRaises(ValueError):
            # Less than 2 assets
            ledoit_wolf_shrinkage(self.returns_df[['Asset_0']])
        with self.assertRaises(ValueError):
            # Less than 3 observations
            ledoit_wolf_shrinkage(self.returns_df.iloc[:2])


class TestEWMACovariance(unittest.TestCase):
    """Test suite for RiskMetrics EWMA covariance."""

    def setUp(self):
        np.random.seed(42)
        returns = np.random.randn(150, 4) * 0.01
        self.returns_df = pd.DataFrame(returns, columns=['A', 'B', 'C', 'D'])

    def test_ewma_covariance_properties(self):
        cov = ewma_covariance(self.returns_df, decay_factor=0.94, annualized=True)
        self.assertIsInstance(cov, pd.DataFrame)
        self.assertEqual(cov.shape, (4, 4))
        self.assertTrue(is_positive_definite(cov))
        self.assertTrue(np.allclose(cov.values, cov.values.T))

    def test_ewma_decay_weighting(self):
        # Recent shock in asset A should strongly elevate variance under EWMA
        df_spike = self.returns_df.copy()
        df_spike.iloc[-1, 0] = 0.50
        
        cov_regular = ewma_covariance(self.returns_df, decay_factor=0.94, annualized=False)
        cov_spike = ewma_covariance(df_spike, decay_factor=0.94, annualized=False)
        
        self.assertGreater(cov_spike.loc['A', 'A'], cov_regular.loc['A', 'A'] * 5)

    def test_ewma_annualization_scaling(self):
        cov_unann = ewma_covariance(self.returns_df, decay_factor=0.94, annualized=False)
        cov_ann = ewma_covariance(self.returns_df, decay_factor=0.94, annualized=True, periods_per_year=252)
        self.assertTrue(np.allclose(cov_ann.values, cov_unann.values * 252))

    def test_ewma_invalid_parameters(self):
        with self.assertRaises(TypeError):
            ewma_covariance(self.returns_df.values)
        with self.assertRaises(ValueError):
            ewma_covariance(self.returns_df, decay_factor=1.0)
        with self.assertRaises(ValueError):
            ewma_covariance(self.returns_df, decay_factor=0.0)
        with self.assertRaises(ValueError):
            ewma_covariance(self.returns_df, decay_factor=-0.5)
        with self.assertRaises(ValueError):
            ewma_covariance(self.returns_df, periods_per_year=0)


class TestRollingCorrelation(unittest.TestCase):
    """Test suite for rolling correlation calculation."""

    def setUp(self):
        np.random.seed(99)
        dates = pd.date_range('2023-01-01', periods=100)
        x = np.random.randn(100)
        y = 0.8 * x + 0.2 * np.random.randn(100)
        self.returns_df = pd.DataFrame({'AssetA': x, 'AssetB': y}, index=dates)

    def test_rolling_correlation_computation(self):
        window = 30
        rc = rolling_correlation(self.returns_df, 'AssetA', 'AssetB', window=window)
        self.assertIsInstance(rc, pd.Series)
        self.assertEqual(len(rc), 100)
        self.assertTrue(rc.iloc[:window-1].isna().all())
        self.assertFalse(rc.iloc[window-1:].isna().any())
        self.assertTrue(np.all(rc.iloc[window-1:] > 0.5))

    def test_rolling_correlation_errors(self):
        with self.assertRaises(TypeError):
            rolling_correlation(self.returns_df.values, 'AssetA', 'AssetB')
        with self.assertRaises(KeyError):
            rolling_correlation(self.returns_df, 'AssetA', 'AssetNonExistent')
        with self.assertRaises(KeyError):
            rolling_correlation(self.returns_df, 'AssetNonExistent', 'AssetB')
        with self.assertRaises(ValueError):
            rolling_correlation(self.returns_df, 'AssetA', 'AssetB', window=1)


class TestPositiveDefiniteAndNearestCorrelation(unittest.TestCase):
    """Test suite for positive definiteness check and Higham (2002) algorithm."""

    def test_is_positive_definite(self):
        # Identity is PD
        self.assertTrue(is_positive_definite(np.eye(4)))
        
        # Singular matrix is not PD
        singular = np.ones((3, 3))
        self.assertFalse(is_positive_definite(singular))
        
        # Indefinite matrix
        indef = np.array([[1.0, 2.0], [2.0, 1.0]])
        self.assertFalse(is_positive_definite(indef))

        # Asymmetric matrix
        asym = np.array([[1.0, 0.5], [0.1, 1.0]])
        self.assertFalse(is_positive_definite(asym))

        # Non-square matrix
        non_sq = np.ones((3, 4))
        self.assertFalse(is_positive_definite(non_sq))

    def test_nearest_correlation_matrix_higham(self):
        # Create non-positive definite matrix with unit diagonal
        non_psd = np.array([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.9],
            [0.7, 0.9, 1.0]
        ])
        non_psd[0, 2] = -0.9
        non_psd[2, 0] = -0.9
        
        self.assertFalse(is_positive_definite(non_psd))
        
        near_corr = nearest_correlation_matrix(non_psd)
        self.assertIsInstance(near_corr, np.ndarray)
        self.assertTrue(np.allclose(np.diag(near_corr), 1.0))
        self.assertTrue(np.allclose(near_corr, near_corr.T))
        self.assertTrue(is_positive_definite(near_corr))

    def test_nearest_correlation_matrix_dataframe(self):
        df_bad = pd.DataFrame([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0]
        ], index=['X', 'Y', 'Z'], columns=['X', 'Y', 'Z'])
        
        res = nearest_correlation_matrix(df_bad)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(list(res.columns), ['X', 'Y', 'Z'])
        self.assertEqual(list(res.index), ['X', 'Y', 'Z'])
        self.assertTrue(is_positive_definite(res))

    def test_nearest_correlation_matrix_already_pd(self):
        # Valid correlation matrix should remain unchanged
        pd_corr = np.array([
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        res = nearest_correlation_matrix(pd_corr)
        self.assertTrue(np.allclose(res, pd_corr, atol=1e-5))


class TestPCAFactorEngine(unittest.TestCase):
    """Test suite for PCA Factor Engine."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        self.assets = ['EQ1', 'EQ2', 'EQ3', 'EQ4', 'EQ5']
        
        # Create returns with 2 dominant market/sector factors
        f1 = np.random.randn(300)
        f2 = np.random.randn(300)
        noise = np.random.randn(300, 5) * 0.2
        
        returns = np.zeros((300, 5))
        returns[:, 0] = 0.8 * f1 + 0.3 * f2 + noise[:, 0]
        returns[:, 1] = 0.7 * f1 + 0.4 * f2 + noise[:, 1]
        returns[:, 2] = 0.6 * f1 - 0.5 * f2 + noise[:, 2]
        returns[:, 3] = -0.5 * f1 + 0.6 * f2 + noise[:, 3]
        returns[:, 4] = -0.7 * f1 - 0.4 * f2 + noise[:, 4]
        
        self.returns_df = pd.DataFrame(returns, index=dates, columns=self.assets)

    def test_pca_fit_and_eigenvalues(self):
        pca = PCAFactorEngine(use_correlation=True)
        pca.fit(self.returns_df)
        
        eigvals = pca.get_eigenvalues()
        self.assertEqual(len(eigvals), 5)
        # Eigenvalues sorted descending
        self.assertTrue(np.all(np.diff(eigvals.values) <= 0))
        
        var_ratio = pca.get_explained_variance_ratio()
        self.assertAlmostEqual(var_ratio.sum(), 1.0, places=6)
        
        cum_var = pca.get_cumulative_explained_variance()
        self.assertAlmostEqual(cum_var.iloc[-1], 1.0, places=6)
        self.assertTrue(np.all(np.diff(cum_var.values) >= 0))

    def test_pca_covariance_mode(self):
        pca = PCAFactorEngine(use_correlation=False)
        pca.fit(self.returns_df)
        eigvals = pca.get_eigenvalues()
        self.assertEqual(len(eigvals), 5)
        self.assertTrue(np.all(np.diff(eigvals.values) <= 0))
        rec = pca.reconstruct_returns(self.returns_df)
        self.assertTrue(np.allclose(rec.values, self.returns_df.values, atol=1e-12))

    def test_pca_loadings_orthonormality(self):
        pca = PCAFactorEngine(use_correlation=True).fit(self.returns_df)
        loadings = pca.get_loadings()
        
        # Loadings matrix V should be orthonormal: V^T V = I
        V = loadings.values
        identity_test = V.T @ V
        self.assertTrue(np.allclose(identity_test, np.eye(5), atol=1e-7))

    def test_pca_transformation_and_principal_portfolios(self):
        pca = PCAFactorEngine(use_correlation=True).fit(self.returns_df)
        factors = pca.transform(self.returns_df, n_components=3)
        self.assertEqual(factors.shape, (300, 3))
        self.assertEqual(list(factors.columns), ['PC1', 'PC2', 'PC3'])
        
        # Test different normalizations
        for norm in ['unit_sum', 'unit_l1', 'unit_l2']:
            port_weights = pca.get_principal_portfolios(n_components=3, normalize=norm)
            self.assertEqual(port_weights.shape, (5, 3))
            if norm == 'unit_l1':
                for col in port_weights.columns:
                    self.assertAlmostEqual(np.sum(np.abs(port_weights[col])), 1.0, places=6)
            elif norm == 'unit_l2':
                for col in port_weights.columns:
                    self.assertAlmostEqual(np.linalg.norm(port_weights[col]), 1.0, places=6)
            
        port_returns = pca.principal_portfolio_returns(self.returns_df, n_components=3)
        self.assertEqual(port_returns.shape, (300, 3))

    def test_pca_reconstruction_lossless(self):
        pca = PCAFactorEngine(use_correlation=True).fit(self.returns_df)
        reconstructed = pca.reconstruct_returns(self.returns_df, n_components=5)
        max_error = np.max(np.abs(reconstructed.values - self.returns_df.values))
        self.assertLess(max_error, 1e-12)

    def test_scree_test_and_dimension_reduction(self):
        pca = PCAFactorEngine(use_correlation=True).fit(self.returns_df)
        k = pca.scree_test(threshold=0.80)
        self.assertGreaterEqual(k, 1)
        self.assertLessEqual(k, 5)
        self.assertGreaterEqual(pca.get_cumulative_explained_variance().iloc[k-1], 0.80)
        
        reduced_df, k_ret = pca.dimension_reduction(self.returns_df, threshold=0.80)
        self.assertEqual(reduced_df.shape[1], k_ret)
        self.assertEqual(k, k_ret)

    def test_pca_unfitted_errors(self):
        pca = PCAFactorEngine()
        with self.assertRaises(RuntimeError):
            pca.get_eigenvalues()
        with self.assertRaises(RuntimeError):
            pca.transform(self.returns_df)
        with self.assertRaises(RuntimeError):
            pca.reconstruct_returns(self.returns_df)


class TestClusteringAndTreeSeriation(unittest.TestCase):
    """Test suite for correlation distance, clustering, and quasi-diagonalization."""

    def setUp(self):
        # 4 assets with clear block correlation structure: (A, B) high, (C, D) high
        self.corr_df = pd.DataFrame([
            [1.0, 0.85, 0.10, 0.15],
            [0.85, 1.0, 0.12, 0.18],
            [0.10, 0.12, 1.0, 0.90],
            [0.15, 0.18, 0.90, 1.0]
        ], index=['A', 'B', 'C', 'D'], columns=['A', 'B', 'C', 'D'])

    def test_correlation_distance(self):
        dist_df = correlation_distance(self.corr_df)
        self.assertIsInstance(dist_df, pd.DataFrame)
        self.assertEqual(dist_df.shape, (4, 4))
        
        # Diagonal is zero
        self.assertTrue(np.allclose(np.diag(dist_df.values), 0.0))
        # Symmetric
        self.assertTrue(np.allclose(dist_df.values, dist_df.values.T))
        # Bounded [0, 1]
        self.assertTrue(np.all(dist_df.values >= 0.0) and np.all(dist_df.values <= 1.0))
        
        # Exact values check
        # rho = 1 -> d = 0
        # rho = 0 -> d = sqrt(0.5)
        # rho = -1 -> d = 1.0
        test_corr = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.5], [-1.0, 0.5, 1.0]])
        test_dist = correlation_distance(test_corr)
        self.assertAlmostEqual(test_dist[0, 1], np.sqrt(0.5))
        self.assertAlmostEqual(test_dist[0, 2], 1.0)
        self.assertAlmostEqual(test_dist[0, 0], 0.0)

    def test_hierarchical_clustering_linkage(self):
        for method in ['ward', 'single', 'complete', 'average']:
            Z = hierarchical_correlation_clustering(self.corr_df, method=method)
            self.assertIsInstance(Z, np.ndarray)
            self.assertEqual(Z.shape, (3, 4))

    def test_hierarchical_clustering_invalid_method(self):
        with self.assertRaises(ValueError):
            hierarchical_correlation_clustering(self.corr_df, method='invalid_method')

    def test_quasi_diagonalize_seriation(self):
        Z = hierarchical_correlation_clustering(self.corr_df, method='ward')
        
        # Test index ordering
        order_idx = quasi_diagonalize(Z)
        self.assertEqual(len(order_idx), 4)
        self.assertEqual(set(order_idx), {0, 1, 2, 3})
        
        # Test labels ordering
        order_labels = quasi_diagonalize(Z, labels=list(self.corr_df.columns))
        self.assertEqual(len(order_labels), 4)
        self.assertEqual(set(order_labels), {'A', 'B', 'C', 'D'})
        
        # Verify cluster grouping: A and B are adjacent, C and D are adjacent
        pos_A = order_labels.index('A')
        pos_B = order_labels.index('B')
        pos_C = order_labels.index('C')
        pos_D = order_labels.index('D')
        
        self.assertEqual(abs(pos_A - pos_B), 1)
        self.assertEqual(abs(pos_C - pos_D), 1)

    def test_quasi_diagonalize_mismatched_labels(self):
        Z = hierarchical_correlation_clustering(self.corr_df, method='ward')
        with self.assertRaises(ValueError):
            quasi_diagonalize(Z, labels=['A', 'B'])


if __name__ == '__main__':
    unittest.main()
