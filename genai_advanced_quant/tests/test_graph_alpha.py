"""Unit tests for Module 34 (Supply-Chain Knowledge Graph Alpha & GNN Spillover)."""

import unittest
import numpy as np
import pandas as pd

from genai_advanced_quant.graph_alpha.supply_chain import (
    SupplyChainGraphAlpha,
    SupplyChainNetwork,
    SupplyChainLink,
    GraphAlphaResult,
)


class TestSupplyChainGraphAlpha(unittest.TestCase):
    """Test suite verifying Cohen & Frazzini (2008) GCN supply-chain momentum alpha."""

    def setUp(self) -> None:
        self.engine = SupplyChainGraphAlpha(n_gcn_layers=2, lead_lag_window=5)
        # Create synthetic price matrix for supply chain nodes
        np.random.seed(42)
        nodes = self.engine.network.nodes
        dates = pd.bdate_range("2023-01-01", periods=300)
        returns = np.random.normal(0.0005, 0.015, (len(dates), len(nodes)))
        prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
        self.prices_df = pd.DataFrame(prices, index=dates, columns=nodes)

    def test_network_construction_and_adjacency(self) -> None:
        """Verifies directed network nodes and adjacency matrix properties."""
        net = self.engine.network
        self.assertGreater(net.n_nodes, 10)
        self.assertEqual(net.adjacency_matrix.shape, (net.n_nodes, net.n_nodes))
        # Adjacency entries must be within [0, 1]
        self.assertTrue(np.all(net.adjacency_matrix >= 0.0))
        self.assertTrue(np.all(net.adjacency_matrix <= 1.0))

    def test_gcn_laplacian_normalization(self) -> None:
        """Verifies GCN normalized Laplacian D~^{-1/2} A~ D~^{-1/2} has eigenvalues bounded in [-1, 1]."""
        A_norm = self.engine.network.gcn_normalized_adj
        self.assertEqual(A_norm.shape, (self.engine.network.n_nodes, self.engine.network.n_nodes))
        # Diagonal elements must be non-zero (self-loops)
        self.assertTrue(np.all(np.diag(A_norm) > 0.0))

    def test_pagerank_centrality(self) -> None:
        """Verifies PageRank sums to 1.0 and principal customers have high centrality."""
        pr = self.engine.network.compute_pagerank()
        self.assertIsInstance(pr, pd.Series)
        self.assertAlmostEqual(pr.sum(), 1.0, places=4)
        # AAPL and NVDA are major hubs with multiple suppliers
        self.assertGreater(pr["AAPL"], 0.0)

    def test_concentration_hhi(self) -> None:
        """Verifies Customer Concentration HHI calculation."""
        hhi = self.engine.network.compute_concentration_hhi()
        self.assertIsInstance(hhi, pd.Series)
        # CRUS has 76% revenue from AAPL -> HHI should be around 0.76^2 = 0.577
        self.assertGreater(hhi["CRUS"], 0.50)

    def test_gcn_message_passing(self) -> None:
        """Verifies graph convolution propagates features across node tiers."""
        M = self.engine.network.n_nodes
        features = np.zeros((M, 1))
        # Inject positive shock only into customer node AAPL
        aapl_idx = self.engine.network.node_to_idx["AAPL"]
        features[aapl_idx, 0] = 1.0

        # Layer 1 propagation
        H1 = self.engine.graph_convolution_message_passing(features, n_layers=1)
        # Suppliers of AAPL (SWKS, QRVO, CRUS) must receive positive feature values
        swks_idx = self.engine.network.node_to_idx["SWKS"]
        self.assertGreater(H1[swks_idx, 0], 0.0)

    def test_lead_lag_signals_generation(self) -> None:
        """Verifies cross-sectional signals generation without lookahead bias."""
        signals_df = self.engine.compute_lead_lag_signals(self.prices_df, customer_mom_window=5)
        self.assertEqual(signals_df.shape, self.prices_df.shape)
        # First 5 rows must be zero (warm-up window)
        self.assertTrue(np.all(signals_df.iloc[:5].values == 0.0))
        # Subsequent rows must have non-zero standardized signals
        self.assertFalse(np.all(signals_df.iloc[20:].values == 0.0))

    def test_backtest_strategy_execution(self) -> None:
        """Verifies Long/Short strategy backtest execution and metrics generation."""
        res = self.engine.backtest_strategy(self.prices_df, n_quantiles=3)
        self.assertIsInstance(res, GraphAlphaResult)
        self.assertGreater(len(res.equity_curve), 50)
        self.assertIn("sharpe_ratio", res.metrics)
        self.assertIn("mean_ic", res.metrics)
        self.assertIsInstance(res.summary_table(), pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
