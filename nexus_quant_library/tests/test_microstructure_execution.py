"""
Unit tests for Market Microstructure, L2 Limit Order Book, Optimal Execution, VPIN, Hawkes & Avellaneda-Stoikov.
"""

import unittest
import numpy as np
import pandas as pd
from nexus_quant.microstructure_execution.order_book import (
    LimitOrderBook,
    Order,
)
from nexus_quant.microstructure_execution.optimal_execution import (
    AlmgrenChrissModel,
    BenchmarkExecutors,
    ImplementationShortfallAttributor,
)
from nexus_quant.microstructure_execution.vpin import (
    VPINEngine,
)
from nexus_quant.microstructure_execution.hawkes_process import (
    MultivariateHawkesEngine,
)
from nexus_quant.microstructure_execution.avellaneda_stoikov import (
    AvellanedaStoikovMarketMaker,
)


class TestMicrostructureExecutionSuite(unittest.TestCase):
    """Test suite for Market Microstructure and Execution Engine."""

    def test_limit_order_book_fifo_and_micro_price(self):
        """Test L2 Order Book insertion, matching, OBI, and Volume-Weighted Micro-Price."""
        lob = LimitOrderBook(name="TEST_BOOK")

        # Insert bids and asks
        lob.add_limit_order(Order("B1", "buy", 99.90, 500.0, 1.0))
        lob.add_limit_order(Order("B2", "buy", 100.00, 300.0, 1.1))  # best bid
        lob.add_limit_order(Order("A1", "sell", 100.10, 200.0, 1.2)) # best ask
        lob.add_limit_order(Order("A2", "sell", 100.20, 600.0, 1.3))

        self.assertEqual(lob.best_bid, 100.00)
        self.assertEqual(lob.best_ask, 100.10)
        self.assertAlmostEqual(lob.spread, 0.10, places=4)
        self.assertAlmostEqual(lob.mid_price, 100.05, places=4)

        # OBI = (300 - 200) / (300 + 200) = +0.20
        self.assertAlmostEqual(lob.order_book_imbalance, 0.20, places=4)

        # Micro-price = (300*100.10 + 200*100.00) / 500 = 100.06
        self.assertAlmostEqual(lob.micro_price, 100.06, places=4)

        # Market order execution
        trades, filled = lob.execute_market_order("buy", 150.0, timestamp=2.0)
        self.assertEqual(filled, 150.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 100.10)

        # Order cancellation
        cancelled = lob.cancel_order("B1")
        self.assertTrue(cancelled)

    def test_almgren_chriss_optimal_execution(self):
        """Test Almgren-Chriss (2000) optimal trajectory, expected shortfall & risk trade-off."""
        model = AlmgrenChrissModel(
            total_shares=1_000_000,
            horizon=1.0,
            n_intervals=20,
            volatility=0.30,
            temp_impact=2.5e-6,
            perm_impact=2.5e-7,
        )

        traj_risk_neutral = model.solve_trajectory(risk_aversion=1e-8)
        traj_risk_averse = model.solve_trajectory(risk_aversion=1e-4)

        # Terminal holdings must be zero
        self.assertAlmostEqual(traj_risk_neutral.holdings[-1], 0.0, places=4)
        self.assertAlmostEqual(traj_risk_averse.holdings[-1], 0.0, places=4)

        # Sum of trade sizes equals initial total shares
        self.assertAlmostEqual(np.sum(traj_risk_neutral.trade_sizes), 1_000_000.0, delta=1.0)
        self.assertAlmostEqual(np.sum(traj_risk_averse.trade_sizes), 1_000_000.0, delta=1.0)

        # Risk-averse execution trades faster in early intervals
        self.assertGreater(traj_risk_averse.trade_sizes[0], traj_risk_neutral.trade_sizes[0])
        # Higher risk aversion yields lower variance / risk of shortfall
        self.assertLess(traj_risk_averse.std_shortfall, traj_risk_neutral.std_shortfall)

    def test_implementation_shortfall_attribution(self):
        """Test Perold Implementation Shortfall decomposition."""
        trades = np.array([250_000, 250_000, 250_000, 250_000])
        exec_prices = np.array([100.05, 100.10, 100.15, 100.20])

        attr = ImplementationShortfallAttributor.attribute(
            total_shares=1_000_000,
            decision_price=100.00,
            arrival_price=100.02,
            terminal_price=100.25,
            trade_sizes=trades,
            execution_prices=exec_prices,
            side="buy",
        )

        self.assertGreater(attr["Total_Shortfall_USD"], 0.0)
        self.assertGreater(attr["Delay_Cost_USD"], 0.0)
        self.assertGreater(attr["Temporary_Impact_USD"], 0.0)

    def test_vpin_computation_and_toxicity_bounds(self):
        """Test Volume Synchronized Probability of Toxicity (VPIN) bounds and BVC classification."""
        np.random.seed(42)
        n_ticks = 1000
        # Simulating heavy sell pressure
        p0 = 100.0
        price_shocks = np.random.randn(n_ticks) * 0.10 - 0.02
        prices = p0 + np.cumsum(price_shocks)
        volumes = np.random.exponential(100.0, n_ticks) + 10.0

        vpin_engine = VPINEngine(n_buckets=20, sigma_window=10)
        res = vpin_engine.compute_vpin_from_ticks(prices, volumes, bucket_size=500.0)

        # VPIN must strictly lie in [0, 1]
        self.assertTrue(np.all((res.vpin_series >= 0.0) & (res.vpin_series <= 1.0)))
        self.assertGreater(res.mean_vpin, 0.0)
        self.assertIn(res.toxicity_regime, [
            "NORMAL_BALANCED_FLOW",
            "HIGH_TOXICITY_ADVERSE_SELECTION",
            "CRITICAL_TOXICITY_FLASH_CRASH_RISK",
        ])

    def test_multivariate_hawkes_calibration(self):
        """Test Multivariate Hawkes Process MLE estimation and branching matrix."""
        # Synthetic event times
        np.random.seed(42)
        news_events = np.sort(np.random.uniform(0, 100, 15))
        # Trades clustered following news events
        jump_events = []
        for n_t in news_events:
            n_clust = np.random.poisson(3)
            jump_events.extend(n_t + np.random.exponential(0.5, n_clust))
        jump_events = np.sort(np.array(jump_events))

        engine = MultivariateHawkesEngine(dimension=2)
        fit_res = engine.fit([news_events, jump_events], horizon=105.0, max_iter=50)

        self.assertGreater(fit_res.alpha[1, 0], 0.0)
        self.assertLess(fit_res.endogeneity_ratio, 1.0)

        half_life = engine.impulse_response_half_life(source_node=0, target_node=1)
        self.assertGreater(half_life, 0.0)

    def test_avellaneda_stoikov_market_making(self):
        """Test Avellaneda-Stoikov reservation price skew and session simulation."""
        mm = AvellanedaStoikovMarketMaker(
            risk_aversion_gamma=0.10,
            order_book_liquidity_kappa=1.50,
            asset_volatility_sigma=0.20,
            time_horizon_T=1.0,
            max_inventory_limit=20,
        )

        spot = 100.0
        # Zero inventory: reservation price equals spot
        r_zero = mm.reservation_price(spot, inventory=0, current_time=0.5)
        self.assertEqual(r_zero, spot)

        # Long inventory: reservation price < spot (downward skew)
        r_long = mm.reservation_price(spot, inventory=+5, current_time=0.5)
        self.assertLess(r_long, spot)

        # Short inventory: reservation price > spot (upward skew)
        r_short = mm.reservation_price(spot, inventory=-5, current_time=0.5)
        self.assertGreater(r_short, spot)

        # Session simulation
        sim_res = mm.simulate_session(initial_price=100.0, n_steps=200, seed=42)
        self.assertEqual(len(sim_res.pnl_path), 201)
        self.assertLessEqual(sim_res.max_inventory_held, 20)


if __name__ == "__main__":
    unittest.main()
