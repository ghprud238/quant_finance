import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
"""Unit Tests for Module 26: Limit Order Book & Microstructure Simulator."""

import unittest
import numpy as np
import pandas as pd

from interview_quant.microstructure.order_book import Order, LimitOrderBook, MatchResult, Level2Snapshot
from interview_quant.microstructure.simulator import MarketMicrostructureSimulator


class TestLimitOrderBook(unittest.TestCase):
    """Tests L2 matching logic, priority rules, cancellations, and order book metrics."""

    def setUp(self):
        self.lob = LimitOrderBook(name="Test_LOB")

    def test_empty_book_properties(self):
        """Verifies properties on an empty order book."""
        self.assertIsNone(self.lob.best_bid)
        self.assertIsNone(self.lob.best_ask)
        self.assertIsNone(self.lob.spread)
        self.assertEqual(self.lob.total_bid_volume, 0.0)
        self.assertEqual(self.lob.total_ask_volume, 0.0)
        self.assertEqual(self.lob.order_book_imbalance, 0.0)

    def test_limit_order_insertion_and_priority(self):
        """Tests resting limit order insertion and FIFO queue priority."""
        o1 = Order("B1", "buy", 100.0, 50.0, 1.0)
        o2 = Order("B2", "buy", 100.0, 30.0, 2.0)
        o3 = Order("B3", "buy", 100.5, 20.0, 3.0)  # Higher price level

        self.lob.add_limit_order(o1)
        self.lob.add_limit_order(o2)
        self.lob.add_limit_order(o3)

        self.assertEqual(self.lob.best_bid, 100.5)
        self.assertEqual(self.lob.total_bid_volume, 100.0)
        self.assertEqual(self.lob.best_bid_volume, 20.0)

        # Market sell should hit B3 first (price priority), then B1 (time priority)
        trades, filled = self.lob.execute_market_order("sell", 60.0, timestamp=4.0)
        self.assertEqual(filled, 60.0)
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].buyer_id, "B3")
        self.assertEqual(trades[0].volume, 20.0)
        self.assertEqual(trades[1].buyer_id, "B1")
        self.assertEqual(trades[1].volume, 40.0)

        # B1 should have 10 remaining volume
        self.assertEqual(self.lob.best_bid, 100.0)
        self.assertEqual(self.lob.best_bid_volume, 40.0)  # 10 (B1) + 30 (B2)

    def test_marketable_limit_order_crossing(self):
        """Tests limit order that immediately crosses the spread."""
        ask = Order("A1", "sell", 100.2, 100.0, 1.0)
        self.lob.add_limit_order(ask)

        crossing_bid = Order("B1", "buy", 100.5, 40.0, 2.0)
        trades = self.lob.add_limit_order(crossing_bid)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 100.2)
        self.assertEqual(trades[0].volume, 40.0)
        self.assertEqual(self.lob.best_ask_volume, 60.0)
        self.assertIsNone(self.lob.best_bid)

    def test_order_cancellation(self):
        """Tests cancellation of a resting order."""
        o1 = Order("A1", "sell", 101.0, 50.0, 1.0)
        o2 = Order("A2", "sell", 101.0, 50.0, 2.0)
        self.lob.add_limit_order(o1)
        self.lob.add_limit_order(o2)

        self.assertEqual(self.lob.best_ask_volume, 100.0)
        success = self.lob.cancel_order("A1")
        self.assertTrue(success)
        self.assertEqual(self.lob.best_ask_volume, 50.0)

        # Cancel non-existent order
        self.assertFalse(self.lob.cancel_order("NON_EXISTENT"))

    def test_microstructure_metrics(self):
        """Tests exact calculation of spread, micro-price, and order book imbalance."""
        self.lob.add_limit_order(Order("B1", "buy", 100.0, 300.0, 1.0))
        self.lob.add_limit_order(Order("A1", "sell", 101.0, 100.0, 1.0))

        self.assertEqual(self.lob.spread, 1.0)
        self.assertEqual(self.lob.mid_price, 100.5)

        # Imbalance: (300 - 100) / (300 + 100) = 200 / 400 = 0.50
        self.assertAlmostEqual(self.lob.order_book_imbalance, 0.50, places=4)

        # Micro-price: (V_b * P_a + V_a * P_b) / (V_b + V_a)
        # = (300 * 101 + 100 * 100) / 400 = (30300 + 10000) / 400 = 40300 / 400 = 100.75
        self.assertAlmostEqual(self.lob.micro_price, 100.75, places=4)

    def test_level2_snapshot_table(self):
        """Tests formatted Level 2 snapshot table output."""
        self.lob.add_limit_order(Order("B1", "buy", 99.9, 100.0, 1.0))
        self.lob.add_limit_order(Order("B2", "buy", 99.8, 200.0, 1.0))
        self.lob.add_limit_order(Order("A1", "sell", 100.1, 150.0, 1.0))
        self.lob.add_limit_order(Order("A2", "sell", 100.2, 250.0, 1.0))

        df = self.lob.get_snapshot_table(depth=2)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, 'Bid_Price'], 99.9)
        self.assertEqual(df.loc[0, 'Ask_Price'], 100.1)


class TestMarketMicrostructureSimulator(unittest.TestCase):
    """Tests Poisson microstructure simulation execution."""

    def test_simulation_run(self):
        sim = MarketMicrostructureSimulator(tick_size=0.01, lambda_limit=5.0, lambda_market=2.0, lambda_cancel=1.0)
        lob, df_log = sim.simulate(n_events=100, initial_mid=100.0, seed=42)

        self.assertEqual(len(df_log), 100)
        self.assertTrue('mid_price' in df_log.columns)
        self.assertTrue('micro_price' in df_log.columns)
        self.assertTrue('order_book_imbalance' in df_log.columns)
        self.assertGreater(lob.total_bid_volume + lob.total_ask_volume, 0)


if __name__ == '__main__':
    unittest.main()
