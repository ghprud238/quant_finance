"""Unit tests for Prediction Market Mispricing & Kelly Execution Engine (Module 51)."""

import unittest
import numpy as np
from quant_mechanics.data.loader import (
    generate_prediction_market_order_books,
    BinaryContractOrderBook,
    OrderBookLevel,
)
from quant_mechanics.prediction_markets.mispricing import (
    PredictionMarketArbitrageEngine,
    ArbitrageType,
)


class TestPredictionMarkets(unittest.TestCase):

    def setUp(self):
        self.engine = PredictionMarketArbitrageEngine(
            default_fractional_kelly=0.50,
            default_latency_half_life_sec=2.0,
            min_net_roi_threshold=0.001,
        )

    def test_order_book_walking_and_slippage(self):
        """Tests L2 order book walking depth and slippage calculation."""
        levels = [
            OrderBookLevel(price=0.45, size=1000.0),
            OrderBookLevel(price=0.47, size=2000.0),
            OrderBookLevel(price=0.50, size=3000.0),
        ]
        
        # Test fill within top level
        res_small = self.engine.walk_order_book(levels, target_size=500.0)
        self.assertEqual(res_small.executed_size, 500.0)
        self.assertEqual(res_small.effective_price, 0.45)
        self.assertEqual(res_small.slippage_bps, 0.0)
        self.assertTrue(res_small.is_fully_filled)

        # Test fill walking multiple levels (1000 @ 0.45 + 1000 @ 0.47 = $920 / 2000 = 0.46)
        res_med = self.engine.walk_order_book(levels, target_size=2000.0)
        self.assertEqual(res_med.executed_size, 2000.0)
        self.assertAlmostEqual(res_med.effective_price, 0.46, places=4)
        self.assertGreater(res_med.slippage_bps, 0.0)
        self.assertTrue(res_med.is_fully_filled)

        # Test fill exceeding book capacity
        res_excess = self.engine.walk_order_book(levels, target_size=10_000.0)
        self.assertEqual(res_excess.executed_size, 6000.0)
        self.assertFalse(res_excess.is_fully_filled)

    def test_intra_venue_arbitrage(self):
        """Tests intra-venue arbitrage detection when Yes Ask + No Ask < 1.0."""
        # Create mispriced book where Yes Ask = 0.42 and No Ask = 0.54 (Sum = 0.96 < 1.00)
        mispriced_book = BinaryContractOrderBook(
            contract_id="TEST_INTRA",
            event_title="Test Event",
            venue="Polymarket",
            timestamp_ns=1000,
            yes_bids=[OrderBookLevel(price=0.40, size=5000.0)],
            yes_asks=[OrderBookLevel(price=0.42, size=5000.0)],
            no_bids=[OrderBookLevel(price=0.52, size=5000.0)],
            no_asks=[OrderBookLevel(price=0.54, size=5000.0)],
            fee_rate=0.00,
            gas_cost_usd=0.02,
        )

        arb = self.engine.check_intra_venue_arbitrage(mispriced_book, target_size=1000.0)
        self.assertIsNotNone(arb)
        self.assertEqual(arb.arb_type, ArbitrageType.INTRA_VENUE)
        self.assertAlmostEqual(arb.gross_combined_price, 0.96, places=4)
        self.assertAlmostEqual(arb.gross_edge_pct, 4.0, places=2)
        self.assertGreater(arb.net_profit_usd, 0.0)
        self.assertTrue(arb.is_profitable)

    def test_cross_venue_arbitrage(self):
        """Tests cross-venue arbitrage between Polymarket and Kalshi."""
        book_poly = BinaryContractOrderBook(
            contract_id="FED_DECISION",
            event_title="Fed Cuts Rates",
            venue="Polymarket",
            timestamp_ns=1000,
            yes_bids=[OrderBookLevel(price=0.44, size=5000.0)],
            yes_asks=[OrderBookLevel(price=0.46, size=5000.0)],
            no_bids=[OrderBookLevel(price=0.52, size=5000.0)],
            no_asks=[OrderBookLevel(price=0.56, size=5000.0)],
            fee_rate=0.00,
            gas_cost_usd=0.015,
        )
        book_kalshi = BinaryContractOrderBook(
            contract_id="FED_DECISION",
            event_title="Fed Cuts Rates",
            venue="Kalshi",
            timestamp_ns=1000,
            yes_bids=[OrderBookLevel(price=0.56, size=5000.0)],
            yes_asks=[OrderBookLevel(price=0.60, size=5000.0)],
            no_bids=[OrderBookLevel(price=0.38, size=5000.0)],
            no_asks=[OrderBookLevel(price=0.44, size=5000.0)],
            fee_rate=0.01,
            gas_cost_usd=0.00,
        )

        # Yes on Poly (0.46) + No on Kalshi (0.44) = 0.90 < 1.00 -> 10% gross edge!
        arbs = self.engine.check_cross_venue_arbitrage(book_poly, book_kalshi, target_size=1000.0)
        self.assertGreaterEqual(len(arbs), 1)
        best_arb = arbs[0]
        self.assertEqual(best_arb.arb_type, ArbitrageType.CROSS_VENUE)
        self.assertAlmostEqual(best_arb.gross_combined_price, 0.90, places=2)
        self.assertAlmostEqual(best_arb.gross_edge_pct, 10.0, places=1)
        self.assertGreater(best_arb.net_profit_usd, 50.0)
        self.assertTrue(best_arb.is_profitable)

    def test_kelly_criterion_sizing(self):
        """Tests continuous and discrete Kelly criterion calculations."""
        # Win prob = 60%, price = $0.50 -> odds b = 1.0 -> full kelly f* = (1*0.6 - 0.4)/1 = 0.20
        res = self.engine.calculate_kelly_fraction(
            true_win_prob=0.60,
            market_price=0.50,
            bankroll_usd=100_000.0,
            fractional_multiplier=0.50,
        )
        self.assertAlmostEqual(res.full_kelly_fraction, 0.20, places=4)
        self.assertAlmostEqual(res.recommended_fraction, 0.10, places=4)
        self.assertAlmostEqual(res.recommended_stake_usd, 10_000.0, places=2)
        self.assertGreater(res.expected_growth_rate, 0.0)

        # Zero edge case: win prob = 50%, price = $0.50 -> kelly = 0
        res_zero = self.engine.calculate_kelly_fraction(
            true_win_prob=0.50,
            market_price=0.50,
            bankroll_usd=100_000.0,
        )
        self.assertEqual(res_zero.full_kelly_fraction, 0.0)
        self.assertEqual(res_zero.recommended_stake_usd, 0.0)

    def test_latency_alpha_decay(self):
        """Tests exponential decay of executable arbitrage edge."""
        edge_0 = 10.0  # 10%
        # After 1 half-life (2.0s), edge should be exactly 5.0%
        edge_1hl = self.engine.calculate_latency_alpha_decay(
            initial_edge_pct=edge_0,
            elapsed_sec=2.0,
            half_life_sec=2.0,
        )
        self.assertAlmostEqual(edge_1hl, 5.0, places=4)

        # After 2 half-lives (4.0s), edge should be 2.5%
        edge_2hl = self.engine.calculate_latency_alpha_decay(
            initial_edge_pct=edge_0,
            elapsed_sec=4.0,
            half_life_sec=2.0,
        )
        self.assertAlmostEqual(edge_2hl, 2.5, places=4)


if __name__ == "__main__":
    unittest.main()
