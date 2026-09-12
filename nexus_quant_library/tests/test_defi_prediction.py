"""
Comprehensive Unit Tests for DeFi AMMs, LVR, Perpetual Basis & Prediction Market Arbitrage (Tracks 9, 11).
"""

import unittest
import numpy as np
import pandas as pd

from nexus_quant.defi_prediction_markets import (
    ConcentratedLiquidityAMM,
    ConstantProductAMM,
    ImpermanentLossCalculator,
    LossVersusRebalancingEngine,
    PerpetualFundingEngine,
    CashAndCarryBasisTrader,
    PredictionMarketArbitrageEngine,
    BinaryOrderBook,
    OrderBookLevel,
)


class TestDeFiPrediction(unittest.TestCase):

    def test_uniswap_v2_constant_product(self):
        v2 = ConstantProductAMM(reserve_x=1000.0, reserve_y=3_000_000.0, fee_rate=0.003)
        self.assertAlmostEqual(v2.spot_price, 3000.0, places=2)
        swap = v2.swap_exact_in(amount_in_x=10.0)
        self.assertGreater(swap.amount_out, 0.0)
        self.assertLess(v2.spot_price, 3000.0)
        self.assertGreater(swap.price_impact_pct, 0.0)

    def test_uniswap_v3_concentrated_liquidity(self):
        v3 = ConcentratedLiquidityAMM(current_price=3000.0, fee_tier=0.003)
        pos = v3.mint_position(owner="Alice", price_lower=2500.0, price_upper=3500.0, amount_x=10.0, amount_y=30_000.0)
        self.assertGreater(pos.liquidity, 0.0)
        
        eff = v3.capital_efficiency_multiplier(2500.0, 3500.0)
        self.assertGreater(eff, 5.0)

        swap_x = v3.swap(amount_in=2.0, token_in="ETH")
        self.assertGreater(swap_x.amount_out, 0.0)
        self.assertLess(v3.current_price, 3000.0)

        swap_y = v3.swap(amount_in=10_000.0, token_in="USDC")
        self.assertGreater(swap_y.amount_out, 0.0)
        self.assertGreater(v3.current_price, 2900.0)

    def test_impermanent_loss_formulas(self):
        # 25% price increase -> IL ~ -0.6%
        il_v2 = ImpermanentLossCalculator.standard_cfmm_il(1.25)
        self.assertLess(il_v2, 0.0)
        self.assertAlmostEqual(il_v2, -0.006, places=2)

        # Concentrated IL within range
        il_v3 = ImpermanentLossCalculator.concentrated_liquidity_il(3000.0, 3500.0, 2500.0, 4000.0)
        self.assertLess(il_v3, 0.0)
        self.assertLess(il_v3, il_v2)  # Concentrated IL is magnified

    def test_lvr_engine_and_breakeven_vol(self):
        prices = pd.Series(3000.0 * np.exp(np.cumsum(np.random.normal(0, 0.005, 500))))
        volumes = pd.Series(np.random.uniform(500_000, 2_000_000, 500))
        lvr_engine = LossVersusRebalancingEngine(fee_rate=0.0030)
        sim = lvr_engine.simulate_lp_performance(prices, volumes, initial_capital_usd=100_000.0)
        self.assertGreater(sim.total_lvr_usd, 0.0)
        self.assertGreater(sim.total_fee_revenue_usd, 0.0)
        self.assertGreater(sim.breakeven_volatility_annual, 0.0)
        self.assertIn("Initial LP Capital ($)", sim.summary_table["Metric"].values)

    def test_perpetual_funding_rate_clamps(self):
        fr_normal = PerpetualFundingEngine.calculate_funding_rate(0.0008, 0.0003)
        self.assertLessEqual(fr_normal, 0.0075)
        self.assertGreaterEqual(fr_normal, -0.0075)

        fr_extreme = PerpetualFundingEngine.calculate_funding_rate(0.025, 0.0003)
        self.assertAlmostEqual(fr_extreme, 0.0075, places=4)

        fr_neg = PerpetualFundingEngine.calculate_funding_rate(-0.025, 0.0003)
        self.assertAlmostEqual(fr_neg, -0.0075, places=4)

    def test_cash_and_carry_basis_trading(self):
        n = 300
        df = pd.DataFrame({
            "spot_price": 3000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))),
            "funding_rate": np.random.uniform(0.0001, 0.0005, n),
        })
        trader = CashAndCarryBasisTrader(initial_capital_usd=100_000.0)
        res = trader.backtest(df)
        self.assertGreater(res.final_equity_usd, res.initial_capital_usd)
        self.assertGreater(res.cagr, 0.0)
        self.assertGreater(res.sharpe_ratio, 0.0)
        self.assertLessEqual(res.max_drawdown_pct, 0.05)
        self.assertIn("Cash-and-Carry Delta-Neutral Performance", res.summary())

    def test_prediction_market_intra_venue_arbitrage(self):
        engine = PredictionMarketArbitrageEngine()
        book = BinaryOrderBook(
            venue="Polymarket",
            contract_id="ELECTION_YES",
            yes_bids=[OrderBookLevel(0.44, 5000)],
            yes_asks=[OrderBookLevel(0.45, 5000)],
            no_bids=[OrderBookLevel(0.47, 5000)],
            no_asks=[OrderBookLevel(0.48, 5000)],
        )
        intra = engine.check_intra_venue_arbitrage(book, target_size=1000.0)
        self.assertIsNotNone(intra)
        self.assertTrue(intra.is_executable)
        self.assertAlmostEqual(intra.gross_edge_pct, 7.0, places=1)
        self.assertGreater(intra.net_profit_usd, 50.0)

    def test_prediction_market_cross_venue_arbitrage(self):
        engine = PredictionMarketArbitrageEngine()
        book_a = BinaryOrderBook(
            venue="Polymarket",
            contract_id="FED_RATE",
            yes_bids=[OrderBookLevel(0.50, 5000)],
            yes_asks=[OrderBookLevel(0.52, 5000)],
            no_bids=[OrderBookLevel(0.46, 5000)],
            no_asks=[OrderBookLevel(0.48, 5000)],
        )
        book_b = BinaryOrderBook(
            venue="Kalshi",
            contract_id="FED_RATE",
            yes_bids=[OrderBookLevel(0.58, 5000)],
            yes_asks=[OrderBookLevel(0.60, 5000)],
            no_bids=[OrderBookLevel(0.38, 5000)],
            no_asks=[OrderBookLevel(0.40, 5000)],
        )
        arbs = engine.check_cross_venue_arbitrage(book_a, book_b, target_size=1000.0)
        self.assertGreater(len(arbs), 0)
        exec_arbs = [a for a in arbs if a.is_executable]
        self.assertGreater(len(exec_arbs), 0)
        self.assertAlmostEqual(exec_arbs[0].gross_edge_pct, 8.0, places=1)

    def test_kelly_criterion_sizing(self):
        engine = PredictionMarketArbitrageEngine()
        kelly = engine.calculate_kelly_fraction(true_win_prob=0.999, market_price=0.93, bankroll_usd=100_000.0)
        self.assertGreater(kelly.recommended_fraction, 0.0)
        self.assertLessEqual(kelly.recommended_fraction, 0.40)
        self.assertGreater(kelly.recommended_stake_usd, 0.0)
        self.assertGreater(kelly.expected_growth_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
