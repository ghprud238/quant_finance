"""Comprehensive unit tests for Strategies & Portfolio Optimization (Tracks 3, 2, 7)."""
import unittest
import numpy as np
import pandas as pd
from nexus_quant.strategies.systematic_engine import SystematicBacktestEngine
from nexus_quant.strategies.momentum_meanrev import MeanReversionStrategy, MomentumStrategy
from nexus_quant.strategies.pairs_stat_arb import PairsStatArbEngine
from nexus_quant.portfolio_optimization.markowitz import MarkowitzOptimizer
from nexus_quant.portfolio_optimization.wasserstein_dro import WassersteinDROOptimizer
from nexus_quant.portfolio_optimization.risk_parity import RiskParityOptimizer


class TestStrategiesOptimization(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        self.prices_y = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.012, len(dates)))), index=dates)
        self.prices_x = pd.Series(50.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.010, len(dates)))), index=dates)

    def test_backtest_engine_and_mean_reversion(self):
        signals = MeanReversionStrategy.generate_signals(self.prices_y, lookback=20, z_entry=1.5, z_exit=0.3)
        engine = SystematicBacktestEngine(fee_bps=5.0, half_spread_bps=2.5)
        bt_res = engine.run(self.prices_y, signals["Position"])
        self.assertIn("Sharpe Ratio", bt_res.metrics)
        self.assertIn("CAGR", bt_res.metrics)
        summary = bt_res.summary_table()
        self.assertGreater(len(summary), 5)

        # Test Momentum Strategy
        mom_sig = MomentumStrategy.generate_signals(self.prices_y, fast_window=10, slow_window=50, target_vol=0.10)
        bt_mom = engine.run(self.prices_y, mom_sig["Position"])
        self.assertIn("Sharpe Ratio", bt_mom.metrics)

    def test_kalman_pairs_trading(self):
        kf_engine = PairsStatArbEngine()
        pos_df = kf_engine.generate_trading_positions(self.prices_y, self.prices_x)
        self.assertEqual(len(pos_df), len(self.prices_y))
        self.assertIn("Weight_Y", pos_df.columns)
        self.assertIn("Weight_X", pos_df.columns)
        self.assertIn("Beta", pos_df.columns)

    def test_markowitz_optimization(self):
        mu = np.array([0.15, 0.12, 0.08, 0.05])
        cov = np.array([
            [0.0625, 0.0250, 0.0150, 0.0050],
            [0.0250, 0.0400, 0.0120, 0.0040],
            [0.0150, 0.0120, 0.0225, 0.0030],
            [0.0050, 0.0040, 0.0030, 0.0100],
        ])
        m_opt = MarkowitzOptimizer(mu, cov)
        min_p = m_opt.min_volatility()
        tan_p = m_opt.max_sharpe_ratio()

        self.assertAlmostEqual(float(np.sum(min_p.weights)), 1.0, places=4)
        self.assertAlmostEqual(float(np.sum(tan_p.weights)), 1.0, places=4)
        self.assertLessEqual(min_p.volatility, tan_p.volatility)
        self.assertGreaterEqual(tan_p.sharpe_ratio, min_p.sharpe_ratio)

        frontier = m_opt.efficient_frontier(n_points=20)
        self.assertGreaterEqual(len(frontier), 15)

    def test_wasserstein_dro_and_risk_parity(self):
        mu = np.array([0.15, 0.12, 0.08, 0.05])
        cov = np.array([
            [0.0625, 0.0250, 0.0150, 0.0050],
            [0.0250, 0.0400, 0.0120, 0.0040],
            [0.0150, 0.0120, 0.0225, 0.0030],
            [0.0050, 0.0040, 0.0030, 0.0100],
        ])
        dro_opt = WassersteinDROOptimizer(mu, cov, risk_aversion=1.5)
        dro_res = dro_opt.optimize(epsilon=0.02)
        self.assertAlmostEqual(float(np.sum(dro_res["weights"])), 1.0, places=4)
        self.assertGreater(dro_res["effective_n_assets"], 1.0)

        sweep = dro_opt.ambiguity_sweep(epsilons=np.linspace(0, 0.05, 5))
        self.assertEqual(len(sweep), 5)

        # Risk Parity
        rp_opt = RiskParityOptimizer(cov)
        w_rp, rc_pct = rp_opt.optimize()
        self.assertAlmostEqual(float(np.sum(w_rp)), 1.0, places=4)
        self.assertTrue(np.allclose(rc_pct, 0.25, atol=0.05))


if __name__ == "__main__":
    unittest.main()
