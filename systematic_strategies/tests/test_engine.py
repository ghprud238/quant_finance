import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
"""Comprehensive unit tests for the systematic trading backtest engine and data layer."""

import unittest
import numpy as np
import pandas as pd

from systematic_strategies.data import (
    generate_date_range,
    generate_equities_data,
    generate_pairs_data,
    generate_macro_data,
    generate_cross_sectional_universe,
    load_equities,
    load_pairs,
    load_macro,
    load_cross_sectional,
)
from systematic_strategies.engine import (
    TransactionCostModel,
    PositionSizer,
    BacktestEngine,
    BacktestResult,
    WalkForwardValidator,
    WalkForwardReport,
)


class TestDataLayer(unittest.TestCase):
    """Tests synthetic data generation and data loaders."""

    def test_date_range(self):
        dates = generate_date_range("2020-01-01", "2020-12-31")
        self.assertGreaterEqual(len(dates), 250)
        self.assertEqual(dates[0].year, 2020)

    def test_equities_generation(self):
        eq_df = generate_equities_data()
        self.assertEqual(list(eq_df.columns), ["SPY", "QQQ", "AAPL", "MSFT"])
        self.assertTrue((eq_df > 0.0).all().all())
        self.assertFalse(eq_df.isna().any().any())

    def test_pairs_generation(self):
        pairs_df = generate_pairs_data()
        self.assertEqual(list(pairs_df.columns), ["KO", "PEP", "XOM", "CVX"])
        self.assertTrue((pairs_df > 0.0).all().all())

    def test_macro_generation(self):
        macro_df = generate_macro_data()
        self.assertEqual(list(macro_df.columns), ["SPY", "TLT", "UUP", "GLD", "USO"])
        self.assertTrue((macro_df > 0.0).all().all())

    def test_cross_sectional_generation(self):
        prices_df, factors_df = generate_cross_sectional_universe(n_stocks=10)
        self.assertEqual(prices_df.shape[1], 10)
        self.assertIn("Momentum", factors_df.columns)
        self.assertIn("Value", factors_df.columns)
        self.assertTrue((prices_df > 0.0).all().all())

    def test_loaders(self):
        eq = load_equities()
        self.assertIsInstance(eq, pd.DataFrame)
        self.assertGreater(len(eq), 100)

        pairs = load_pairs()
        self.assertIsInstance(pairs, pd.DataFrame)

        macro = load_macro()
        self.assertIsInstance(macro, pd.DataFrame)

        cs_prices, cs_factors = load_cross_sectional()
        self.assertIsInstance(cs_prices, pd.DataFrame)
        self.assertIsInstance(cs_factors, pd.DataFrame)


class TestTransactionCostModel(unittest.TestCase):
    """Tests execution friction and slippage modeling."""

    def setUp(self):
        self.cost_model = TransactionCostModel(fee_bps=5.0, half_spread_bps=2.5, market_impact_gamma=0.01)

    def test_linear_cost_calculation(self):
        # 7.5 bps total linear rate = 0.00075
        delta_w = 1.0  # full rebalance
        cost = self.cost_model.compute_trade_cost(delta_w)
        expected = 0.00075 + 0.5 * 0.01 * (1.0 ** 2)
        self.assertAlmostEqual(cost, expected, places=6)

    def test_turnover_series(self):
        weights = pd.Series([0.0, 1.0, 1.0, 0.5, -0.5])
        turnover = self.cost_model.compute_turnover(weights)
        self.assertEqual(turnover.iloc[0], 0.0)
        self.assertEqual(turnover.iloc[1], 1.0)
        self.assertEqual(turnover.iloc[2], 0.0)
        self.assertEqual(turnover.iloc[3], 0.5)
        self.assertEqual(turnover.iloc[4], 1.0)

    def test_multi_asset_turnover(self):
        w_df = pd.DataFrame({
            "A": [0.5, 0.6, 0.4],
            "B": [0.5, 0.4, 0.6],
        })
        turnover = self.cost_model.compute_turnover(w_df)
        self.assertAlmostEqual(turnover.iloc[1], 0.2, places=6)
        self.assertAlmostEqual(turnover.iloc[2], 0.4, places=6)

    def test_apply_costs(self):
        gross_returns = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2020-01-01", periods=3))
        weights = pd.Series([1.0, 1.0, 0.0], index=gross_returns.index)
        net_ret, costs = self.cost_model.apply_costs(gross_returns, weights)
        self.assertTrue((net_ret <= gross_returns).all())
        self.assertGreater(costs.sum(), 0.0)

    def test_cost_breakdown(self):
        weights = pd.Series([0.5, 1.0, 0.0, 0.5] * 50)
        breakdown = self.cost_model.cost_breakdown(weights)
        self.assertIn("annualized_turnover", breakdown)
        self.assertIn("total_annualized_drag_bps", breakdown)
        self.assertGreater(breakdown["total_annualized_drag_bps"], 0.0)


class TestPositionSizer(unittest.TestCase):
    """Tests position sizing and dynamic leverage formulas."""

    def test_fixed_fractional(self):
        signals = pd.Series([1.0, -1.0, 0.5, -0.5])
        sized = PositionSizer.fixed_fractional(signals, fraction=1.5, max_leverage=1.0)
        self.assertEqual(sized.iloc[0], 1.0)
        self.assertEqual(sized.iloc[1], -1.0)
        self.assertEqual(sized.iloc[2], 0.75)
        self.assertEqual(sized.iloc[3], -0.75)

    def test_volatility_targeting(self):
        dates = pd.date_range("2020-01-01", periods=100)
        # Create return series with 20% annualized volatility
        rng = np.random.default_rng(42)
        daily_vol = 0.20 / np.sqrt(252)
        returns = pd.Series(rng.standard_normal(100) * daily_vol, index=dates)
        signals = pd.Series(1.0, index=dates)

        # Target 10% vol -> scalar should be ~0.5
        weights = PositionSizer.volatility_targeting(signals, returns, target_vol=0.10, lookback_window=21, max_leverage=2.0)
        self.assertIsInstance(weights, pd.Series)
        self.assertTrue((weights >= 0.0).all())
        self.assertTrue((weights <= 2.0).all())

    def test_kelly_criterion(self):
        # 60% win rate, 1.5 win/loss ratio
        # f* = (0.6 * 1.5 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.5 / 1.5 = 0.333
        # Half Kelly = 0.1666
        f_half = PositionSizer.kelly_criterion(win_rate=0.60, win_loss_ratio=1.5, fraction=0.5)
        self.assertAlmostEqual(f_half, 0.5 / 3.0, places=4)

        # Negative edge should return 0.0
        f_neg = PositionSizer.kelly_criterion(win_rate=0.40, win_loss_ratio=1.0)
        self.assertEqual(f_neg, 0.0)

    def test_inverse_volatility_weights(self):
        dates = pd.date_range("2020-01-01", periods=100)
        rng = np.random.default_rng(42)
        ret_a = rng.standard_normal(100) * 0.01  # Low vol
        ret_b = rng.standard_normal(100) * 0.03  # High vol
        df_ret = pd.DataFrame({"A": ret_a, "B": ret_b}, index=dates)

        w_df = PositionSizer.inverse_volatility_weights(df_ret, lookback_window=21)
        # Sum of weights should equal 1.0
        np.testing.assert_allclose(w_df.sum(axis=1).iloc[30:], 1.0, rtol=1e-5)
        # Asset A should have higher weight than Asset B
        self.assertGreater(w_df["A"].iloc[50:].mean(), w_df["B"].iloc[50:].mean())

    def test_equal_risk_contribution_weights(self):
        # 2 asset cov matrix with unequal vols
        cov = np.array([[0.04, 0.01], [0.01, 0.16]])
        w_erc = PositionSizer.equal_risk_contribution_weights(cov)
        self.assertAlmostEqual(np.sum(w_erc), 1.0, places=5)
        # Asset 1 (lower vol) should receive larger weight
        self.assertGreater(w_erc[0], w_erc[1])
        # Risk contributions should be equal: w_i * (cov @ w)_i
        rc = w_erc * (cov @ w_erc)
        self.assertAlmostEqual(rc[0], rc[1], places=5)


class TestBacktestEngine(unittest.TestCase):
    """Tests backtesting accounting and performance attribution."""

    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", "2021-12-31")
        self.engine = BacktestEngine(
            cost_model=TransactionCostModel(fee_bps=5.0, half_spread_bps=2.5),
            risk_free_rate=0.02,
        )

    def test_timing_shift_prevents_lookahead(self):
        # Buy on day t should not earn return on day t, but on day t+1
        prices = pd.Series([100.0, 105.0, 110.0], index=self.dates[:3])
        weights = pd.Series([1.0, 1.0, 1.0], index=self.dates[:3])
        res = self.engine.run(prices, weights)
        # On day 0, executed weight is 0.0 (shifted from before), so gross return is 0
        self.assertEqual(res.gross_returns.iloc[0], 0.0)
        # On day 1, return is (105-100)/100 = 5%
        self.assertAlmostEqual(res.gross_returns.iloc[1], 0.05, places=6)

    def test_single_asset_backtest_metrics(self):
        # Synthetic upward drifting price series
        rng = np.random.default_rng(42)
        n = len(self.dates)
        daily_ret = 0.0008 + rng.standard_normal(n) * 0.01
        prices = 100.0 * np.exp(np.cumsum(daily_ret))
        price_series = pd.Series(prices, index=self.dates)
        weights = pd.Series(1.0, index=self.dates)

        res = self.engine.run(price_series, weights, strategy_name="Long Buy & Hold")

        self.assertIsInstance(res, BacktestResult)
        self.assertGreater(res.metrics["cagr"], 0.0)
        self.assertGreater(res.metrics["sharpe_ratio"], 0.0)
        self.assertLess(res.metrics["max_drawdown"], 0.0)
        self.assertIn("var_95", res.metrics)
        self.assertIn("cvar_95", res.metrics)

        # Verify summary table formatting
        summary = res.summary_table()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn("Metric", summary.columns)

    def test_multi_asset_backtest(self):
        rng = np.random.default_rng(42)
        n = len(self.dates)
        p1 = 100.0 * np.exp(np.cumsum(0.0005 + rng.standard_normal(n) * 0.01))
        p2 = 50.0 * np.exp(np.cumsum(0.0003 + rng.standard_normal(n) * 0.015))
        prices_df = pd.DataFrame({"A": p1, "B": p2}, index=self.dates)

        weights_df = pd.DataFrame({"A": 0.6, "B": 0.4}, index=self.dates)
        res = self.engine.run(prices_df, weights_df, strategy_name="60/40 Portfolio")

        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(len(res.equity_curve), n)
        self.assertGreater(res.metrics["total_return"], -0.99)

    def test_benchmark_relative_metrics(self):
        rng = np.random.default_rng(42)
        n = len(self.dates)
        bench_ret = pd.Series(0.0004 + rng.standard_normal(n) * 0.01, index=self.dates)
        strat_prices = pd.Series(100.0 * np.exp(np.cumsum(0.0006 + rng.standard_normal(n) * 0.012)), index=self.dates)
        weights = pd.Series(1.0, index=self.dates)

        res = self.engine.run(strat_prices, weights, benchmark_returns=bench_ret)
        self.assertIn("beta", res.metrics)
        self.assertIn("alpha", res.metrics)
        self.assertIn("tracking_error", res.metrics)
        self.assertIn("information_ratio", res.metrics)


class TestWalkForwardValidation(unittest.TestCase):
    """Tests walk-forward validation and overfitting audit."""

    def setUp(self):
        dates = pd.bdate_range("2018-01-01", "2024-12-31")
        rng = np.random.default_rng(42)
        ret = 0.0005 + rng.standard_normal(len(dates)) * 0.01
        self.prices = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=dates)

    def test_simple_train_test_split(self):
        weights = pd.Series(1.0, index=self.prices.index)
        validator = WalkForwardValidator()
        is_res, oos_res, degradation = validator.simple_train_test_split(self.prices, weights, train_ratio=0.70)

        self.assertEqual(len(is_res.dates) + len(oos_res.dates), len(self.prices))
        self.assertIsInstance(degradation, float)
        self.assertGreaterEqual(degradation, 0.0)

    def test_walk_forward_evaluation(self):
        # Moving average crossover strategy function
        def sma_crossover_strategy(data_slice):
            fast_sma = data_slice.rolling(20, min_periods=5).mean()
            slow_sma = data_slice.rolling(50, min_periods=10).mean()
            signal = np.where(fast_sma > slow_sma, 1.0, 0.0)
            return pd.Series(signal, index=data_slice.index)

        validator = WalkForwardValidator(train_window_days=504, test_window_days=126, step_days=126)
        report = validator.walk_forward_evaluate(sma_crossover_strategy, self.prices)

        self.assertIsInstance(report, WalkForwardReport)
        self.assertGreater(len(report.folds), 0)
        self.assertIn("total_return", report.stitched_oos_metrics)
        self.assertIsInstance(report.summary_table(), pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
