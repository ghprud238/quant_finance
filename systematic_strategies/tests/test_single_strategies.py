"""Unit tests for Systematic Strategies (Projects 11, 12, 13)."""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from systematic_strategies.strategies.mean_reversion import (
    MovingAverageMeanReversionStrategy,
    compute_rsi,
    compute_bollinger_bands,
    compute_zscore,
)
from systematic_strategies.strategies.momentum import (
    MomentumTradingStrategy,
    compute_macd,
    compute_tsmom_returns,
    compute_donchian_channels,
)
from systematic_strategies.strategies.pairs_trading import (
    PairsTradingStrategy,
    engle_granger_cointegration_test,
    fit_ornstein_uhlenbeck,
    kalman_filter_hedge_ratio,
    adf_unit_root_test,
)


class TestMeanReversionStrategy(unittest.TestCase):
    """Tests for Project 11: Moving Average Mean Reversion."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=300, freq='B')
        # Mean-reverting synthetic Ornstein-Uhlenbeck series
        p = [100.0]
        for _ in range(299):
            dp = 0.15 * (100.0 - p[-1]) + np.random.normal(0, 1.5)
            p.append(p[-1] + dp)
        self.prices = pd.Series(p, index=dates)

    def test_bollinger_bands(self):
        ma, upper, lower = compute_bollinger_bands(self.prices, window=20, num_std=2.0)
        self.assertEqual(len(ma), len(self.prices))
        # Check band width equals 4 * std
        std = self.prices.rolling(20).std(ddof=1)
        valid_idx = ~np.isnan(ma)
        np.testing.assert_allclose(upper[valid_idx] - ma[valid_idx], 2.0 * std[valid_idx], rtol=1e-5)
        np.testing.assert_allclose(ma[valid_idx] - lower[valid_idx], 2.0 * std[valid_idx], rtol=1e-5)

    def test_rsi_bounds(self):
        rsi = compute_rsi(self.prices, period=14).dropna()
        self.assertTrue((rsi >= 0.0).all())
        self.assertTrue((rsi <= 100.0).all())

    def test_zscore_properties(self):
        z, ma, std = compute_zscore(self.prices, window=20)
        valid_idx = ~np.isnan(z)
        np.testing.assert_allclose(z[valid_idx], ((self.prices - ma) / std)[valid_idx], rtol=1e-5)

    def test_mean_reversion_signals(self):
        strat = MovingAverageMeanReversionStrategy(
            lookback_window=20,
            z_entry=2.0,
            z_exit=0.5,
            allow_short=True,
        )
        res = strat.generate_signals(self.prices)
        df = res.to_dataframe()

        self.assertIn('Position', df.columns)
        self.assertIn('Z_Score', df.columns)
        # Positions must be strictly -1, 0, or 1
        self.assertTrue(set(df['Position'].unique()).issubset({-1.0, 0.0, 1.0}))

        # When entered long, z-score must have been <= -z_entry at entry
        entry_long_indices = df.index[df['Entry_Long']]
        for idx in entry_long_indices:
            self.assertLessEqual(df.loc[idx, 'Z_Score'], -1.95)

    def test_long_only_mode(self):
        strat = MovingAverageMeanReversionStrategy(
            lookback_window=20,
            z_entry=1.5,
            z_exit=0.5,
            allow_short=False,
        )
        res = strat.generate_signals(self.prices)
        positions = res.position.dropna()
        self.assertTrue((positions >= 0.0).all())


class TestMomentumStrategy(unittest.TestCase):
    """Tests for Project 12: Momentum Trading Strategy."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=400, freq='B')
        # Trending series with upward drift
        returns = np.random.normal(0.0008, 0.012, 400)
        p = 100.0 * np.exp(np.cumsum(returns))
        self.prices = pd.Series(p, index=dates)

    def test_ma_crossover_mode(self):
        strat = MomentumTradingStrategy(mode='crossover', fast_window=20, slow_window=50)
        res = strat.generate_signals(self.prices)
        df = res.to_dataframe()

        # When Fast MA > Slow MA, position must be +1
        valid = (~df['Fast_MA'].isna()) & (~df['Slow_MA'].isna())
        crossover_long = valid & (df['Fast_MA'] > df['Slow_MA'])
        self.assertTrue((df.loc[crossover_long, 'Position'] == 1.0).all())

    def test_tsmom_returns(self):
        tsmom = compute_tsmom_returns(self.prices, lookback=100, lag=10).dropna()
        self.assertGreater(len(tsmom), 0)

    def test_macd_calculation(self):
        macd, signal, hist = compute_macd(self.prices, fast_period=12, slow_period=26, signal_period=9)
        np.testing.assert_allclose(hist.dropna(), (macd - signal).dropna(), rtol=1e-5)

    def test_donchian_breakout(self):
        strat = MomentumTradingStrategy(mode='donchian', donchian_window=20)
        highs = self.prices * 1.01
        lows = self.prices * 0.99
        res = strat.generate_signals(self.prices, high=highs, low=lows)
        self.assertIsNotNone(res.donchian_high)
        self.assertIsNotNone(res.donchian_low)

    def test_composite_momentum(self):
        strat = MomentumTradingStrategy(mode='composite', fast_window=15, slow_window=40)
        res = strat.generate_signals(self.prices)
        self.assertTrue(set(res.position.unique()).issubset({-1.0, 0.0, 1.0}))


class TestPairsTradingStrategy(unittest.TestCase):
    """Tests for Project 13: Statistical Arbitrage & Pairs Trading."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2022-01-01', periods=n, freq='B')

        # Generate cointegrated pair: P2 is random walk, P1 = 2.5 * P2 + stationary AR(1) spread
        p2 = 50.0 + np.cumsum(np.random.normal(0.05, 0.8, n))
        stationary_noise = np.zeros(n)
        for t in range(1, n):
            stationary_noise[t] = 0.82 * stationary_noise[t - 1] + np.random.normal(0, 1.2)

        p1 = 2.5 * p2 + stationary_noise + 15.0
        self.p1 = pd.Series(p1, index=dates)
        self.p2 = pd.Series(p2, index=dates)

        # Independent random walks (not cointegrated)
        self.uncorrelated_p1 = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1.0, n)), index=dates)
        self.uncorrelated_p2 = pd.Series(50.0 + np.cumsum(np.random.normal(0, 1.0, n)), index=dates)

    def test_engle_granger_cointegration(self):
        coint_res = engle_granger_cointegration_test(self.p1, self.p2)
        self.assertTrue(coint_res.is_cointegrated)
        self.assertAlmostEqual(coint_res.hedge_ratio_static, 2.5, delta=0.3)
        self.assertLess(coint_res.p_value, 0.05)

    def test_ornstein_uhlenbeck_half_life(self):
        spread = self.p1 - (2.5 * self.p2)
        ou_params = fit_ornstein_uhlenbeck(spread)
        # AR(1) phi = 0.82 -> theta = -ln(0.82) ~ 0.198 -> half-life = ln(2)/0.198 ~ 3.5 days
        self.assertGreater(ou_params.reversion_speed_theta, 0.05)
        self.assertGreater(ou_params.half_life_days, 1.0)
        self.assertLess(ou_params.half_life_days, 30.0)

    def test_kalman_filter_hedge_ratio(self):
        betas, alphas = kalman_filter_hedge_ratio(self.p1, self.p2)
        self.assertEqual(len(betas), len(self.p1))
        # After burn-in, Kalman beta should converge near 2.5
        self.assertAlmostEqual(betas.iloc[-50:].mean(), 2.5, delta=0.4)

    def test_pairs_trading_signals(self):
        strat = PairsTradingStrategy(lookback_window=40, z_entry=1.8, z_exit=0.5)
        res = strat.generate_signals(self.p1, self.p2)
        df = res.to_dataframe()

        self.assertIn('Pair_Position', df.columns)
        self.assertIn('Weight_1', df.columns)
        self.assertIn('Weight_2', df.columns)

        # Dollar neutrality check: when in trade, weights must be non-zero and opposite signs
        in_trade = df['Pair_Position'] != 0.0
        if in_trade.any():
            self.assertTrue((df.loc[in_trade, 'Weight_1'] * df.loc[in_trade, 'Weight_2'] < 0.0).all())


if __name__ == '__main__':
    unittest.main()
