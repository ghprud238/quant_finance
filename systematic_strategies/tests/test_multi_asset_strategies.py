"""Unit tests for Factor Long/Short and Multi-Asset Trend Strategies (Projects 14 & 15)."""

import unittest
import sys
from pathlib import Path

# Ensure src is on path for discovery
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd


from systematic_strategies.strategies.factor_long_short import (
    FactorLongShortStrategy,
    FactorBacktestResult,
)
from systematic_strategies.strategies.multi_asset_trend import (
    MultiAssetTrendStrategy,
    MultiAssetTrendResult,
)


class TestFactorLongShortStrategy(unittest.TestCase):
    """Tests for Project 14: FactorLongShortStrategy."""

    def setUp(self):
        np.random.seed(42)
        self.n_assets = 25
        self.n_days = 350
        self.tickers = [f"STK_{i:02d}" for i in range(self.n_assets)]
        self.dates = pd.date_range("2023-01-01", periods=self.n_days, freq="B")

        # Generate synthetic asset prices with correlated drift
        drift = 0.0004
        daily_vol = 0.015
        returns = np.random.normal(drift, daily_vol, (self.n_days, self.n_assets))
        self.price_df = pd.DataFrame(
            100.0 * np.exp(np.cumsum(returns, axis=0)),
            index=self.dates,
            columns=self.tickers,
        )

        # Generate 5 factor matrices (Value, Momentum, Quality, Low-Vol, Size)
        self.factor_data = {}
        for factor_name in ["value", "momentum", "quality", "low_vol", "size"]:
            raw_vals = np.random.normal(0, 1, (self.n_days, self.n_assets))
            self.factor_data[factor_name] = pd.DataFrame(
                raw_vals, index=self.dates, columns=self.tickers
            )

        self.strategy = FactorLongShortStrategy(
            n_quantiles=5,
            dollar_neutral=True,
            rebalance_freq=21,
            turnover_smoothing=1.0,
        )

    def test_standardize_cross_section(self):
        """Tests winsorization and z-scoring across universe."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0], index=self.tickers[:5])
        z = self.strategy.standardize_cross_section(s, winsorize=True, clip_val=3.0)
        self.assertAlmostEqual(z.mean(), 0.0, places=5)
        self.assertAlmostEqual(z.std(ddof=1), 1.0, places=5)
        self.assertTrue(np.all(z >= -3.0) and np.all(z <= 3.0))

    def test_composite_scores(self):
        """Tests weighted composite factor score computation."""
        cross_sec = {f: self.factor_data[f].iloc[0] for f in self.factor_data}
        comp_scores = self.strategy.compute_composite_scores(cross_sec)
        self.assertEqual(len(comp_scores), self.n_assets)
        self.assertAlmostEqual(comp_scores.mean(), 0.0, places=4)
        self.assertAlmostEqual(comp_scores.std(ddof=1), 1.0, places=2)

    def test_composite_scores_custom_weights(self):
        """Tests custom factor weighting (e.g. 100% momentum)."""
        cross_sec = {f: self.factor_data[f].iloc[0] for f in self.factor_data}
        custom_weights = {"momentum": 1.0, "value": 0.0, "quality": 0.0, "low_vol": 0.0, "size": 0.0}
        comp_scores = self.strategy.compute_composite_scores(cross_sec, weights=custom_weights)
        raw_mom_z = self.strategy.standardize_cross_section(cross_sec["momentum"], winsorize=False)
        pd.testing.assert_series_equal(comp_scores, raw_mom_z, check_names=False)

    def test_quantile_assignment(self):
        """Tests ranking and quantile allocation into 5 quintiles."""
        scores = pd.Series(np.linspace(-2.0, 2.0, self.n_assets), index=self.tickers)
        quantiles = self.strategy.assign_quantiles(scores, n_quantiles=5)
        self.assertEqual(len(quantiles), self.n_assets)
        self.assertEqual(set(quantiles.unique()), {1, 2, 3, 4, 5})
        self.assertEqual(quantiles.iloc[-1], 5)
        self.assertEqual(quantiles.iloc[0], 1)

    def test_dollar_neutral_weights(self):
        """Tests dollar neutrality (+0.5 long, -0.5 short, 0.0 net, 1.0 gross)."""
        scores = pd.Series(np.random.normal(0, 1, self.n_assets), index=self.tickers)
        weights = self.strategy.construct_portfolio_weights(scores, dollar_neutral=True)

        long_w = weights[weights > 0]
        short_w = weights[weights < 0]

        self.assertAlmostEqual(long_w.sum(), 0.5, places=5)
        self.assertAlmostEqual(short_w.sum(), -0.5, places=5)
        self.assertAlmostEqual(weights.sum(), 0.0, places=5)
        self.assertAlmostEqual(weights.abs().sum(), 1.0, places=5)

    def test_beta_neutral_weights(self):
        """Tests beta-neutral construction eliminates market beta exposure."""
        scores = pd.Series(np.linspace(-2.0, 2.0, self.n_assets), index=self.tickers)
        betas = pd.Series(np.linspace(0.5, 2.0, self.n_assets), index=self.tickers)

        weights = self.strategy.construct_portfolio_weights(
            scores, betas=betas, beta_neutral=True, gross_leverage=1.0
        )

        portfolio_beta = (weights * betas).sum()
        self.assertAlmostEqual(portfolio_beta, 0.0, places=4)
        self.assertAlmostEqual(weights.abs().sum(), 1.0, places=4)

    def test_turnover_smoothing(self):
        """Tests turnover smoothing parameter alpha."""
        strat_smooth = FactorLongShortStrategy(turnover_smoothing=0.5, rebalance_freq=1)
        res = strat_smooth.backtest(prices=self.price_df, factor_data=self.factor_data)
        strat_no_smooth = FactorLongShortStrategy(turnover_smoothing=1.0, rebalance_freq=1)
        res_no_smooth = strat_no_smooth.backtest(prices=self.price_df, factor_data=self.factor_data)

        # Smoothed strategy should have strictly lower turnover
        self.assertLess(res.metrics["Annualized Turnover"], res_no_smooth.metrics["Annualized Turnover"])

    def test_backtest_execution(self):
        """Tests end-to-end backtesting engine and metrics calculation."""
        res = self.strategy.backtest(
            prices=self.price_df,
            factor_data=self.factor_data,
            risk_free_rate=0.02,
        )

        self.assertIsInstance(res, FactorBacktestResult)
        self.assertEqual(len(res.returns), self.n_days)
        self.assertEqual(res.weights.shape, (self.n_days, self.n_assets))
        self.assertIn("Sharpe Ratio", res.metrics)
        self.assertIn("Annualized Return (Net)", res.metrics)
        self.assertIn("Max Drawdown", res.metrics)

        # Summary table formatting
        summary = res.summary_table()
        self.assertTrue(len(summary) >= 8)
        self.assertIn("Metric", summary.columns)
        self.assertIn("Value", summary.columns)


class TestMultiAssetTrendStrategy(unittest.TestCase):
    """Tests for Project 15: MultiAssetTrendStrategy."""

    def setUp(self):
        np.random.seed(42)
        self.dates = pd.date_range("2022-01-01", periods=400, freq="B")
        self.asset_classes = {
            "SPY": "Equities",
            "QQQ": "Equities",
            "TLT": "Bonds",
            "IEF": "Bonds",
            "UUP": "Currencies",
            "FXE": "Currencies",
            "GLD": "Commodities",
            "USO": "Commodities",
        }
        self.tickers = list(self.asset_classes.keys())

        # Generate realistic trend and cyclical price paths
        n_days = len(self.dates)
        price_dict = {}
        for ticker in self.tickers:
            cls = self.asset_classes[ticker]
            if cls == "Equities":
                drift = 0.0004
                vol = 0.012
            elif cls == "Bonds":
                drift = 0.0001
                vol = 0.006
            elif cls == "Currencies":
                drift = 0.0000
                vol = 0.005
            else:  # Commodities
                drift = 0.0002
                vol = 0.015

            r = np.random.normal(drift, vol, n_days)
            price_dict[ticker] = 100.0 * np.exp(np.cumsum(r))

        self.price_df = pd.DataFrame(price_dict, index=self.dates)
        self.strategy = MultiAssetTrendStrategy(
            asset_classes=self.asset_classes,
            lookback_horizons=[21, 63, 126, 252],
            target_asset_vol=0.10,
            target_portfolio_vol=0.10,
            vol_lookback=60,
            use_risk_parity=True,
            rebalance_freq=5,
        )

    def test_trend_conviction_bounds(self):
        """Tests that trend conviction scores are bounded in [-1.0, 1.0]."""
        convictions = self.strategy.compute_trend_conviction(self.price_df)
        self.assertEqual(convictions.shape, self.price_df.shape)
        self.assertTrue(np.all(convictions.dropna() >= -1.0 - 1e-6))
        self.assertTrue(np.all(convictions.dropna() <= 1.0 + 1e-6))

    def test_volatility_targeting_scaling(self):
        """Tests that lower vol assets get larger positions and higher vol gets smaller."""
        convictions = pd.DataFrame(1.0, index=self.dates, columns=self.tickers)
        vols = pd.DataFrame({
            "SPY": 0.20,
            "TLT": 0.05,
            "UUP": 0.05,
            "GLD": 0.15,
        }, index=self.dates)
        strat = MultiAssetTrendStrategy(target_asset_vol=0.10, max_asset_leverage=3.0)
        scaled = strat.compute_volatility_scaled_weights(convictions[vols.columns], vols)

        self.assertAlmostEqual(scaled["TLT"].iloc[0], 2.0, places=3)
        self.assertAlmostEqual(scaled["SPY"].iloc[0], 0.5, places=3)

    def test_risk_parity_solver(self):
        """Tests Equal Risk Contribution solver on 4 asset classes."""
        cov = np.diag([0.04, 0.01, 0.0064, 0.0225])  # 20%, 10%, 8%, 15% vol
        w_erc = self.strategy.solve_risk_parity_weights(cov)

        self.assertAlmostEqual(np.sum(w_erc), 1.0, places=5)
        self.assertTrue(np.all(w_erc > 0))

        # Check risk contributions: w_i * (Sigma w)_i should be equal across all i
        mrc = cov @ w_erc
        rc = w_erc * mrc
        rc_normalized = rc / np.sum(rc)
        for i in range(4):
            self.assertAlmostEqual(rc_normalized[i], 0.25, places=2)

    def test_risk_parity_correlated_matrix(self):
        """Tests ERC solver with non-diagonal correlation structure."""
        corr = np.array([
            [1.0, 0.2, -0.1, 0.3],
            [0.2, 1.0, -0.2, 0.1],
            [-0.1, -0.2, 1.0, 0.0],
            [0.3, 0.1, 0.0, 1.0],
        ])
        vols = np.array([0.18, 0.08, 0.06, 0.15])
        cov = np.diag(vols) @ corr @ np.diag(vols)

        w_erc = self.strategy.solve_risk_parity_weights(cov)
        self.assertAlmostEqual(np.sum(w_erc), 1.0, places=5)

        mrc = cov @ w_erc
        rc = w_erc * mrc
        rc_norm = rc / np.sum(rc)
        for i in range(4):
            self.assertAlmostEqual(rc_norm[i], 0.25, places=2)

    def test_backtest_execution(self):
        """Tests full multi-asset trend following backtest."""
        res = self.strategy.backtest(prices=self.price_df, risk_free_rate=0.02)

        self.assertIsInstance(res, MultiAssetTrendResult)
        self.assertEqual(len(res.returns), len(self.dates))
        self.assertEqual(res.weights.shape, self.price_df.shape)
        self.assertEqual(set(res.asset_class_returns.columns), {"Bonds", "Commodities", "Currencies", "Equities"})
        self.assertIn("Sharpe Ratio", res.metrics)
        self.assertIn("Annualized Return", res.metrics)
        self.assertIn("Average Leverage", res.metrics)

        summary = res.summary_table()
        self.assertTrue(len(summary) >= 7)


if __name__ == "__main__":
    unittest.main()
