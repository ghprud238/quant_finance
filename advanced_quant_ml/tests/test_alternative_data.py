import os
import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
"""Unit tests for Module 25: Alternative Data Alpha Model."""

import unittest
import numpy as np
import pandas as pd
from advanced_quant_ml.alternative_data.alpha_model import (
    AlternativeDataAlphaModel,
    ICDecayReport,
    QuantilePerformance,
    AlternativeAlphaBacktestResult,
)


class TestAlternativeDataAlphaModel(unittest.TestCase):
    """Test suite for AlternativeDataAlphaModel."""

    @classmethod
    def setUpClass(cls):
        cls.model = AlternativeDataAlphaModel(
            decay_factor=0.85,
            winsorize_limits=(0.01, 0.01),
            zscore_clip=3.0,
            n_quantiles=5,
            rebalance_freq=5,
            transaction_cost_bps=5.0,
        )
        cls.data = AlternativeDataAlphaModel.generate_synthetic_data(
            n_stocks=20, n_days=300, seed=42
        )

    def test_synthetic_data_generation(self):
        """Test integrity of generated synthetic datasets."""
        prices = self.data["prices"]
        sentiment = self.data["sentiment"]
        web = self.data["web_traffic"]
        supply = self.data["supply_chain"]
        loadings = self.data["risk_loadings"]

        self.assertEqual(prices.shape, (300, 20))
        self.assertEqual(sentiment.shape, (300, 20))
        self.assertEqual(web.shape, (300, 20))
        self.assertEqual(supply.shape, (300, 20))
        self.assertTrue((prices > 0).all().all())
        self.assertIn("Market_Beta", loadings)
        self.assertIn("Momentum_Beta", loadings)
        self.assertIn("Size_Beta", loadings)

    def test_combine_signals(self):
        """Test multi-signal weighted combination."""
        signals = {
            "sentiment": self.data["sentiment"],
            "web": self.data["web_traffic"],
            "supply": self.data["supply_chain"],
        }
        composite = self.model.combine_signals(signals, weights={"sentiment": 0.5, "web": 0.3, "supply": 0.2})
        self.assertEqual(composite.shape, (300, 20))
        self.assertFalse(composite.isna().all().all())

    def test_exponential_decay_smoothing(self):
        """Test exponential smoothing filter."""
        raw_sig = self.data["sentiment"]
        smoothed = self.model.exponential_decay_smoothing(raw_sig, decay_factor=0.8)
        self.assertEqual(smoothed.shape, raw_sig.shape)
        # Smoothed series should have lower standard deviation of daily changes
        raw_diff_std = raw_sig.diff().std().mean()
        smooth_diff_std = smoothed.diff().std().mean()
        self.assertLess(smooth_diff_std, raw_diff_std)

    def test_standardize_cross_section(self):
        """Test cross-sectional winsorization and z-scoring."""
        raw_sig = self.data["sentiment"]
        z_sig = self.model.standardize_cross_section(raw_sig, winsorize_limits=(0.02, 0.02), zscore_clip=2.5)

        # Check mean is approx 0 and max/min are clipped
        for date, row in z_sig.iloc[10:20].iterrows():
            self.assertAlmostEqual(row.mean(), 0.0, delta=0.05)
            self.assertAlmostEqual(row.std(ddof=0), 1.0, delta=0.08)
            self.assertLessEqual(row.max(), 2.5 + 1e-4)
            self.assertGreaterEqual(row.min(), -2.5 - 1e-4)

    def test_factor_neutralization(self):
        """Test OLS orthogonalization against risk factor loadings."""
        raw_sig = self.data["sentiment"]
        loadings = self.data["risk_loadings"]
        neutral_sig = self.model.neutralize_factors(raw_sig, loadings)

        self.assertEqual(neutral_sig.shape, raw_sig.shape)
        # Cross-sectional correlation with Market Beta should be close to 0 on average
        corrs = []
        for date in neutral_sig.index[10:50]:
            s_row = neutral_sig.loc[date].values
            m_row = loadings["Market_Beta"].loc[date].values
            c_mat = np.corrcoef(s_row, m_row)
            if not np.isnan(c_mat[0, 1]):
                corrs.append(c_mat[0, 1])

        avg_corr = np.nanmean(corrs)
        self.assertAlmostEqual(avg_corr, 0.0, delta=0.08)

    def test_ic_decay_computation(self):
        """Test multi-horizon Information Coefficient calculation."""
        sig = self.model.standardize_cross_section(self.data["sentiment"])
        prices = self.data["prices"]
        horizons = [1, 2, 5, 10, 21]

        report = self.model.compute_ic_decay(sig, prices, horizons=horizons)
        self.assertIsInstance(report, ICDecayReport)
        self.assertEqual(report.horizons, horizons)

        for h in horizons:
            self.assertIn(h, report.mean_ic)
            self.assertIn(h, report.mean_rank_ic)
            self.assertIn(h, report.ic_ir)
            self.assertIn(h, report.ic_t_stat)
            self.assertIn(h, report.ic_p_value)

        summary_df = report.summary_table()
        self.assertEqual(len(summary_df), len(horizons))
        self.assertIn("Mean IC", summary_df.columns)
        self.assertIn("Mean Rank IC", summary_df.columns)

    def test_quantile_analysis(self):
        """Test quantile ranking and forward return evaluation."""
        sig = self.model.standardize_cross_section(self.data["sentiment"])
        prices = self.data["prices"]

        q_perf = self.model.quantile_analysis(sig, prices, n_quantiles=5, forward_horizon=1)
        self.assertIsInstance(q_perf, QuantilePerformance)
        self.assertEqual(q_perf.n_quantiles, 5)
        self.assertEqual(len(q_perf.mean_returns), 5)
        self.assertEqual(len(q_perf.cumulative_curves), 5)

        summary_df = q_perf.summary_table()
        self.assertEqual(len(summary_df), 5)
        self.assertIn("Daily Mean Return (%)", summary_df.columns)
        self.assertIn("Sharpe Ratio", summary_df.columns)

    def test_backtest_long_short_strategy(self):
        """Test dollar-neutral Long Q5 / Short Q1 execution."""
        sig = self.model.standardize_cross_section(self.data["sentiment"])
        prices = self.data["prices"]

        res = self.model.backtest_long_short(
            signal_df=sig,
            prices_df=prices,
            n_quantiles=5,
            rebalance_freq=5,
            transaction_cost_bps=5.0,
            borrow_cost_bps=50.0,
            strategy_name="Sentiment Long/Short",
        )
        self.assertIsInstance(res, AlternativeAlphaBacktestResult)
        self.assertEqual(len(res.net_returns), len(prices))
        self.assertEqual(len(res.equity_curve), len(prices))
        self.assertIn("sharpe_net", res.metrics)
        self.assertIn("cagr_net", res.metrics)
        self.assertIn("max_drawdown", res.metrics)
        self.assertIn("annualized_turnover", res.metrics)

        # Check dollar neutrality on rebalance days (sum of weights ~ 0.0)
        rebalance_dates = prices.index[::5]
        for d in rebalance_dates[5:20]:
            w = res.weights.loc[d]
            if (w != 0).any():
                self.assertAlmostEqual(w.sum(), 0.0, places=4)
                # Gross leverage ~ 1.0 (0.5 long + 0.5 short)
                self.assertAlmostEqual(w.abs().sum(), 1.0, places=4)

        summary_df = res.summary_table()
        self.assertIn("Sharpe Ratio (Net)", summary_df.index)
        self.assertIn("Maximum Drawdown", summary_df.index)


if __name__ == "__main__":
    unittest.main()
