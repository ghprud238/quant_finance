"""Unit tests for Module 33 (VPIN Microstructure Engine)."""

import unittest
import numpy as np
import pandas as pd

from genai_advanced_quant.vpin_microstructure.vpin import (
    VPINEngine,
    VPINResult,
    VolumeBucket,
    ToxicityAlert,
)


class TestVPINMicrostructure(unittest.TestCase):
    """Test suite verifying Easley et al. (2012) VPIN implementation."""

    def setUp(self) -> None:
        self.engine = VPINEngine(n_buckets=20, sigma_window=10)
        self.trades_df = VPINEngine.generate_sample_trade_flow(
            n_trades=3000, initial_price=100.0, inject_flash_crash=True, seed=42
        )

    def test_volume_bucket_conservation(self) -> None:
        """Verifies volume buckets equal target size and sum volume is conserved."""
        bucket_size = 500.0
        buckets_df = self.engine.compute_volume_buckets(self.trades_df, bucket_size=bucket_size)

        self.assertGreater(len(buckets_df), 10)
        # Every bucket must have exactly bucket_size volume
        np.testing.assert_allclose(buckets_df["volume"].values, bucket_size)
        # Total bucketed volume must be less than or equal to total traded volume
        total_bucketed = len(buckets_df) * bucket_size
        self.assertLessEqual(total_bucketed, self.trades_df["volume"].sum())

    def test_bvc_buy_sell_volume_sum(self) -> None:
        """Verifies Bulk Volume Classification buy + sell volume exactly equals bucket size."""
        bucket_size = 500.0
        buckets_df = self.engine.compute_volume_buckets(self.trades_df, bucket_size=bucket_size)
        bvc_df = self.engine.bulk_volume_classification(buckets_df, sigma_window=10)

        total_v = bvc_df["buy_volume"].values + bvc_df["sell_volume"].values
        np.testing.assert_allclose(total_v, bucket_size, rtol=1e-5)
        # Buy and sell volumes must be non-negative
        self.assertTrue(np.all(bvc_df["buy_volume"].values >= 0.0))
        self.assertTrue(np.all(bvc_df["sell_volume"].values >= 0.0))

    def test_vpin_strict_bounds(self) -> None:
        """Verifies VPIN metric is strictly bounded within [0, 1]."""
        result = self.engine.compute_vpin(self.trades_df, bucket_size=400.0, n_buckets=15)

        vpin_vals = result.vpin_series.dropna().values
        self.assertTrue(np.all(vpin_vals >= 0.0), f"VPIN below 0: {vpin_vals.min()}")
        self.assertTrue(np.all(vpin_vals <= 1.0), f"VPIN above 1: {vpin_vals.max()}")
        self.assertGreater(result.mean_vpin, 0.0)
        self.assertLess(result.mean_vpin, 1.0)

    def test_toxic_shock_surge(self) -> None:
        """Verifies toxic selling burst creates significant peak in VPIN."""
        result = self.engine.compute_vpin(self.trades_df, bucket_size=400.0, n_buckets=20)

        # Max VPIN during flash crash window must be substantially higher than min VPIN
        self.assertGreater(result.max_vpin, result.min_vpin * 1.5)
        self.assertGreater(result.max_vpin, 0.40)

    def test_toxicity_alerts_generation(self) -> None:
        """Verifies toxicity alerts are triggered during extreme order imbalance."""
        result = self.engine.compute_vpin(
            self.trades_df, bucket_size=400.0, n_buckets=20, alert_percentile_95=90.0, alert_percentile_99=98.0
        )

        self.assertIsInstance(result.alerts, list)
        self.assertGreater(len(result.alerts), 0)
        severities = {a.severity for a in result.alerts}
        self.assertTrue("CRITICAL" in severities or "WARNING" in severities)

    def test_empty_dataframe_raises(self) -> None:
        """Verifies empty DataFrame input raises ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "price", "volume"])
        with self.assertRaises(ValueError):
            self.engine.compute_volume_buckets(empty_df, bucket_size=100.0)

    def test_summary_table_structure(self) -> None:
        """Verifies summary table returns valid DataFrame with metrics."""
        result = self.engine.compute_vpin(self.trades_df, bucket_size=500.0, n_buckets=10)
        summary = result.summary_table()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn("Metric", summary.columns)
        self.assertIn("Value", summary.columns)


if __name__ == "__main__":
    unittest.main()
