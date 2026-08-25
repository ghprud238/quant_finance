"""Unit tests for Project 45: On-Chain Blockchain Telemetry, MVRV, Exchange Flows & Whale Alpha."""

import unittest
import numpy as np
import pandas as pd

from defi_crypto_quant.onchain_alpha.onchain_model import (
    OnChainAlphaEngine,
    OnChainRegime,
    OnChainMetrics,
    OnChainBacktestResult,
)


class TestOnChainAlpha(unittest.TestCase):
    """Validates MVRV, exchange flows, whale index, NVT, regime classifier, and backtest strategy."""

    def setUp(self):
        self.engine = OnChainAlphaEngine()
        self.synthetic_df = self.engine.generate_synthetic_onchain_data(n_days=500, initial_price=10000.0, seed=42)

    def test_mvrv_calculation_and_bounds(self):
        mcap = self.synthetic_df["Market_Cap"]
        rcap = self.synthetic_df["Realized_Cap"]

        mvrv, mvrv_z = self.engine.calculate_mvrv(mcap, rcap)

        self.assertEqual(len(mvrv), len(mcap))
        self.assertEqual(len(mvrv_z), len(mcap))
        self.assertTrue((mvrv > 0.0).all())
        self.assertFalse(mvrv.isna().any())
        self.assertFalse(mvrv_z.isna().any())

        # When Market Cap > Realized Cap, MVRV must be > 1.0
        sample_idx = (mcap > rcap)
        self.assertTrue((mvrv[sample_idx] > 1.0).all())

    def test_exchange_flow_imbalance(self):
        inflows = pd.Series([1000.0, 5000.0, 200.0, 0.0])
        outflows = pd.Series([500.0, 1000.0, 2000.0, 0.0])

        net_flow, efi = self.engine.calculate_exchange_flow_imbalance(inflows, outflows, smoothing_window=1)

        # Net flow = in - out
        self.assertAlmostEqual(net_flow.iloc[0], 500.0)
        self.assertAlmostEqual(net_flow.iloc[1], 4000.0)
        self.assertAlmostEqual(net_flow.iloc[2], -1800.0)

        # EFI in [-1.0, 1.0]
        self.assertTrue((efi >= -1.0).all())
        self.assertTrue((efi <= 1.0).all())
        self.assertGreater(efi.iloc[0], 0.0)  # Inflows > Outflows -> positive
        self.assertLess(efi.iloc[2], 0.0)     # Outflows > Inflows -> negative

    def test_whale_accumulation_index(self):
        # Steeper upward trend in whale balances
        whale_bal = pd.Series(np.linspace(100000, 150000, 200), index=pd.date_range("2023-01-01", periods=200))
        whale_idx = self.engine.calculate_whale_accumulation_index(whale_bal, lookback_days=10, z_window=50)

        self.assertEqual(len(whale_idx), len(whale_bal))
        self.assertFalse(whale_idx.isna().any())
        self.assertTrue((whale_idx >= -4.0).all())
        self.assertTrue((whale_idx <= 4.0).all())

    def test_nvt_metrics(self):
        mcap = self.synthetic_df["Market_Cap"]
        tx_vol = self.synthetic_df["Tx_Volume_USD"]

        nvt, nvt_signal = self.engine.calculate_nvt_metrics(mcap, tx_vol)

        self.assertEqual(len(nvt), len(mcap))
        self.assertEqual(len(nvt_signal), len(mcap))
        self.assertTrue((nvt > 0.0).all())
        self.assertTrue((nvt_signal > 0.0).all())

    def test_address_velocity(self):
        addresses = self.synthetic_df["Active_Addresses"]
        velocity = self.engine.calculate_address_velocity(addresses)

        self.assertEqual(len(velocity), len(addresses))
        self.assertFalse(velocity.isna().any())
        self.assertTrue((velocity >= -1.0).all())
        self.assertTrue((velocity <= 1.0).all())

    def test_regime_classification(self):
        dates = pd.date_range("2023-01-01", periods=4)
        mvrv = pd.Series([0.85, 1.80, 3.50, 1.30], index=dates)
        mvrv_z = pd.Series([0.05, 1.50, 4.20, 0.60], index=dates)
        efi = pd.Series([-0.20, 0.00, 0.30, 0.25], index=dates)
        whale = pd.Series([1.5, 0.2, -1.0, -0.8], index=dates)
        velocity = pd.Series([0.10, 0.05, -0.02, -0.20], index=dates)

        regimes = self.engine.classify_regimes(mvrv, mvrv_z, efi, whale, velocity)

        self.assertEqual(regimes.iloc[0], OnChainRegime.ACCUMULATION_BOTTOM)
        self.assertEqual(regimes.iloc[1], OnChainRegime.BULL_EXPANSION)
        self.assertEqual(regimes.iloc[2], OnChainRegime.OVERHEATED_EUPHORIA)
        self.assertEqual(regimes.iloc[3], OnChainRegime.CAPITULATION_BEAR)

    def test_composite_signal_bounds_and_direction(self):
        mvrv, mvrv_z = self.engine.calculate_mvrv(self.synthetic_df["Market_Cap"], self.synthetic_df["Realized_Cap"])
        net_flow, efi = self.engine.calculate_exchange_flow_imbalance(self.synthetic_df["Exchange_Inflows"], self.synthetic_df["Exchange_Outflows"])
        whale_acc = self.engine.calculate_whale_accumulation_index(self.synthetic_df["Whale_Balance"])
        nvt_ratio, nvt_signal = self.engine.calculate_nvt_metrics(self.synthetic_df["Market_Cap"], self.synthetic_df["Tx_Volume_USD"])
        addr_velocity = self.engine.calculate_address_velocity(self.synthetic_df["Active_Addresses"])

        sig = self.engine.compute_composite_signal(mvrv, mvrv_z, efi, whale_acc, nvt_signal, addr_velocity)

        self.assertEqual(len(sig), len(self.synthetic_df))
        self.assertFalse(sig.isna().any())
        self.assertTrue((sig >= -1.0).all())
        self.assertTrue((sig <= 1.0).all())

    def test_backtest_strategy_execution(self):
        res = self.engine.backtest_strategy(self.synthetic_df, initial_capital=100000.0, max_leverage=1.5)

        self.assertIsInstance(res, OnChainBacktestResult)
        self.assertIn("Strategy CAGR", res.metrics)
        self.assertIn("Sharpe Ratio (Rf=3%)", res.metrics)
        self.assertIn("Maximum Drawdown", res.metrics)
        self.assertIn("Information Coefficient (IC)", res.metrics)

        # Ensure summary table is properly populated
        summary = res.summary_table()
        self.assertFalse(summary.empty)
        self.assertIn("Metric", summary.columns)
        self.assertIn("Value", summary.columns)

        # Positions are bounded by max leverage
        self.assertTrue((res.positions <= 1.5).all())
        self.assertTrue((res.positions >= -1.0).all())

        # Cumulative strategy must be positive
        self.assertTrue((res.cumulative_strategy > 0.0).all())


if __name__ == "__main__":
    unittest.main()
