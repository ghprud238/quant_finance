"""
Comprehensive Unit Tests for Validation Rigor, Deflated Sharpe & Master Orchestrator (Tracks 6, 11).
"""

import unittest
import numpy as np
import pandas as pd

from nexus_quant.validation_rigor import (
    DeflatedSharpeRatioCalculator,
    QuantResearchValidationPipeline,
)
from nexus_quant.orchestrator import NexusMasterOrchestrator


class TestValidationOrchestrator(unittest.TestCase):

    def test_psr_and_non_normality(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        
        # High Sharpe over 1 year
        psr = dsr_calc.compute_psr(observed_sharpe=2.5, benchmark_sharpe=0.0, sample_length=252, skewness=-0.5, kurtosis=4.5)
        self.assertGreater(psr.psr_value, 0.95)
        self.assertTrue(psr.is_significant_95)

        # Low Sharpe
        psr_low = dsr_calc.compute_psr(observed_sharpe=0.4, benchmark_sharpe=0.0, sample_length=252)
        self.assertLess(psr_low.psr_value, 0.90)
        self.assertFalse(psr_low.is_significant_95)

    def test_expected_max_sharpe_multiple_testing(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        
        # 1 trial vs 1,000 trials
        em_1, _ = dsr_calc.expected_max_sharpe(num_trials=1, var_sharpe_trials=0.25)
        em_1000, n_eff = dsr_calc.expected_max_sharpe(num_trials=1000, var_sharpe_trials=0.25)
        
        self.assertEqual(em_1, 0.0)
        self.assertGreater(em_1000, 1.2)
        self.assertEqual(n_eff, 1000.0)

    def test_deflated_sharpe_ratio_audit(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        
        # Genuine Strategy: Sharpe 2.8 under 1,000 trials across 5 years (1260 days)
        dsr_pass = dsr_calc.compute_dsr(best_sharpe_ratio=2.8, num_trials=1000, sample_length=1260, skewness=-0.5, kurtosis=4.0)
        self.assertGreater(dsr_pass.deflated_sharpe_ratio, 0.95)
        self.assertTrue(dsr_pass.is_significant_95)
        self.assertIn("GENUINE ALPHA", dsr_pass.summary())

        # Overfitted Strategy: Sharpe 1.2 under 10,000 trials across 1 year (252 days)
        dsr_fail = dsr_calc.compute_dsr(best_sharpe_ratio=1.2, num_trials=10000, sample_length=252)
        self.assertLess(dsr_fail.deflated_sharpe_ratio, 0.95)
        self.assertFalse(dsr_fail.is_significant_95)
        self.assertIn("OVERFITTED", dsr_fail.summary())

    def test_min_track_record_length(self):
        dsr_calc = DeflatedSharpeRatioCalculator()
        min_trl = dsr_calc.min_track_record_length(observed_sharpe=2.0, benchmark_sharpe=0.0)
        self.assertGreater(min_trl, 0.0)
        self.assertLess(min_trl, 5.0)

    def test_validation_pipeline_and_readiness_checklist(self):
        np.random.seed(42)
        pipe = QuantResearchValidationPipeline(risk_free_rate=0.02, tx_cost_bps=5.0)
        p = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0015, 0.010, 500))))
        w = pd.Series(1.0, index=p.index)

        tear_sheet = pipe.evaluate_strategy(p, w)
        self.assertGreater(tear_sheet.cagr, 0.0)
        self.assertGreater(tear_sheet.sharpe_ratio, 0.5)
        self.assertGreater(tear_sheet.estimated_capacity_usd, 1_000_000.0)
        self.assertIn("CAGR (Annual Return)", tear_sheet.metrics_table["Metric"].values)

        readiness = pipe.score_deployment_readiness(tear_sheet, dsr_p_value=0.98)
        self.assertGreaterEqual(readiness.readiness_score, 80)
        self.assertTrue(readiness.is_deployable)
        self.assertEqual(len(readiness.failure_reasons), 0)

    def test_nexus_master_orchestrator_integration(self):
        orchestrator = NexusMasterOrchestrator(initial_capital_usd=10_000_000.0)
        res = orchestrator.build_cross_asset_portfolio(n_days=1000, seed=42)
        
        self.assertGreater(res.sharpe_ratio, 1.5)
        self.assertGreater(res.cagr, 0.08)
        self.assertLess(res.max_drawdown, 0.0)
        self.assertEqual(len(res.asset_allocations), 10)
        self.assertAlmostEqual(sum(res.asset_allocations.values()), 1.0, places=4)
        self.assertEqual(len(res.stress_test_results), 4)
        self.assertGreater(res.deflated_sharpe_ratio, 0.95)
        self.assertIn("Nexus 11-Track Master Portfolio", res.summary())


if __name__ == "__main__":
    unittest.main()
