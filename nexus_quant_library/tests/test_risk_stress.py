"""Comprehensive unit tests for Risk, Stress Testing & Climate Risk (Tracks 2, 6, 8)."""
import unittest
import numpy as np
import pandas as pd
from nexus_quant.risk.var_cvar import RiskMetricsEngine
from nexus_quant.risk.stress_testing import PortfolioStressTestingEngine, StressScenario
from nexus_quant.risk.climate_risk import NGFSClimateStressEngine


class TestRiskStress(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.returns = np.random.normal(0.0005, 0.015, 1000)

    def test_var_cvar_metrics(self):
        var_hist = RiskMetricsEngine.historical_var(self.returns, confidence=0.95)
        cvar_hist = RiskMetricsEngine.historical_cvar(self.returns, confidence=0.95)
        self.assertGreater(var_hist, 0.0)
        self.assertGreaterEqual(cvar_hist, var_hist)

        var_param = RiskMetricsEngine.parametric_gaussian_var(self.returns, confidence=0.95, horizon_days=10)
        self.assertGreater(var_param, 0.0)

        var_cf = RiskMetricsEngine.parametric_cornish_fisher_var(self.returns, confidence=0.95)
        self.assertGreater(var_cf, 0.0)

        var_mc, cvar_mc, paths = RiskMetricsEngine.monte_carlo_var_cvar(self.returns, confidence=0.95, n_sims=5000)
        self.assertGreater(var_mc, -1.0)
        self.assertGreaterEqual(cvar_mc, var_mc)
        self.assertEqual(len(paths), 5000)

        n_ex, pi, lr, p_val, passed = RiskMetricsEngine.kupiec_pof_test(self.returns, var_hist, confidence=0.95)
        self.assertTrue(passed)

    def test_macro_stress_testing(self):
        engine = PortfolioStressTestingEngine(portfolio_value=10_000_000.0)
        suite = engine.run_standard_crisis_suite()
        self.assertEqual(len(suite), 5)
        gfc = suite[suite["scenario"] == "2008 Global Financial Crisis"].iloc[0]
        self.assertLess(gfc["pnl_pct"], -0.05)
        self.assertLess(gfc["pnl_dollar"], -500_000.0)

        corr_res = engine.correlation_breakdown_stress(crisis_alpha=0.70)
        self.assertGreater(corr_res["vol_surge_pct"], 50.0)

    def test_climate_risk_and_merton(self):
        c_engine = NGFSClimateStressEngine()
        for scenario in ["Net Zero 2050", "Delayed Transition", "Current Policies (Hot House)"]:
            var_res = c_engine.evaluate_corporate_climate_var(
                company_name="IndustrialCorp",
                market_cap_m=20000.0,
                ebitda_m=4000.0,
                scope1_t=5000000.0,
                scope2_t=1000000.0,
                scope3_t=15000000.0,
                scenario_name=scenario,
            )
            self.assertLess(var_res["climate_var_pct"], 0.0)
            self.assertGreater(var_res["ebitda_hit_pct"], 0.0)

        merton_res = c_engine.merton_stressed_default_probability(
            asset_value=100.0, debt_face_value=60.0, equity_impairment_pct=-0.30
        )
        self.assertGreaterEqual(merton_res["pd_stressed_pct"], merton_res["pd_base_pct"])
        self.assertGreater(merton_res["credit_spread_widening_bps"], 0.0)


if __name__ == "__main__":
    unittest.main()
