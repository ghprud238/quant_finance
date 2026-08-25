"""Unit tests for Module 28: Portfolio Risk & Stress Testing Engine."""

import unittest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from interview_quant.stress_testing.engine import (
    AssetPosition,
    StressScenario,
    ScenarioResult,
    PortfolioStressTestingEngine,
    get_standard_historical_scenarios,
    create_sample_multi_asset_portfolio,
)


class TestPortfolioStressTestingEngine(unittest.TestCase):
    def setUp(self):
        self.portfolio_value = 10_000_000.0
        self.engine = PortfolioStressTestingEngine(portfolio_value=self.portfolio_value)

    def test_sample_portfolio_weights_normalization(self):
        """Test that portfolio weights sum to 1.0."""
        total_w = sum(a.weight for a in self.engine.assets)
        self.assertAlmostEqual(total_w, 1.0, places=6)

    def test_standard_historical_scenarios_benchmarks(self):
        """Verify exact benchmark scenario P&L values matching targets:
        - 2008 GFC: -12.4%
        - 2020 COVID: -8.7%
        - 2022 Rate Shock: -6.1%
        - Market Crash (-30%): -15.3%
        - Stagflation: -9.2%
        """
        results = self.engine.run_all_historical_scenarios()
        
        self.assertAlmostEqual(results["2008 Global Financial Crisis"].portfolio_pnl_pct * 100, -12.4, places=1)
        self.assertAlmostEqual(results["2020 COVID Crash"].portfolio_pnl_pct * 100, -8.7, places=1)
        self.assertAlmostEqual(results["2022 Interest Rate Shock"].portfolio_pnl_pct * 100, -6.1, places=1)
        self.assertAlmostEqual(results["Market Crash (-30%)"].portfolio_pnl_pct * 100, -15.3, places=1)
        self.assertAlmostEqual(results["Custom / Geopolitical Stagflation"].portfolio_pnl_pct * 100, -9.2, places=1)

    def test_scenario_loss_series(self):
        """Test scenario loss series helper output."""
        loss_series = self.engine.scenario_loss_series()
        self.assertIsInstance(loss_series, pd.Series)
        self.assertEqual(len(loss_series), 5)
        self.assertIn("2008 Global Financial Crisis", loss_series.index)
        self.assertLess(loss_series["2008 Global Financial Crisis"], 0.0)

    def test_summary_table_structure(self):
        """Test summary table dataframe formatting."""
        df = self.engine.summary_table()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertIn("Scenario", df.columns)
        self.assertIn("Portfolio P&L (%)", df.columns)
        self.assertIn("Portfolio P&L ($)", df.columns)
        self.assertIn("Largest Loss Contributor", df.columns)
        self.assertIn("Stressed 95% Daily VaR", df.columns)

    def test_asset_loss_contribution_sum(self):
        """Test that asset loss percentage contributions sum to 100% during losses."""
        scenarios = get_standard_historical_scenarios()
        for sc in scenarios:
            res = self.engine.evaluate_scenario(sc)
            if res.portfolio_pnl_pct < 0:
                total_contrib = sum(res.asset_loss_contribution_pct.values())
                self.assertAlmostEqual(total_contrib, 100.0, places=2)

    def test_fixed_income_duration_convexity(self):
        """Test fixed income price response to yield shocks: dP/P ~= -D * dy + 0.5 * C * (dy)^2."""
        bond = AssetPosition(
            name="10Y Bond",
            asset_class="fixed_income",
            weight=1.0,
            duration=8.0,
            convexity=0.8,
        )
        custom_engine = PortfolioStressTestingEngine(assets=[bond], portfolio_value=1_000_000.0)
        
        # Upward yield shock: +100 bps (+0.01)
        up_scen = StressScenario(name="Yield +100bps", description="test", yield_shock_bps=100.0)
        res_up = custom_engine.evaluate_scenario(up_scen)
        expected_ret = -8.0 * 0.01 + 0.5 * 0.8 * (0.01 ** 2)
        self.assertAlmostEqual(res_up.portfolio_pnl_pct, expected_ret, places=6)

    def test_derivatives_greeks_response(self):
        """Test derivative position response: dV ~= Delta * dS + 0.5 * Gamma * dS^2 + Vega * dVol."""
        option = AssetPosition(
            name="Index Call Option",
            asset_class="derivative",
            weight=1.0,
            delta=0.50,
            gamma=0.04,
            vega=0.15,
        )
        custom_engine = PortfolioStressTestingEngine(assets=[option], portfolio_value=1_000_000.0)
        
        scen = StressScenario(name="Crash+Vol", description="test", equity_shock=-0.20, vol_shock_pct=0.50)
        res = custom_engine.evaluate_scenario(scen)
        expected_ret = 0.50 * (-0.20) + 0.5 * 0.04 * ((-0.20) ** 2) + 0.15 * 0.50
        self.assertAlmostEqual(res.portfolio_pnl_pct, expected_ret, places=6)

    def test_factor_sensitivity_grid(self):
        """Test 2D factor sensitivity matrix across equity vs yield shocks."""
        grid = self.engine.run_factor_sensitivity_grid(
            equity_shocks=[-0.20, 0.0, 0.20],
            yield_shocks_bps=[-100.0, 0.0, 100.0]
        )
        self.assertIsInstance(grid, pd.DataFrame)
        self.assertEqual(grid.shape, (3, 3))
        # Negative equity and positive yield should produce negative return
        self.assertLess(grid.loc["-20%", "+100 bps"], 0.0)

    def test_correlation_breakdown_stress(self):
        """Test that correlation breakdown surges portfolio volatility and VaR."""
        corr_res = self.engine.correlation_breakdown_stress(crisis_alpha=0.70)
        
        self.assertGreater(corr_res["crisis_annual_vol"], corr_res["base_annual_vol"])
        self.assertGreater(corr_res["vol_surge_pct"], 0.0)
        self.assertGreater(corr_res["crisis_95_var_daily"], corr_res["base_95_var_daily"])
        self.assertEqual(corr_res["crisis_correlation_matrix"].shape, (6, 6))

    def test_stressed_var_and_cvar_coherence(self):
        """Test that CVaR > VaR under stressed conditions (coherence)."""
        scenarios = get_standard_historical_scenarios()
        for sc in scenarios:
            res = self.engine.evaluate_scenario(sc)
            self.assertGreater(res.stressed_cvar_95, res.stressed_var_95)
            self.assertGreater(res.stressed_var_99, res.stressed_var_95)


if __name__ == "__main__":
    unittest.main()
