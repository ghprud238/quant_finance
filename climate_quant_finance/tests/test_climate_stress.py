"""Unit tests for Module 38: NGFS Climate Scenario Stress Testing & Physical/Transition Risk Engine."""

import unittest
import numpy as np
import pandas as pd

from climate_quant.climate_stress.ngfs_engine import (
    NGFSClimateStressEngine,
    NGFSScenarioType,
    CompanyClimateProfile,
    ClimateStressResult,
    PortfolioClimateStressReport,
)
from climate_quant.data.loader import generate_corporate_climate_universe


class TestClimateStressEngine(unittest.TestCase):
    """Validates NGFS scenario modeling, transition/physical risk calculations, Climate VaR, and Merton PD migration."""

    def setUp(self):
        self.engine = NGFSClimateStressEngine()
        self.portfolio = generate_corporate_climate_universe()
        self.oil_major = self.portfolio[0]  # XOM
        self.tech_major = self.portfolio[8]  # MSFT

    def test_ngfs_scenario_initialization(self):
        self.assertIn(NGFSScenarioType.NET_ZERO_2050, self.engine.scenarios)
        self.assertIn(NGFSScenarioType.DELAYED_TRANSITION, self.engine.scenarios)
        self.assertIn(NGFSScenarioType.CURRENT_POLICIES, self.engine.scenarios)

        net_zero = self.engine.scenarios[NGFSScenarioType.NET_ZERO_2050]
        self.assertGreater(net_zero.carbon_price_trajectory[2030], 100.0)
        self.assertGreater(net_zero.carbon_price_trajectory[2050], 300.0)

    def test_transition_risk_cost_calculation(self):
        net_zero = self.engine.scenarios[NGFSScenarioType.NET_ZERO_2050]
        res = self.engine.evaluate_transition_risk(self.oil_major, net_zero, year=2030)

        # Carbon price = $140/t
        self.assertEqual(res["carbon_price"], 140.0)
        self.assertGreater(res["gross_carbon_cost_m"], 0.0)
        # Net cost should be strictly less than gross cost due to pass-through rate (40%)
        self.assertLess(res["net_carbon_cost_m"], res["gross_carbon_cost_m"])
        expected_unpassed = 1.0 - self.oil_major.carbon_pass_through_rate
        self.assertAlmostEqual(res["net_carbon_cost_m"] / res["gross_carbon_cost_m"], expected_unpassed, places=4)

    def test_physical_risk_damage_calculation(self):
        hot_house = self.engine.scenarios[NGFSScenarioType.CURRENT_POLICIES]
        net_zero = self.engine.scenarios[NGFSScenarioType.NET_ZERO_2050]

        phys_hot = self.engine.evaluate_physical_risk(self.oil_major, hot_house, year=2050)
        phys_net_zero = self.engine.evaluate_physical_risk(self.oil_major, net_zero, year=2050)

        # Physical damage in Hot House should be substantially higher than in Net Zero 2050
        self.assertGreater(phys_hot["annual_damage_m"], phys_net_zero["annual_damage_m"])
        self.assertGreater(phys_hot["hazard_multiplier"], phys_net_zero["hazard_multiplier"])

    def test_merton_credit_migration(self):
        # 20% asset impairment
        credit_res = self.engine.compute_merton_credit_migration(self.oil_major, asset_impairment_pct=0.20)

        self.assertGreater(credit_res["stressed_pd"], credit_res["baseline_pd"])
        self.assertGreater(credit_res["stressed_spread_bps"], credit_res["baseline_spread_bps"])
        self.assertGreater(credit_res["spread_delta_bps"], 0.0)

    def test_single_company_stress_test(self):
        res = self.engine.stress_test_company(self.oil_major, NGFSScenarioType.NET_ZERO_2050, year=2030)

        self.assertIsInstance(res, ClimateStressResult)
        self.assertEqual(res.ticker, "XOM")
        self.assertLess(res.stressed_ebitda, res.baseline_ebitda)
        self.assertLess(res.stressed_equity_val, res.baseline_equity_val)
        self.assertLess(res.climate_equity_var_pct, 0.0)  # Negative equity impact
        self.assertGreater(res.stressed_pd, res.baseline_pd)
        self.assertGreater(res.credit_spread_widening_bps, 0.0)

    def test_sector_heterogeneity(self):
        # High-emitting Energy firm (XOM) should face much higher Climate VaR under Net Zero 2050 than low-emitting Tech firm (MSFT)
        res_oil = self.engine.stress_test_company(self.oil_major, NGFSScenarioType.NET_ZERO_2050, year=2030)
        res_tech = self.engine.stress_test_company(self.tech_major, NGFSScenarioType.NET_ZERO_2050, year=2030)

        self.assertLess(res_oil.climate_equity_var_pct, res_tech.climate_equity_var_pct)
        self.assertGreater(abs(res_oil.ebitda_impairment_pct), abs(res_tech.ebitda_impairment_pct))

    def test_portfolio_stress_test_and_summary(self):
        rep = self.engine.run_portfolio_stress_test(self.portfolio, NGFSScenarioType.DELAYED_TRANSITION, year=2035)

        self.assertIsInstance(rep, PortfolioClimateStressReport)
        self.assertEqual(rep.n_companies, len(self.portfolio))
        self.assertLess(rep.portfolio_climate_var_pct, 0.0)
        self.assertGreater(rep.total_annual_transition_cost, 0.0)

        summary_df = rep.summary_table()
        self.assertFalse(summary_df.empty)
        self.assertEqual(len(summary_df), len(self.portfolio))
        self.assertIn("Ticker", summary_df.columns)
        self.assertIn("Climate_VaR_%", summary_df.columns)

    def test_multi_scenario_comparison(self):
        comp_df = self.engine.multi_scenario_comparison(self.portfolio, year=2030)

        self.assertIsInstance(comp_df, pd.DataFrame)
        self.assertEqual(len(comp_df), 3)
        self.assertIn("NGFS_Scenario", comp_df.columns)
        self.assertIn("Portfolio_Climate_VaR_%", comp_df.columns)


if __name__ == "__main__":
    unittest.main()
