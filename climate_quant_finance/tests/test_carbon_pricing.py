"""Unit tests for Project 36: Carbon Allowance Pricing & ETS Fuel-Switching Dynamics."""

import unittest
import numpy as np
import pandas as pd

from climate_quant.data.loader import generate_carbon_market_data
from climate_quant.carbon_pricing.eua_model import (
    CarbonAllowanceModel,
    CleanSpreadsResult,
    FuelSwitchingParity,
    CarbonSimulationResult,
)


class TestCarbonPricing(unittest.TestCase):
    """Validates Clean Spark/Dark spreads, Fuel-Switching parity, Jump-Diffusion, and Futures curves."""

    def setUp(self):
        self.model = CarbonAllowanceModel(
            efficiency_gas=0.50,
            efficiency_coal=0.38,
            emission_factor_gas=0.37,
            emission_factor_coal=0.95,
        )
        self.market_data = generate_carbon_market_data(seed=42)

    def test_clean_spark_spread_formula(self):
        # CSS = 100 - (30 / 0.50) - (0.37 * 80) = 100 - 60 - 29.6 = 10.4
        css = self.model.compute_clean_spark_spread(
            power_price=100.0,
            gas_price=30.0,
            carbon_price=80.0,
        )
        self.assertAlmostEqual(float(css), 10.4, places=2)

    def test_clean_dark_spread_formula(self):
        # CDS = 100 - (20 / 0.38) - (0.95 * 80) = 100 - 52.6316 - 76.0 = -28.6316
        cds = self.model.compute_clean_dark_spread(
            power_price=100.0,
            coal_price=20.0,
            carbon_price=80.0,
        )
        self.assertAlmostEqual(float(cds), -28.6316, places=3)

    def test_fuel_switching_parity_equilibrium(self):
        gas_p = 35.0
        coal_p = 15.0
        p_switch = float(self.model.compute_fuel_switching_price(gas_price=gas_p, coal_price=coal_p))
        
        # Verify that at P_carbon = P_switch, CSS strictly equals CDS
        power_p = 120.0
        css = self.model.compute_clean_spark_spread(power_p, gas_p, p_switch)
        cds = self.model.compute_clean_dark_spread(power_p, coal_p, p_switch)
        self.assertAlmostEqual(float(css), float(cds), places=4)

    def test_spread_snapshot_and_series(self):
        snap = self.model.evaluate_spread_snapshot(
            power_price=110.0,
            gas_price=32.0,
            coal_price=18.0,
            carbon_price=75.0,
        )
        self.assertIsInstance(snap, CleanSpreadsResult)
        self.assertIn(snap.dominant_fuel, ["Gas (CCGT)", "Coal (Thermal)"])

        series_df = self.model.compute_spreads_series(self.market_data)
        self.assertIn("CSS", series_df.columns)
        self.assertIn("CDS", series_df.columns)
        self.assertIn("P_Switch_Parity", series_df.columns)
        self.assertIn("Switching_Incentive", series_df.columns)
        self.assertEqual(len(series_df), len(self.market_data))

    def test_jump_diffusion_simulation(self):
        sim_res = self.model.simulate_jump_diffusion(
            s0=80.0,
            horizon_years=1.0,
            n_steps=100,
            n_paths=500,
            kappa=1.5,
            theta=85.0,
            sigma=0.30,
            jump_intensity=4.0,
            seed=42,
        )
        self.assertIsInstance(sim_res, CarbonSimulationResult)
        self.assertEqual(sim_res.paths.shape, (101, 500))
        self.assertGreater(sim_res.mean_terminal_price, 20.0)
        self.assertLess(sim_res.mean_terminal_price, 250.0)
        self.assertTrue((sim_res.upper_95_path >= sim_res.expected_path - 1e-8).all())
        self.assertTrue((sim_res.expected_path >= sim_res.lower_95_path - 1e-8).all())

    def test_jump_diffusion_calibration(self):
        eua_series = self.market_data["EUA_Carbon_Spot"]
        calib_dict = self.model.calibrate_jump_diffusion(eua_series)
        
        self.assertIn("kappa", calib_dict)
        self.assertIn("theta", calib_dict)
        self.assertIn("sigma", calib_dict)
        self.assertIn("jump_intensity_annual", calib_dict)
        self.assertGreater(calib_dict["kappa"], 0.0)
        self.assertGreater(calib_dict["sigma"], 0.0)
        self.assertGreater(calib_dict["theta"], 10.0)

    def test_futures_curve_cost_of_carry(self):
        curve_df = self.model.construct_futures_curve(
            spot_price=85.0,
            tenors_years=[0.25, 0.5, 1.0, 2.0],
            r=0.03,
            convenience_yield=0.01,
            storage_cost=0.002,
        )
        self.assertEqual(len(curve_df), 4)
        self.assertIn("Futures_Price", curve_df.columns)
        # Positive net carry (r + u - y = 0.03 + 0.002 - 0.01 = 0.022 > 0) implies contango
        self.assertTrue((curve_df["Futures_Price"].diff().dropna() > 0).all())


if __name__ == "__main__":
    unittest.main()
