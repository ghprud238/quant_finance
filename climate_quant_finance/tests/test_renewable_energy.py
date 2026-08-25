"""Unit tests for Project 40: Renewable Energy PPA Valuation & Weather Derivatives."""

import unittest
import numpy as np
import pandas as pd

from climate_quant.renewable_energy.ppa_weather import (
    RenewableAssetConfig,
    HourlyGenerationProfile,
    PPAType,
    PPAContract,
    PPAValuationResult,
    RenewablePPAValuator,
    TemperatureModelParams,
    WeatherContractType,
    WeatherDerivativeContract,
    WeatherPricingResult,
    WeatherDerivativePricer,
)


class TestRenewablePPAValuation(unittest.TestCase):
    """Validates solar/wind yield simulation, capture prices, cannibalization, and PPA cash flows."""

    def setUp(self):
        self.solar_config = RenewableAssetConfig(asset_type="solar", capacity_mw=100.0, latitude_deg=35.0)
        self.wind_config = RenewableAssetConfig(asset_type="wind", capacity_mw=100.0, weibull_k=2.1, weibull_c=8.5)
        self.solar_valuator = RenewablePPAValuator(self.solar_config)
        self.wind_valuator = RenewablePPAValuator(self.wind_config)

    def test_solar_generation_profile(self):
        profile = self.solar_valuator.simulate_hourly_solar_profile(year=2026, seed=42)
        self.assertEqual(len(profile.timestamps), 8760)
        self.assertEqual(len(profile.generation_mwh), 8760)
        # Non-negativity
        self.assertTrue((profile.generation_mwh >= 0.0).all())
        # Peak generation cannot exceed installed capacity
        self.assertLessEqual(profile.peak_generation_mw, self.solar_config.capacity_mw + 1e-4)
        # Solar capacity factor typical range 15% to 30%
        self.assertGreater(profile.capacity_factor, 0.15)
        self.assertLess(profile.capacity_factor, 0.35)
        # Night hours generation should be zero
        night_mask = profile.timestamps.hour.isin([0, 1, 2, 3, 22, 23])
        self.assertAlmostEqual(float(np.sum(profile.generation_mwh[night_mask])), 0.0, places=3)

    def test_wind_generation_profile(self):
        profile = self.wind_valuator.simulate_hourly_wind_profile(year=2026, seed=42)
        self.assertEqual(len(profile.timestamps), 8760)
        self.assertEqual(len(profile.generation_mwh), 8760)
        self.assertTrue((profile.generation_mwh >= 0.0).all())
        self.assertLessEqual(profile.peak_generation_mw, self.wind_config.capacity_mw + 1e-4)
        # Wind capacity factor typical range 25% to 55%
        self.assertGreater(profile.capacity_factor, 0.25)
        self.assertLess(profile.capacity_factor, 0.55)

    def test_capture_price_and_cannibalization(self):
        solar_gen = self.solar_valuator.simulate_hourly_solar_profile(year=2026, seed=42)
        spot_prices = self.solar_valuator.simulate_electricity_spot_prices(
            solar_gen, base_price_mwh=60.0, cannibalization_beta=0.30, seed=42
        )

        base_p, cap_p, cap_rate, cann_disc = self.solar_valuator.calculate_capture_metrics(
            solar_gen.generation_mwh, spot_prices.values
        )

        # Basic properties
        self.assertGreater(base_p, 30.0)
        self.assertGreater(cap_p, 20.0)
        # Due to cannibalization beta > 0, capture price for solar should be discounted vs baseload
        self.assertLess(cap_p, base_p)
        self.assertLess(cap_rate, 1.0)
        self.assertGreater(cann_disc, 0.0)

    def test_pap_ppa_valuation(self):
        solar_gen = self.solar_valuator.simulate_hourly_solar_profile(year=2026, seed=42)
        spot_prices = self.solar_valuator.simulate_electricity_spot_prices(solar_gen, base_price_mwh=60.0, seed=42)
        contract = PPAContract(contract_type=PPAType.PAY_AS_PRODUCED, strike_price=55.0, tenor_years=10)

        res = self.solar_valuator.value_ppa_contract(solar_gen, spot_prices, contract)
        self.assertEqual(res.contract_type, "PAY_AS_PRODUCED")
        self.assertGreater(res.npv_revenue_usd, 0.0)
        self.assertGreater(res.npv_net_cashflow_usd, 0.0)
        self.assertGreater(res.npv_per_kw_installed, 0.0)
        self.assertEqual(res.shaping_cost_total_usd, 0.0)  # No shaping cost in PAP

    def test_baseload_ppa_valuation_and_shaping(self):
        wind_gen = self.wind_valuator.simulate_hourly_wind_profile(year=2026, seed=42)
        spot_prices = self.wind_valuator.simulate_electricity_spot_prices(wind_gen, base_price_mwh=60.0, seed=42)
        contract = PPAContract(contract_type=PPAType.BASELOAD, strike_price=55.0, tenor_years=10)

        res = self.wind_valuator.value_ppa_contract(wind_gen, spot_prices, contract)
        self.assertEqual(res.contract_type, "BASELOAD")
        self.assertGreater(res.npv_net_cashflow_usd, 0.0)
        # Baseload PPA involves shaping volume risk
        self.assertGreater(res.annual_generation_mwh, 0.0)

    def test_collar_and_merchant_structures(self):
        solar_gen = self.solar_valuator.simulate_hourly_solar_profile(year=2026, seed=42)
        spot_prices = self.solar_valuator.simulate_electricity_spot_prices(solar_gen, base_price_mwh=60.0, seed=42)

        collar_contract = PPAContract(contract_type=PPAType.COLLAR_PPA, floor_price=45.0, cap_price=75.0, tenor_years=5)
        merchant_contract = PPAContract(contract_type=PPAType.MERCHANT, tenor_years=5)

        collar_res = self.solar_valuator.value_ppa_contract(solar_gen, spot_prices, collar_contract)
        merchant_res = self.solar_valuator.value_ppa_contract(solar_gen, spot_prices, merchant_contract)

        self.assertGreater(collar_res.npv_net_cashflow_usd, 0.0)
        self.assertGreater(merchant_res.npv_net_cashflow_usd, 0.0)


class TestWeatherDerivativesPricing(unittest.TestCase):
    """Validates Degree Day calculations, OU temperature SDE simulation, and weather option pricing."""

    def setUp(self):
        self.params = TemperatureModelParams(
            base_temp_A=14.5,
            seasonal_amp_C=11.5,
            phase_phi=108.0,
            kappa=0.28,
            sigma0=2.80,
            sigma1=1.10,
            market_price_of_risk_lambda=0.06,
        )
        self.pricer = WeatherDerivativePricer(self.params)

    def test_degree_day_math(self):
        # Known single-day temperatures
        temps = np.array([10.0, 15.0, 18.0, 22.0, 25.0])
        # Base = 18°C
        # HDD: max(18 - T, 0) -> [8, 3, 0, 0, 0] -> sum = 11
        # CDD: max(T - 18, 0) -> [0, 0, 0, 4, 7] -> sum = 11
        hdd_val = self.pricer.compute_degree_days(temps, base_temp=18.0, index_type="HDD")
        cdd_val = self.pricer.compute_degree_days(temps, base_temp=18.0, index_type="CDD")

        self.assertAlmostEqual(float(hdd_val), 11.0)
        self.assertAlmostEqual(float(cdd_val), 11.0)

    def test_temperature_simulation_shapes(self):
        dates, paths = self.pricer.simulate_temperature_paths(
            start_date="2026-11-01", end_date="2027-03-31", n_paths=500, seed=42
        )
        # Winter season: 151 days
        self.assertEqual(len(dates), 151)
        self.assertEqual(paths.shape, (500, 151))
        # Mean winter temperature should be colder than baseline annual mean (14.5°C)
        self.assertLess(float(np.mean(paths)), self.params.base_temp_A)

    def test_hdd_swap_pricing(self):
        # Winter HDD swap
        contract = WeatherDerivativeContract(
            contract_type=WeatherContractType.HDD_SWAP,
            strike=850.0,
            tick_size_usd=10000.0,
            start_date="2026-11-01",
            end_date="2027-03-31",
            discount_rate=0.045,
        )
        res = self.pricer.price_contract(contract, n_mc_sims=2000, seed=42)

        self.assertEqual(res.contract_type, "HDD_SWAP")
        self.assertGreater(res.expected_index_value, 500.0)
        self.assertGreater(res.index_std_dev, 20.0)
        self.assertGreater(res.fair_swap_strike, 500.0)
        self.assertEqual(res.probability_of_exercise, 1.0)

    def test_hdd_call_option_pricing(self):
        contract = WeatherDerivativeContract(
            contract_type=WeatherContractType.HDD_CALL,
            strike=900.0,
            tick_size_usd=10000.0,
            max_payout_cap_usd=2500000.0,
            start_date="2026-11-01",
            end_date="2027-03-31",
        )
        res = self.pricer.price_contract(contract, n_mc_sims=2000, seed=42)

        self.assertEqual(res.contract_type, "HDD_CALL")
        self.assertGreaterEqual(res.fair_premium_usd, 0.0)
        self.assertLessEqual(res.fair_premium_usd, contract.max_payout_cap_usd)
        self.assertGreaterEqual(res.probability_of_exercise, 0.0)
        self.assertLessEqual(res.probability_of_exercise, 1.0)

    def test_hdd_put_option_and_collar(self):
        put_contract = WeatherDerivativeContract(
            contract_type=WeatherContractType.HDD_PUT,
            strike=800.0,
            tick_size_usd=10000.0,
            max_payout_cap_usd=2000000.0,
            start_date="2026-11-01",
            end_date="2027-03-31",
        )
        collar_contract = WeatherDerivativeContract(
            contract_type=WeatherContractType.WEATHER_COLLAR,
            strike=850.0,
            floor_strike=800.0,
            cap_strike=900.0,
            tick_size_usd=10000.0,
            max_payout_cap_usd=2000000.0,
            start_date="2026-11-01",
            end_date="2027-03-31",
        )

        put_res = self.pricer.price_contract(put_contract, n_mc_sims=2000, seed=42)
        collar_res = self.pricer.price_contract(collar_contract, n_mc_sims=2000, seed=42)

        self.assertEqual(put_res.contract_type, "HDD_PUT")
        self.assertGreaterEqual(put_res.fair_premium_usd, 0.0)
        self.assertEqual(collar_res.contract_type, "WEATHER_COLLAR")


if __name__ == '__main__':
    unittest.main()
