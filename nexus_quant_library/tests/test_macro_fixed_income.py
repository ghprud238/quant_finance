"""
Unit tests for Macro Fixed Income, Yield Curves, Central Bank NLP, FX Carry & Carbon Markets.
"""

import unittest
import numpy as np
import pandas as pd
from nexus_quant.macro_fixed_income.yield_curve import (
    NelsonSiegelCalibrator,
    YieldCurveBootstrapper,
    YieldCurvePCA,
)
from nexus_quant.macro_fixed_income.central_bank_nlp import (
    CentralBankStanceIndexer,
)
from nexus_quant.macro_fixed_income.fx_carry import (
    FXCarryParityEngine,
    MalzFXVolatilitySurface,
)
from nexus_quant.macro_fixed_income.carbon_ets import (
    CarbonAllowanceModel,
    GreenBondValuationEngine,
)


class TestMacroFixedIncomeSuite(unittest.TestCase):
    """Test suite for Macro Fixed Income, Yield Curves, NLP & Carbon Pricing."""

    def setUp(self):
        self.maturities = np.array([0.25, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        self.par_yields = np.array([5.20, 5.00, 4.70, 4.40, 4.20, 4.10, 4.15, 4.25, 4.50, 4.60])

    def test_nelson_siegel_calibration_and_forward_rates(self):
        """Test Nelson-Siegel model fitting and forward rate consistency."""
        ns_params = NelsonSiegelCalibrator.fit_nelson_siegel(self.maturities, self.par_yields)

        self.assertGreater(ns_params.lambda_param, 0.0)
        # 10Y spot rate should closely match market
        model_10y = ns_params.spot_rate(10.0)
        self.assertAlmostEqual(model_10y, 4.25, delta=0.25)

        # Instantaneous forward rate should be positive
        fwd_5y = ns_params.forward_rate(5.0)
        self.assertGreater(fwd_5y, 0.0)

        # Discount factor at T=0 is 1.0
        df_0 = ns_params.discount_factor(0.0)
        self.assertAlmostEqual(df_0, 1.0, places=4)

    def test_zero_coupon_bootstrapping(self):
        """Test zero-coupon bootstrapping discount factors and zero rates."""
        boot = YieldCurveBootstrapper.bootstrap_par_yields(self.maturities, self.par_yields)

        # Discount factors should be strictly decreasing with maturity
        self.assertTrue(np.all(np.diff(boot.discount_factors) < 0.0))
        self.assertLess(boot.discount_factors[-1], boot.discount_factors[0])

        # Interpolation tests
        df_3y = boot.get_discount_factor(3.0)
        self.assertGreater(df_3y, 0.0)
        self.assertLessEqual(df_3y, 1.0)

    def test_yield_curve_pca(self):
        """Test PCA decomposition of daily yield shifts."""
        np.random.seed(42)
        n_days = 250
        # Correlated yield changes
        base_dy = np.random.randn(n_days, len(self.maturities)) * 0.05
        # Add strong level shift
        level_shock = np.random.randn(n_days, 1) * 0.10
        sim_dy = base_dy + level_shock
        sim_yields = pd.DataFrame(np.cumsum(sim_dy, axis=0) + self.par_yields, columns=self.maturities)

        pca = YieldCurvePCA(n_components=3).fit(sim_yields)
        summary = pca.summary()

        self.assertGreater(summary["PC1_Level_Explained_Pct"], 50.0)
        self.assertGreater(summary["Total_3PC_Explained_Pct"], 85.0)

    def test_central_bank_nlp_hawk_dove_scoring(self):
        """Test central bank text scoring and Taylor Rule gap analysis."""
        indexer = CentralBankStanceIndexer()

        hawkish_statement = (
            "Inflation remains elevated above our target. The committee is resolute in its commitment "
            "to tightening monetary policy and raising rates to curb wage pressures and persistent demand."
        )
        dovish_statement = (
            "The committee observed economic slack and signs of disinflation. In response to growth headwinds, "
            "we are adopting an accommodative stance with rate cuts and liquidity stimulus."
        )

        hawk_res = indexer.score_statement(hawkish_statement, central_bank="FED", actual_policy_rate=5.25, cpi_inflation=3.5, gdp_growth=2.2)
        dove_res = indexer.score_statement(dovish_statement, central_bank="FED", actual_policy_rate=3.50, cpi_inflation=1.5, gdp_growth=1.0)

        self.assertEqual(hawk_res.stance_category, "HAWKISH")
        self.assertGreater(hawk_res.hawk_dove_score, 0.20)
        self.assertEqual(dove_res.stance_category, "DOVISH")
        self.assertLess(dove_res.hawk_dove_score, -0.20)
        self.assertIsNotNone(hawk_res.taylor_rule_residual)

    def test_fx_cip_and_carry_backtest(self):
        """Test CIP basis evaluation and FX carry trade execution."""
        cip_res = FXCarryParityEngine.evaluate_cip(
            spot=5.00, forward=5.10, domestic_rate=5.0, foreign_rate=10.0, tenor_years=1.0
        )
        self.assertGreater(cip_res.interest_rate_differential, 0.0)

        # Carry strategy backtest on synthetic series
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", "2024-12-31", freq="B")
        fx_spots = pd.DataFrame({
            "USD": 1.0,
            "EUR": np.exp(np.cumsum(np.random.randn(len(dates)) * 0.005)),
            "BRL": 5.0 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.008)),
            "MXN": 18.0 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.007)),
        }, index=dates)

        fx_rates = pd.DataFrame({
            "USD": 5.0,
            "EUR": 3.5,
            "BRL": 12.5,
            "MXN": 11.0,
        }, index=dates)

        bt_res = FXCarryParityEngine.backtest_cross_economy_carry(
            fx_spot_df=fx_spots,
            interest_rates_df=fx_rates,
            funding_currencies=["USD", "EUR"],
            target_currencies=["BRL", "MXN"],
        )
        self.assertGreater(len(bt_res.equity_curve), 0)
        self.assertGreater(bt_res.annualized_volatility, 0.0)

    def test_carbon_pricing_and_green_bond_greenium(self):
        """Test EU ETS fuel-switching parity and greenium decomposition."""
        carbon_model = CarbonAllowanceModel()
        res = carbon_model.evaluate_spreads(
            power_price_mwh=100.0, gas_price_mwh=25.0, coal_price_mwh=12.0, carbon_price_tco2=70.0
        )
        self.assertGreater(res.fuel_switch_parity_price, 0.0)
        self.assertIn(res.merit_order_dominant_fuel, ["GAS_CCGT", "COAL_THERMAL"])

        # Green bond twin pair test
        greenium_res = GreenBondValuationEngine.decompose_twin_pair(
            green_yield_pct=3.85,
            vanilla_yield_pct=3.95,
            green_bid_ask_bps=6.0,
            vanilla_bid_ask_bps=3.0,
            green_duration=7.5,
            vanilla_duration=7.5,
        )
        self.assertAlmostEqual(greenium_res.raw_greenium_bps, 10.0, places=4)
        self.assertGreater(greenium_res.pure_fundamental_greenium_bps, 0.0)


if __name__ == "__main__":
    unittest.main()
