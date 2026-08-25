"""Unit tests for Module 39: Geospatial & Satellite GHG Emissions Alternative Data Alpha."""

import unittest
import numpy as np
import pandas as pd

from climate_quant.emissions_satellite.plume_alpha import (
    SatelliteEmissionsAlpha,
    SatelliteObservation,
    EmissionsSurpriseSignal,
    SatelliteAlphaBacktestResult,
)
from climate_quant.data.loader import (
    generate_disclosed_emissions_universe,
    generate_satellite_plume_observations,
    generate_climate_equity_prices,
)


class TestSatelliteEmissionsAlpha(unittest.TestCase):
    """Validates satellite plume aggregation, emissions surprise z-scores, and dollar-neutral backtesting."""

    def setUp(self):
        self.alpha_model = SatelliteEmissionsAlpha(decay_half_life_days=30.0, transaction_cost_bps=5.0)
        self.disclosed_df = generate_disclosed_emissions_universe()
        self.observations = generate_satellite_plume_observations()
        self.prices_df = generate_climate_equity_prices()

    def test_satellite_observation_annualization(self):
        obs = SatelliteObservation(
            observation_id="OBS_001",
            ticker="XOM",
            facility_name="Baytown",
            latitude=29.74,
            longitude=-95.01,
            gas_type="CH4",
            plume_rate_kg_hr=1000.0,
            timestamp=pd.Timestamp("2024-01-05"),
            confidence_score=0.95,
        )
        # 1000 kg/hr * 8760 hr/yr / 1000 kg/t = 8760 tCO2e/yr
        self.assertEqual(obs.annualized_emissions_t, 8760.0)

    def test_facility_plume_aggregation(self):
        agg_df = self.alpha_model.aggregate_facility_plumes(self.observations)

        self.assertFalse(agg_df.empty)
        self.assertIn("ticker", agg_df.columns)
        self.assertIn("satellite_measured_emissions_t", agg_df.columns)
        # Verify tickers are unique
        self.assertEqual(len(agg_df), len(agg_df["ticker"].unique()))
        self.assertTrue((agg_df["satellite_measured_emissions_t"] > 0).all())

    def test_emissions_surprise_calculation(self):
        satellite_df = self.alpha_model.aggregate_facility_plumes(self.observations)
        surprise_df = self.alpha_model.compute_emissions_surprises(self.disclosed_df, satellite_df)

        self.assertFalse(surprise_df.empty)
        self.assertIn("ticker", surprise_df.columns)
        self.assertIn("emissions_gap_t", surprise_df.columns)
        self.assertIn("sector_z_score", surprise_df.columns)
        self.assertIn("alpha_signal", surprise_df.columns)
        self.assertIn("recommendation", surprise_df.columns)

        # Alpha signals must be bounded in [-1.0, +1.0]
        self.assertTrue((surprise_df["alpha_signal"] >= -1.0).all())
        self.assertTrue((surprise_df["alpha_signal"] <= 1.0).all())

        # Clean firms should have positive alpha (Long recommendation)
        clean_recs = surprise_df[surprise_df["alpha_signal"] >= 0.25]["recommendation"].tolist()
        for rec in clean_recs:
            self.assertIn("LONG", rec)

        # High-leak firms should have negative alpha (Short recommendation)
        short_recs = surprise_df[surprise_df["alpha_signal"] <= -0.25]["recommendation"].tolist()
        for rec in short_recs:
            self.assertIn("SHORT", rec)

    def test_backtest_strategy_execution(self):
        # Create multi-period signals dictionary
        signal_dates = self.prices_df.index[::21]
        signals_dict = {}

        for dt in signal_dates:
            satellite_df = self.alpha_model.aggregate_facility_plumes(self.observations, as_of_date=dt)
            if not satellite_df.empty:
                sig_df = self.alpha_model.compute_emissions_surprises(self.disclosed_df, satellite_df, as_of_date=dt)
                signals_dict[dt] = sig_df

        res = self.alpha_model.backtest_strategy(
            prices_df=self.prices_df,
            signals_dict_by_date=signals_dict,
            rebalance_freq_days=21,
            quantile_cutoff=0.30,
        )

        self.assertIsInstance(res, SatelliteAlphaBacktestResult)
        self.assertEqual(len(res.strategy_equity), len(self.prices_df))
        self.assertFalse(res.signals_df.empty)
        self.assertIn("Strategy Sharpe Ratio (Rf=2%)", res.metrics)
        self.assertIn("Strategy Annualized Return (CAGR)", res.metrics)

        # Summary table formatting
        summary_table = res.summary_table()
        self.assertIsInstance(summary_table, pd.DataFrame)
        self.assertIn("Metric", summary_table.columns)
        self.assertIn("Value", summary_table.columns)

    def test_dollar_neutrality(self):
        satellite_df = self.alpha_model.aggregate_facility_plumes(self.observations)
        sig_df = self.alpha_model.compute_emissions_surprises(self.disclosed_df, satellite_df)
        dt0 = self.prices_df.index[0]
        signals_dict = {dt0: sig_df}

        res = self.alpha_model.backtest_strategy(
            prices_df=self.prices_df,
            signals_dict_by_date=signals_dict,
            rebalance_freq_days=21,
            quantile_cutoff=0.30,
        )

        # Check that weights sum to approximately 0.0 (dollar neutrality)
        weights_sum = res.signals_df.iloc[0].sum()
        self.assertAlmostEqual(weights_sum, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
