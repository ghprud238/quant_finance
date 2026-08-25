"""Unit tests for Project 37: Green Bond Valuation & Greenium Decomposition Engine."""

import unittest
import numpy as np
import pandas as pd

from climate_quant.data.loader import generate_green_bond_pairs
from climate_quant.green_bonds.greenium import (
    GreenBondValuationEngine,
    GreeniumDecompositionResult,
    GreeniumFactorAttribution,
    NelsonSiegelGreeniumFit,
)


class TestGreenBonds(unittest.TestCase):
    """Validates Greenium calculation, yield spread decomposition, factor regression, and term structure."""

    def setUp(self):
        self.engine = GreenBondValuationEngine(default_yield_curve_slope=0.08)
        self.pairs_df = generate_green_bond_pairs(n_pairs=40, seed=42)

    def test_greenium_calculation(self):
        # 3.20% vanilla vs 3.15% green -> 5.0 bps
        greenium = GreenBondValuationEngine.compute_greenium_bps(3.20, 3.15)
        self.assertAlmostEqual(greenium, 5.0, places=2)

    def test_pair_spread_decomposition(self):
        sample_row = self.pairs_df.iloc[0]
        decomp = self.engine.decompose_pair(sample_row)
        
        self.assertIsInstance(decomp, GreeniumDecompositionResult)
        self.assertGreater(decomp.raw_spread_bps, 0.0)
        self.assertGreater(decomp.pure_greenium_bps, 0.0)
        self.assertIn("Pure Fundamental Greenium (bps)", decomp.summary())

    def test_universe_decomposition(self):
        decomp_df = self.engine.decompose_universe(self.pairs_df)
        self.assertEqual(len(decomp_df), len(self.pairs_df))
        self.assertIn("Raw Spread (bps)", decomp_df.columns)
        self.assertIn("Pure Fundamental Greenium (bps)", decomp_df.columns)

    def test_factor_attribution_regression(self):
        attr = self.engine.attribute_factors(self.pairs_df)
        
        self.assertIsInstance(attr, GreeniumFactorAttribution)
        self.assertGreater(attr.r_squared, 0.20)
        self.assertGreater(attr.n_observations, 20)
        self.assertIn("Credit_Rating_Score", attr.coefficients)
        self.assertIn("ESG_Score", attr.coefficients)
        self.assertIn("Liquidity_Diff_bps", attr.coefficients)
        
        summary_df = attr.summary_dataframe()
        self.assertFalse(summary_df.empty)
        self.assertIn("t_Stat", summary_df.columns)
        self.assertIn("p_Value", summary_df.columns)

    def test_nelson_siegel_term_structure_fit(self):
        maturities = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        # Empirical observed greenium curve (higher for long tenors)
        observed_greeniums = [1.5, 2.1, 2.8, 3.6, 4.4, 5.2, 5.9, 6.4, 6.8]
        
        ns_fit = self.engine.fit_nelson_siegel_term_structure(
            maturities=maturities,
            greeniums_bps=observed_greeniums,
            fixed_lambda=5.0,
        )
        self.assertIsInstance(ns_fit, NelsonSiegelGreeniumFit)
        self.assertGreater(ns_fit.r_squared, 0.90)
        self.assertLess(ns_fit.rmse_bps, 1.0)
        
        # Test predictions
        pred_10y = ns_fit.predict(10.0)
        self.assertAlmostEqual(pred_10y, 5.2, delta=0.6)
        
        preds_array = ns_fit.predict(np.array([5.0, 10.0, 20.0]))
        self.assertEqual(len(preds_array), 3)


if __name__ == "__main__":
    unittest.main()
