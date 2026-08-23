import sys
from pathlib import Path
_src = str(Path(__file__).resolve().parent.parent / 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

"""Unit tests for Historical VaR Calculator."""

import unittest
import numpy as np
import pandas as pd
from quant_risk_models.var.historical import HistoricalVaRCalculator


class TestHistoricalVaRCalculator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 1000 simulated normal returns
        self.returns = pd.Series(np.random.normal(0.0005, 0.015, 1000))
        self.calc = HistoricalVaRCalculator(self.returns)

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            HistoricalVaRCalculator([])
        with self.assertRaises(ValueError):
            df_multi = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            HistoricalVaRCalculator(df_multi)

    def test_empirical_var_bounds(self):
        var_95 = self.calc.compute_var(confidence_level=0.95, as_loss=True)
        var_99 = self.calc.compute_var(confidence_level=0.99, as_loss=True)
        
        self.assertGreater(var_95, 0.0)
        self.assertGreater(var_99, var_95)
        
        # Test as_loss=False (negative return quantile)
        ret_95 = self.calc.compute_var(confidence_level=0.95, as_loss=False)
        self.assertAlmostEqual(var_95, -ret_95, places=6)

    def test_multi_confidence_levels(self):
        res = self.calc.compute_var([0.90, 0.95, 0.99], as_loss=True)
        self.assertEqual(len(res), 3)
        self.assertIn(0.95, res)
        self.assertGreater(res[0.99], res[0.95])
        self.assertGreater(res[0.95], res[0.90])

    def test_cvar_greater_than_var(self):
        var_95 = self.calc.compute_var(0.95, as_loss=True)
        cvar_95 = self.calc.compute_cvar(0.95, as_loss=True)
        self.assertGreaterEqual(cvar_95, var_95)

    def test_rolling_var(self):
        rolling_95 = self.calc.rolling_var(window=252, confidence_level=0.95, as_loss=True)
        self.assertEqual(len(rolling_95), len(self.returns))
        self.assertTrue(pd.isna(rolling_95.iloc[0]))
        self.assertFalse(pd.isna(rolling_95.iloc[251]))
        
        with self.assertRaises(ValueError):
            self.calc.rolling_var(window=2000)

    def test_age_weighted_var(self):
        aw_var = self.calc.age_weighted_var(confidence_level=0.95, decay_factor=0.98, as_loss=True)
        self.assertIsInstance(aw_var, float)
        self.assertGreater(aw_var, 0.0)
        
        aw_cvar = self.calc.age_weighted_cvar(confidence_level=0.95, decay_factor=0.98, as_loss=True)
        self.assertGreaterEqual(aw_cvar, aw_var)

    def test_bootstrap_confidence_intervals(self):
        point, lower, upper = self.calc.bootstrap_confidence_interval(
            confidence_level=0.95, ci_level=0.95, n_bootstraps=300, random_state=42, as_loss=True
        )
        self.assertLessEqual(lower, point)
        self.assertGreaterEqual(upper, point)
        self.assertLess(lower, upper)


if __name__ == '__main__':
    unittest.main()
