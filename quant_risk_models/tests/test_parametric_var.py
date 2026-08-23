import sys
from pathlib import Path
_src = str(Path(__file__).resolve().parent.parent / 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

"""Unit tests for Parametric VaR Model."""

import unittest
import numpy as np
import pandas as pd
from scipy import stats
from quant_risk_models.var.parametric import ParametricVaRModel


class TestParametricVaRModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.0005, 0.015, 1000))
        self.model = ParametricVaRModel(self.returns)

    def test_gaussian_var_exact_formula(self):
        mu = float(np.mean(self.returns))
        sigma = float(np.std(self.returns, ddof=1))
        z95 = stats.norm.ppf(0.95)
        
        expected_loss = -(mu - z95 * sigma)
        calc_loss = self.model.gaussian_var(confidence_level=0.95, as_loss=True)
        self.assertAlmostEqual(calc_loss, expected_loss, places=6)

    def test_horizon_scaling(self):
        var_1d = self.model.gaussian_var(confidence_level=0.95, horizon=1, mean=0.0, std=0.01, as_loss=True)
        var_10d = self.model.gaussian_var(confidence_level=0.95, horizon=10, mean=0.0, std=0.01, as_loss=True)
        self.assertAlmostEqual(var_10d, var_1d * np.sqrt(10), places=6)

    def test_student_t_fat_tails(self):
        # At 99% confidence, t-distribution with df=4 should exhibit higher VaR than Gaussian
        var_norm_99 = self.model.gaussian_var(confidence_level=0.99, as_loss=True)
        var_t_99 = self.model.student_t_var(confidence_level=0.99, df=4.0, as_loss=True)
        self.assertGreater(var_t_99, var_norm_99)

    def test_cornish_fisher_expansion(self):
        # If skew=0 and kurtosis=0, Cornish Fisher should match Gaussian
        cf_var = self.model.cornish_fisher_var(confidence_level=0.95, skew=0.0, kurtosis=0.0, as_loss=True)
        gauss_var = self.model.gaussian_var(confidence_level=0.95, as_loss=True)
        self.assertAlmostEqual(cf_var, gauss_var, places=6)
        
        # Negative skew should increase loss VaR
        cf_neg_skew = self.model.cornish_fisher_var(confidence_level=0.95, skew=-1.0, kurtosis=3.0, as_loss=True)
        self.assertGreater(cf_neg_skew, gauss_var)

    def test_portfolio_analytical_var_and_components(self):
        weights = {'AAPL': 0.6, 'TLT': 0.4}
        cov_matrix = pd.DataFrame([
            [0.0004, -0.00005],
            [-0.00005, 0.0001]
        ], index=['AAPL', 'TLT'], columns=['AAPL', 'TLT'])
        mean_returns = {'AAPL': 0.0008, 'TLT': 0.0001}
        
        port_var, comp_vars = self.model.portfolio_analytical_var(
            weights=weights,
            cov_matrix=cov_matrix,
            mean_returns=mean_returns,
            confidence_level=0.95,
            as_loss=True,
        )
        
        self.assertGreater(port_var, 0.0)
        # Component VaR sum property
        self.assertAlmostEqual(sum(comp_vars.values()), port_var, places=6)


if __name__ == '__main__':
    unittest.main()
