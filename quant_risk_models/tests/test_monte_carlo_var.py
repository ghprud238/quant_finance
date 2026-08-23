import sys
from pathlib import Path
_src = str(Path(__file__).resolve().parent.parent / 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

"""Unit tests for Monte Carlo VaR Engine."""

import unittest
import numpy as np
import pandas as pd
from quant_risk_models.var.monte_carlo import MonteCarloVaREngine


class TestMonteCarloVaREngine(unittest.TestCase):
    def setUp(self):
        self.engine = MonteCarloVaREngine(mean=0.0005, std=0.015)

    def test_simulate_gbm_shape_and_properties(self):
        paths = self.engine.simulate_gbm(
            n_simulations=5000,
            horizon=10,
            n_steps=10,
            initial_value=100.0,
            random_state=42
        )
        self.assertEqual(paths.shape, (5000, 11))
        self.assertTrue(np.all(paths[:, 0] == 100.0))
        self.assertTrue(np.all(paths > 0.0)) # Log-normal non-negativity

    def test_simulate_merton_jump_diffusion(self):
        paths_jump = self.engine.simulate_merton_jump_diffusion(
            n_simulations=5000,
            horizon=21,
            n_steps=21,
            initial_value=100.0,
            jump_intensity=0.1,
            jump_mean=-0.05,
            jump_std=0.05,
            random_state=42
        )
        self.assertEqual(paths_jump.shape, (5000, 22))
        self.assertTrue(np.all(paths_jump > 0.0))

    def test_simulate_correlated_portfolio(self):
        weights = np.array([0.5, 0.5])
        cov = np.array([[0.0004, 0.0001], [0.0001, 0.0002]])
        means = np.array([0.0008, 0.0004])
        
        port_rets, asset_rets = self.engine.simulate_correlated_portfolio(
            weights=weights,
            cov_matrix=cov,
            mean_returns=means,
            n_simulations=10000,
            horizon=1,
            random_state=42
        )
        
        self.assertEqual(len(port_rets), 10000)
        self.assertEqual(asset_rets.shape, (10000, 2))
        
        var_95 = self.engine.compute_var(port_rets, confidence_level=0.95, as_loss=True)
        cvar_95 = self.engine.compute_cvar(port_rets, confidence_level=0.95, as_loss=True)
        
        self.assertGreater(var_95, 0.0)
        self.assertGreaterEqual(cvar_95, var_95)

    def test_fan_chart_generation(self):
        time_steps, percentile_dict = self.engine.generate_fan_chart_data(
            n_simulations=2000,
            horizon=252,
            n_steps=252,
            initial_value=100.0,
            percentiles=[5, 25, 50, 75, 95],
            random_state=42
        )
        self.assertEqual(len(time_steps), 253)
        self.assertIn(5, percentile_dict)
        self.assertIn(95, percentile_dict)
        self.assertTrue(np.all(percentile_dict[95] >= percentile_dict[5]))


if __name__ == '__main__':
    unittest.main()
