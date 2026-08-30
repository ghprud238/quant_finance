"""Unit tests for Information Diffusion & Multivariate Hawkes Process Engine (Module 52)."""

import unittest
import numpy as np
from quant_mechanics.data.loader import generate_macro_news_and_trades
from quant_mechanics.hawkes_diffusion.hawkes_model import (
    MultivariateHawkesEngine,
    HawkesFitResult,
)


class TestHawkesDiffusion(unittest.TestCase):

    def setUp(self):
        self.engine = MultivariateHawkesEngine(dimension=2)

    def test_hawkes_simulation_and_fit(self):
        """Tests Ogata's thinning simulation and MLE parameter estimation."""
        # Define known parameters
        mu = np.array([0.05, 0.10])
        alpha = np.array([[0.10, 0.05], [0.30, 0.15]])
        beta = np.array([[1.00, 0.80], [0.90, 1.20]])
        horizon = 1000.0  # seconds

        # Simulate 2-variate process
        events = self.engine.simulate(mu, alpha, beta, horizon=horizon, seed=42)
        self.assertEqual(len(events), 2)
        self.assertGreater(len(events[0]), 20)
        self.assertGreater(len(events[1]), 50)

        # Fit model on simulated events
        fit_res = self.engine.fit(events, horizon=horizon, max_iter=100)
        self.assertIsInstance(fit_res, HawkesFitResult)
        self.assertEqual(fit_res.dimension, 2)
        self.assertTrue(np.all(fit_res.mu > 0))
        self.assertTrue(np.all(fit_res.alpha >= 0))
        self.assertTrue(np.all(fit_res.beta > 0))
        self.assertLess(fit_res.spectral_radius, 1.0)
        self.assertGreater(fit_res.endogeneity_ratio, 0.0)

    def test_impulse_response_and_half_life(self):
        """Tests analytical impulse response kernel and half-life calculations."""
        # Manually create fit result
        mu = np.array([0.01, 0.02])
        alpha = np.array([[0.05, 0.01], [0.40, 0.10]])
        beta = np.array([[1.00, 1.00], [0.50, 1.00]])
        branching = alpha / beta
        
        self.engine.fitted_params = HawkesFitResult(
            dimension=2,
            mu=mu,
            alpha=alpha,
            beta=beta,
            branching_matrix=branching,
            spectral_radius=float(np.max(np.abs(np.linalg.eigvals(branching)))),
            endogeneity_ratio=0.45,
            log_likelihood=-150.0,
            aic=320.0,
            bic=335.0,
            n_events=[50, 200],
            horizon=3600.0,
            half_lives=np.log(2.0) / beta,
        )

        # Information absorption half-life from News (0) to Price Jumps (1): ln(2)/0.50 = 1.3863s
        hl = self.engine.information_absorption_half_life(from_node=0, to_node=1)
        self.assertAlmostEqual(hl, np.log(2.0) / 0.50, places=4)

        # Impulse response at t=0: phi_10(0) = alpha_10 = 0.40
        t_grid = np.array([0.0, hl])
        ir = self.engine.impulse_response_kernel(t_grid, from_node=0, to_node=1)
        self.assertAlmostEqual(ir[0], 0.40, places=4)
        self.assertAlmostEqual(ir[1], 0.20, places=4)  # halved at t_1/2

    def test_residual_diagnostics_time_change(self):
        """Tests Papangelou compensator residual transformation."""
        mu = np.array([0.05, 0.08])
        alpha = np.array([[0.10, 0.02], [0.20, 0.10]])
        beta = np.array([[1.20, 1.00], [0.80, 1.10]])
        horizon = 1200.0

        events = self.engine.simulate(mu, alpha, beta, horizon=horizon, seed=123)
        self.engine.fit(events, horizon=horizon, max_iter=80)

        diag = self.engine.evaluate_residual_diagnostics(events)
        self.assertEqual(len(diag.compensators), 2)
        # Check that mean compensator inter-arrival is reasonably close to 1.0 (Exp(1))
        self.assertAlmostEqual(diag.mean_inter_arrival[0], 1.0, delta=0.35)
        self.assertAlmostEqual(diag.mean_inter_arrival[1], 1.0, delta=0.35)


if __name__ == "__main__":
    unittest.main()
