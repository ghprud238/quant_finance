import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
"""Unit Tests for Module 27: Almgren-Chriss & Optimal Execution Benchmark Models."""

import unittest
import numpy as np
import pandas as pd

from interview_quant.execution.almgren_chriss import AlmgrenChrissModel, ExecutionTrajectoryResult, ExecutionFrontierResult
from interview_quant.execution.benchmark_executors import TWAPExecutor, VWAPExecutor, POVExecutor, ImplementationShortfallAttributor


class TestAlmgrenChrissModel(unittest.TestCase):
    """Tests Almgren-Chriss optimal trajectory, expected shortfall, and execution frontier."""

    def setUp(self):
        self.model = AlmgrenChrissModel(
            total_shares=1_000_000.0,
            horizon=1.0,
            n_intervals=20,
            volatility=0.30,
            temp_impact=2.5e-6,
            perm_impact=2.5e-7,
            fixed_cost=0.0,
            initial_price=100.0,
        )

    def test_risk_neutral_twap_limit(self):
        """When risk aversion lambda -> 0, trajectory converges to uniform linear TWAP."""
        traj = self.model.solve_trajectory(risk_aversion=0.0)
        self.assertAlmostEqual(traj.kappa, 0.0, places=5)
        self.assertEqual(len(traj.holdings), 21)
        self.assertAlmostEqual(traj.holdings[0], 1_000_000.0)
        self.assertAlmostEqual(traj.holdings[-1], 0.0)

        # Uniform slice size
        expected_slice = 1_000_000.0 / 20.0
        for n in traj.trade_sizes:
            self.assertAlmostEqual(n, expected_slice, places=4)

    def test_risk_averse_front_loading(self):
        """Higher risk aversion lambda forces front-loaded trading to reduce holding variance."""
        traj_risk_neutral = self.model.solve_trajectory(risk_aversion=0.0)
        traj_risk_averse = self.model.solve_trajectory(risk_aversion=1e-5)

        # Risk-averse trajectory trades more in earlier intervals than linear TWAP
        self.assertGreater(traj_risk_averse.trade_sizes[0], traj_risk_neutral.trade_sizes[0])
        # And less in terminal intervals
        self.assertLess(traj_risk_averse.trade_sizes[-1], traj_risk_neutral.trade_sizes[-1])

        # Variance is strictly lower under risk-averse execution
        self.assertLess(traj_risk_averse.variance_shortfall, traj_risk_neutral.variance_shortfall)
        # Expected cost is higher due to larger market impact
        self.assertGreater(traj_risk_averse.expected_shortfall, traj_risk_neutral.expected_shortfall)

    def test_efficient_frontier_of_execution(self):
        """Verifies monotonicity of the execution efficient frontier."""
        frontier = self.model.efficient_frontier(n_points=20)
        self.assertEqual(len(frontier.lambda_values), 20)

        # As lambda increases, cost increases and standard deviation decreases
        self.assertTrue(np.all(np.diff(frontier.expected_shortfalls) >= -1e-6))
        self.assertTrue(np.all(np.diff(frontier.std_shortfalls) <= 1e-6))


class TestBenchmarkExecutors(unittest.TestCase):
    """Tests TWAP, VWAP, POV, and implementation shortfall attribution."""

    def test_twap_execution_simulation(self):
        price_path = np.linspace(100.0, 98.0, 21)
        res = TWAPExecutor.simulate_execution(100_000.0, price_path, side='sell')
        self.assertEqual(res.algorithm_name, 'TWAP')
        self.assertEqual(res.executed_shares, 100_000.0)
        self.assertGreater(res.total_cost, 0.0)

    def test_vwap_u_shaped_profile(self):
        profile = VWAPExecutor.u_shaped_volume_profile(10)
        self.assertEqual(len(profile), 10)
        self.assertAlmostEqual(np.sum(profile), 1.0, places=5)
        # U-shaped: open and close volumes higher than midday
        self.assertGreater(profile[0], profile[5])
        self.assertGreater(profile[-1], profile[5])

    def test_implementation_shortfall_attribution(self):
        trade_sizes = np.full(10, 10_000.0)
        exec_prices = np.linspace(99.8, 99.2, 10)

        attribution = ImplementationShortfallAttributor.attribute_costs(
            total_shares=100_000.0,
            decision_price=100.0,
            arrival_price=99.9,
            terminal_price=99.0,
            trade_sizes=trade_sizes,
            execution_prices=exec_prices,
            side='sell',
        )

        self.assertIn('Total_Shortfall_Dollars', attribution)
        self.assertIn('Delay_Cost_Dollars', attribution)
        self.assertIn('Temporary_Impact_Dollars', attribution)
        self.assertIn('Permanent_Impact_Dollars', attribution)
        self.assertGreater(attribution['Total_Shortfall_Dollars'], 0.0)


if __name__ == '__main__':
    unittest.main()
