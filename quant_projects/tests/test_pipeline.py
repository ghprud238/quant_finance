"""Unit tests for Module 29: End-to-End Quant Research Pipeline."""

import unittest
import numpy as np
import pandas as pd

from interview_quant.data.loader import generate_market_data
from interview_quant.pipeline.workflow import (
    QuantResearchPipeline,
    DataSanityReport,
    FeatureStoreReport,
    BacktestTearSheet,
    ProductionDeploymentReport,
)


class TestQuantResearchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_data = generate_market_data(
            tickers=["SPY", "QQQ", "AAPL", "MSFT"],
            start_date="2019-01-01",
            end_date="2022-12-31",
            seed=42,
        )
        cls.pipeline = QuantResearchPipeline(
            risk_free_rate=0.02,
            target_volatility=0.10,
            transaction_cost_bps=5.0,
            half_spread_bps=2.5,
            borrow_cost_bps=50.0,
            adv_usd=50_000_000.0,
        )

    def test_stage_1_data_validation(self):
        # Inject artificial anomaly and missing values
        bad_data = self.raw_data.copy()
        bad_data.iloc[5, bad_data.columns.get_loc(("AAPL", "High"))] = 10.0
        bad_data.iloc[5, bad_data.columns.get_loc(("AAPL", "Low"))] = 50.0
        bad_data.iloc[10, bad_data.columns.get_loc(("SPY", "Close"))] = np.nan

        clean_df, report = self.pipeline.stage_1_validate_data(bad_data)
        self.assertIsInstance(report, DataSanityReport)
        self.assertTrue(report.total_records > 0)
        self.assertTrue(report.price_anomalies_detected >= 1)
        self.assertTrue(report.missing_values_imputed >= 1)
        self.assertFalse(clean_df.isna().any().any())

    def test_stage_2_feature_engineering(self):
        clean_df, _ = self.pipeline.stage_1_validate_data(self.raw_data)
        features_df, report = self.pipeline.stage_2_engineer_features(clean_df, frac_diff_d=0.35)

        self.assertIsInstance(report, FeatureStoreReport)
        self.assertGreater(report.n_features, 10)
        self.assertGreater(report.stationary_features_count, 0)
        self.assertFalse(features_df.isna().any().any())
        self.assertGreater(len(features_df.index), 100)

    def test_stage_3_backtest_accounting(self):
        clean_df, _ = self.pipeline.stage_1_validate_data(self.raw_data)
        close = clean_df.xs("Close", level="Field", axis=1)

        # Equal weight long strategy
        target_weights = pd.DataFrame(0.25, index=close.index, columns=close.columns)
        bt = self.pipeline.stage_3_backtest(clean_df, target_weights)

        self.assertIn("gross_returns", bt)
        self.assertIn("net_returns", bt)
        self.assertIn("cumulative_equity", bt)
        self.assertIn("turnover_daily", bt)
        self.assertIn("friction", bt)

        # Net returns must be strictly less than or equal to gross returns due to friction
        self.assertTrue((bt["net_returns"] <= bt["gross_returns"] + 1e-9).all())

    def test_stage_4_evaluation_tearsheet(self):
        clean_df, _ = self.pipeline.stage_1_validate_data(self.raw_data)
        close = clean_df.xs("Close", level="Field", axis=1)
        target_weights = pd.DataFrame(0.25, index=close.index, columns=close.columns)
        bt = self.pipeline.stage_3_backtest(clean_df, target_weights)

        ts = self.pipeline.stage_4_evaluate(bt, strategy_name="Equal Weight Long")
        self.assertIsInstance(ts, BacktestTearSheet)
        self.assertIsInstance(ts.cagr, float)
        self.assertIsInstance(ts.sharpe_ratio, float)
        self.assertIsInstance(ts.max_drawdown, float)
        self.assertLessEqual(ts.max_drawdown, 0.0)
        self.assertGreaterEqual(ts.win_rate, 0.0)
        self.assertLessEqual(ts.win_rate, 1.0)
        self.assertGreater(ts.estimated_capacity_usd, 0.0)
        self.assertIn("2020_COVID_Liquidity_Shock", ts.stress_test_results)
        self.assertFalse(ts.metrics_table.empty)

    def test_stage_5_deployment_health(self):
        clean_df, _ = self.pipeline.stage_1_validate_data(self.raw_data)
        close = clean_df.xs("Close", level="Field", axis=1)
        target_weights = pd.DataFrame(0.25, index=close.index, columns=close.columns)
        bt = self.pipeline.stage_3_backtest(clean_df, target_weights)
        ts = self.pipeline.stage_4_evaluate(bt)

        deploy_rep = self.pipeline.stage_5_deployment_and_health_monitor(
            ts, live_returns=bt["net_returns"].iloc[-50:], feature_drift_pvalue=0.60
        )
        self.assertIsInstance(deploy_rep, ProductionDeploymentReport)
        self.assertGreaterEqual(deploy_rep.production_readiness_score, 0.0)
        self.assertLessEqual(deploy_rep.production_readiness_score, 100.0)
        self.assertIn(deploy_rep.drift_alert_level, ["GREEN", "YELLOW", "RED"])

    def test_full_pipeline_integration(self):
        def dummy_strategy(df):
            close = df.xs("Close", level="Field", axis=1)
            sma_20 = close.rolling(20).mean()
            return (close > sma_20).astype(float) * 0.25

        pipeline_result = self.pipeline.run_full_pipeline(
            raw_market_data=self.raw_data,
            strategy_logic_fn=dummy_strategy,
            strategy_name="Trend Filtered Strategy",
        )
        self.assertIn("data_report", pipeline_result)
        self.assertIn("feature_report", pipeline_result)
        self.assertIn("backtest_output", pipeline_result)
        self.assertIn("tear_sheet", pipeline_result)
        self.assertIn("deploy_report", pipeline_result)
        self.assertTrue(pipeline_result["data_report"].is_valid)


if __name__ == "__main__":
    unittest.main()
