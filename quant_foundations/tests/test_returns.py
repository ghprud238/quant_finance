"""Comprehensive unit tests for Module 1 (Data Layer & Volatility Analyzer)."""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Ensure package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quant_foundations.analyzer.distribution import (
    fit_distributions,
    jarque_bera_test,
    kurtosis,
    skewness,
)
from quant_foundations.analyzer.returns import (
    annualized_return,
    cumulative_returns,
    log_returns,
    rolling_returns,
    simple_returns,
)
from quant_foundations.analyzer.volatility import (
    close_to_close_volatility,
    garman_klass_volatility,
    parkinson_volatility,
    rogers_satchell_volatility,
    volatility_cone,
    yang_zhang_volatility,
)
from quant_foundations.data.loader import load_factors, load_prices
from quant_foundations.data.synthetic import (
    ALL_TICKERS,
    CORE_TICKERS,
    FACTOR_NAMES,
    generate_and_save_sample_data,
    generate_synthetic_factors,
    generate_synthetic_prices,
)


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test synthetic financial series and factor generators."""

    def setUp(self):
        self.start_date = "2020-01-01"
        self.end_date = "2020-12-31"
        self.seed = 42

    def test_generate_synthetic_prices_structure(self):
        df_prices = generate_synthetic_prices(
            start_date=self.start_date,
            end_date=self.end_date,
            seed=self.seed,
        )
        self.assertIsInstance(df_prices, pd.DataFrame)
        self.assertIsInstance(df_prices.columns, pd.MultiIndex)
        self.assertEqual(df_prices.columns.names, ["Ticker", "Field"])

        # Check all tickers present
        tickers = list(df_prices.columns.levels[0])
        for t in ALL_TICKERS:
            self.assertIn(t, tickers)

        # Check fields present
        fields = list(df_prices.columns.levels[1])
        for f in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(f, fields)

    def test_ohlcv_integrity(self):
        df_prices = generate_synthetic_prices(
            start_date="2022-01-01",
            end_date="2022-06-30",
            seed=101,
        )
        for ticker in ALL_TICKERS:
            df_t = df_prices[ticker]
            # High must be >= max(Open, Close)
            max_oc = np.maximum(df_t["Open"], df_t["Close"])
            self.assertTrue((df_t["High"] >= max_oc - 1e-6).all(), f"High < max(Open, Close) for {ticker}")

            # Low must be <= min(Open, Close)
            min_oc = np.minimum(df_t["Open"], df_t["Close"])
            self.assertTrue((df_t["Low"] <= min_oc + 1e-6).all(), f"Low > min(Open, Close) for {ticker}")

            # Prices and volume must be strictly positive
            self.assertTrue((df_t["Low"] > 0).all(), f"Low <= 0 for {ticker}")
            self.assertTrue((df_t["Volume"] > 0).all(), f"Volume <= 0 for {ticker}")

    def test_generate_synthetic_factors_structure(self):
        df_factors = generate_synthetic_factors(
            start_date=self.start_date,
            end_date=self.end_date,
            seed=self.seed,
        )
        self.assertIsInstance(df_factors, pd.DataFrame)
        for f in FACTOR_NAMES:
            self.assertIn(f, df_factors.columns)

        # Risk-free rate must be non-negative
        self.assertTrue((df_factors["RF"] >= 0.0).all())

    def test_generate_and_save_sample_data(self):
        temp_dir = tempfile.mkdtemp()
        try:
            p_df, f_df = generate_and_save_sample_data(
                data_dir=temp_dir,
                start_date="2023-01-01",
                end_date="2023-03-31",
                seed=42,
            )
            p_path = os.path.join(temp_dir, "sample_prices.csv")
            f_path = os.path.join(temp_dir, "sample_factors.csv")

            self.assertTrue(os.path.exists(p_path))
            self.assertTrue(os.path.exists(f_path))
            self.assertGreater(os.path.getsize(p_path), 0)
            self.assertGreater(os.path.getsize(f_path), 0)
        finally:
            shutil.rmtree(temp_dir)


class TestDataLoader(unittest.TestCase):
    """Test data loader functions and auto-generation fallbacks."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_prices_auto_generate(self):
        # File doesn't exist yet, should auto-generate
        df = load_prices(data_dir=self.temp_dir, auto_generate=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("AAPL", df.columns.levels[0])

    def test_load_prices_filters(self):
        df_aapl = load_prices(data_dir=self.temp_dir, auto_generate=True, ticker="AAPL")
        self.assertIn("Close", df_aapl.columns)
        self.assertIn("Open", df_aapl.columns)

        df_close = load_prices(data_dir=self.temp_dir, auto_generate=True, field="Close")
        self.assertIn("AAPL", df_close.columns)
        self.assertIn("SPY", df_close.columns)

    def test_load_prices_missing_raises(self):
        empty_dir = os.path.join(self.temp_dir, "empty_subdir")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            load_prices(data_dir=empty_dir, auto_generate=False)

    def test_load_factors_auto_generate(self):
        df_fac = load_factors(data_dir=self.temp_dir, auto_generate=True)
        self.assertIsInstance(df_fac, pd.DataFrame)
        self.assertIn("MKT-RF", df_fac.columns)
        self.assertIn("RF", df_fac.columns)


class TestReturnsAnalyzer(unittest.TestCase):
    """Test simple, log, cumulative, rolling, and annualized return calculations."""

    def setUp(self):
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        self.prices_series = pd.Series([100.0, 105.0, 99.75, 104.7375, 115.21125], index=dates, name="Asset")
        self.prices_df = pd.DataFrame(
            {
                "A": [100.0, 110.0, 121.0, 133.1, 146.41],
                "B": [100.0, 90.0, 81.0, 72.9, 65.61],
            },
            index=dates,
        )

    def test_simple_returns(self):
        ret = simple_returns(self.prices_series)
        self.assertTrue(np.isnan(ret.iloc[0]))
        self.assertAlmostEqual(ret.iloc[1], 0.05)
        self.assertAlmostEqual(ret.iloc[2], -0.05)

        ret_filled = simple_returns(self.prices_series, fillna_zero=True)
        self.assertEqual(ret_filled.iloc[0], 0.0)

        # DataFrame support
        df_ret = simple_returns(self.prices_df)
        self.assertAlmostEqual(df_ret["A"].iloc[1], 0.10)
        self.assertAlmostEqual(df_ret["B"].iloc[1], -0.10)

    def test_log_returns(self):
        log_ret = log_returns(self.prices_series)
        self.assertTrue(np.isnan(log_ret.iloc[0]))
        self.assertAlmostEqual(log_ret.iloc[1], np.log(1.05))

        simp_ret = simple_returns(self.prices_series).dropna()
        log_ret_clean = log_ret.dropna()
        # Verify exp(r) - 1 == R
        np.testing.assert_allclose(np.exp(log_ret_clean) - 1.0, simp_ret, rtol=1e-7)

    def test_cumulative_returns(self):
        simp_ret = simple_returns(self.prices_series)
        cum_simp = cumulative_returns(simp_ret, is_log=False)
        self.assertAlmostEqual(cum_simp.iloc[0], 0.0)
        self.assertAlmostEqual(cum_simp.iloc[-1], (115.21125 - 100.0) / 100.0)

        log_ret = log_returns(self.prices_series)
        cum_log = cumulative_returns(log_ret, is_log=True)
        self.assertAlmostEqual(cum_log.iloc[-1], np.log(115.21125 / 100.0))

    def test_rolling_returns(self):
        roll_simp = rolling_returns(self.prices_series, window=2, is_log=False)
        self.assertTrue(np.isnan(roll_simp.iloc[1]))
        expected_2d = (99.75 - 100.0) / 100.0
        self.assertAlmostEqual(roll_simp.iloc[2], expected_2d)

        roll_log = rolling_returns(self.prices_series, window=2, is_log=True)
        self.assertAlmostEqual(roll_log.iloc[2], np.log(99.75 / 100.0))

    def test_annualized_return(self):
        # 10% daily return compounded over 252 days = (1.10)^252 - 1
        daily_ret = pd.Series([0.10] * 252)
        cagr = annualized_return(daily_ret, is_log=False, periods_per_year=252)
        expected_cagr = (1.10 ** 252) - 1.0
        self.assertAlmostEqual(cagr, expected_cagr, places=4)

        # Log return annualized
        daily_log_ret = pd.Series([0.001] * 252)
        ann_log = annualized_return(daily_log_ret, is_log=True, periods_per_year=252)
        self.assertAlmostEqual(ann_log, 0.001 * 252)


class TestVolatilityEstimators(unittest.TestCase):
    """Test Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, and Volatility Cone."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        # Create synthetic OHLC
        c = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.015, size=100)))
        o = c * np.exp(np.random.normal(0, 0.005, size=100))
        h = np.maximum(o, c) * 1.01
        l = np.minimum(o, c) * 0.99
        self.df_ohlc = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=dates)

    def test_close_to_close_volatility(self):
        vol_rolling = close_to_close_volatility(self.df_ohlc["Close"], window=21, annualized=True)
        self.assertIsInstance(vol_rolling, pd.Series)
        self.assertEqual(len(vol_rolling), 100)
        # First 21 price rows yield 20 return observations -> NaN at iloc[20], valid at iloc[21]
        self.assertTrue(np.isnan(vol_rolling.iloc[20]))
        self.assertFalse(np.isnan(vol_rolling.iloc[21]))

        # Full sample scalar
        vol_scalar = close_to_close_volatility(self.df_ohlc["Close"], window=None, annualized=True)
        self.assertIsInstance(vol_scalar, float)
        self.assertGreater(vol_scalar, 0.0)

    def test_parkinson_volatility(self):
        vol_p = parkinson_volatility(self.df_ohlc, window=21, annualized=True)
        self.assertIsInstance(vol_p, pd.Series)
        self.assertFalse(np.isnan(vol_p.iloc[20]))
        self.assertTrue((vol_p.dropna() > 0).all())

        # Exact formula test with synthetic constant H/L ratio
        dates_const = pd.date_range("2023-01-01", periods=10, freq="B")
        h = pd.Series([102.0] * 10, index=dates_const)
        l = pd.Series([100.0] * 10, index=dates_const)
        df_const = pd.DataFrame({"High": h, "Low": l})
        vol_scalar = parkinson_volatility(df_const, window=None, annualized=True, periods_per_year=252)
        expected = np.sqrt(252.0 / (4.0 * np.log(2.0)) * (np.log(102.0 / 100.0) ** 2))
        self.assertAlmostEqual(vol_scalar, expected, places=6)

    def test_garman_klass_volatility(self):
        vol_gk = garman_klass_volatility(self.df_ohlc, window=21, annualized=True)
        self.assertIsInstance(vol_gk, pd.Series)
        self.assertFalse(np.isnan(vol_gk.iloc[20]))
        self.assertTrue((vol_gk.dropna() > 0).all())

    def test_rogers_satchell_volatility(self):
        vol_rs = rogers_satchell_volatility(self.df_ohlc, window=21, annualized=True)
        self.assertIsInstance(vol_rs, pd.Series)
        self.assertFalse(np.isnan(vol_rs.iloc[20]))
        self.assertTrue((vol_rs.dropna() >= 0).all())

    def test_yang_zhang_volatility(self):
        vol_yz = yang_zhang_volatility(self.df_ohlc, window=21, annualized=True)
        self.assertIsInstance(vol_yz, pd.Series)
        # Shift in overnight returns means first valid 21-window value is at iloc[21]
        self.assertTrue(np.isnan(vol_yz.iloc[20]))
        self.assertFalse(np.isnan(vol_yz.iloc[21]))
        self.assertTrue((vol_yz.dropna() >= 0).all())

        vol_scalar = yang_zhang_volatility(self.df_ohlc, window=None, annualized=True)
        self.assertIsInstance(vol_scalar, float)
        self.assertGreater(vol_scalar, 0.0)

    def test_volatility_cone(self):
        cone = volatility_cone(
            self.df_ohlc,
            windows=[10, 20, 30],
            estimator="yang_zhang",
            quantiles=[0.0, 0.25, 0.50, 0.75, 1.0],
        )
        self.assertIsInstance(cone, pd.DataFrame)
        self.assertEqual(list(cone.index), [10, 20, 30])
        for col in ["min", "25%", "50%", "75%", "max", "current"]:
            self.assertIn(col, cone.columns)

        # Monotonicity of quantiles
        for _, row in cone.iterrows():
            self.assertTrue(row["min"] <= row["25%"] <= row["50%"] <= row["75%"] <= row["max"])


class TestDistributionMetrics(unittest.TestCase):
    """Test skewness, kurtosis, Jarque-Bera test, and Gaussian / Student-t distribution fitting."""

    def setUp(self):
        np.random.seed(42)
        # Normal sample
        self.norm_returns = pd.Series(np.random.normal(0.0005, 0.015, size=2000))
        # Heavy-tailed Student-t sample
        self.t_returns = pd.Series(np.random.standard_t(df=4, size=2000) * 0.012 + 0.0005)

    def test_skewness_and_kurtosis(self):
        # Normal data should have skew ~ 0 and excess kurtosis ~ 0
        sk_norm = skewness(self.norm_returns)
        kt_norm = kurtosis(self.norm_returns, excess=True)
        self.assertAlmostEqual(sk_norm, 0.0, delta=0.15)
        self.assertAlmostEqual(kt_norm, 0.0, delta=0.35)

        # Student-t (df=4) should have significant excess kurtosis (> 1.0)
        kt_t = kurtosis(self.t_returns, excess=True)
        self.assertGreater(kt_t, 1.0)

    def test_jarque_bera_test(self):
        # Normal data should fail to reject normality (p > 0.01)
        stat_n, pval_n, is_norm_n = jarque_bera_test(self.norm_returns, alpha=0.01)
        self.assertTrue(is_norm_n)
        self.assertGreater(pval_n, 0.01)

        # Student-t data should reject normality (p < 0.05, is_normal=False)
        stat_t, pval_t, is_norm_t = jarque_bera_test(self.t_returns, alpha=0.05)
        self.assertFalse(is_norm_t)
        self.assertLess(pval_t, 0.05)

    def test_fit_distributions(self):
        fit_results = fit_distributions(self.t_returns)
        self.assertIn("gaussian", fit_results)
        self.assertIn("student_t", fit_results)

        g = fit_results["gaussian"]
        t = fit_results["student_t"]

        for key in ["mean", "std", "log_likelihood", "aic", "bic", "ks_stat", "ks_pvalue"]:
            self.assertIn(key, g)
        for key in ["df", "loc", "scale", "log_likelihood", "aic", "bic", "ks_stat", "ks_pvalue"]:
            self.assertIn(key, t)

        # Student-t should fit heavy-tailed data better than Gaussian:
        # Higher log likelihood and lower AIC
        self.assertGreater(t["log_likelihood"], g["log_likelihood"])
        self.assertLess(t["aic"], g["aic"])
        # Fitted degrees of freedom should be close to 4 (e.g. between 2.5 and 6.0)
        self.assertAlmostEqual(t["df"], 4.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
