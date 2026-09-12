"""Comprehensive unit tests for Core Math & Foundations (Track 1)."""
import unittest
import numpy as np
import pandas as pd
from nexus_quant.core.math_utils import (
    is_positive_definite,
    nearest_correlation_matrix,
    cholesky_factor,
    skewness,
    kurtosis,
    jarque_bera_stat,
    solve_constrained_qp,
)
from nexus_quant.core.data_engine import UnifiedDataEngine
from nexus_quant.foundations.returns import (
    simple_returns,
    log_returns,
    cumulative_returns,
    annualized_return,
    rolling_returns,
)
from nexus_quant.foundations.volatility import (
    close_to_close_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
    volatility_cone,
)
from nexus_quant.foundations.regimes import (
    GaussianHMMRegimeDetector,
    TrendVolRegimeFilter,
)


class TestFoundations(unittest.TestCase):

    def setUp(self):
        self.engine = UnifiedDataEngine(seed=42)
        self.df_ohlcv = self.engine.generate_equity_ohlcv(tickers=["AAPL", "SPY"], start_date="2022-01-01", end_date="2023-12-31")
        self.yields = self.engine.generate_macro_yield_data(start_date="2022-01-01", end_date="2023-12-31")
        self.crypto = self.engine.generate_crypto_data(start_date="2022-01-01", end_date="2023-12-31")
        self.carbon = self.engine.generate_carbon_data(start_date="2022-01-01", end_date="2023-12-31")

    def test_data_generation_engines(self):
        self.assertGreater(len(self.df_ohlcv), 400)
        self.assertIn("10Y", self.yields.columns)
        self.assertIn("BTC_Price", self.crypto.columns)
        self.assertIn("EUA_Carbon_Price", self.carbon.columns)

    def test_math_utilities(self):
        mat = np.array([[1.0, 0.8], [0.8, 1.0]])
        self.assertTrue(is_positive_definite(mat))
        non_psd = np.array([[1.0, 1.5], [1.5, 1.0]])
        self.assertFalse(is_positive_definite(non_psd))
        psd = nearest_correlation_matrix(non_psd)
        self.assertTrue(is_positive_definite(psd))
        L = cholesky_factor(psd)
        self.assertEqual(L.shape, (2, 2))

        data = np.random.normal(0, 1, 500)
        s = skewness(data)
        k = kurtosis(data)
        jb_stat, jb_p, is_norm = jarque_bera_stat(data)
        self.assertIsInstance(s, float)
        self.assertIsInstance(k, float)
        self.assertIsInstance(is_norm, bool)

    def test_returns_calculations(self):
        aapl_close = self.df_ohlcv["AAPL"]["Close"]
        s_ret = simple_returns(aapl_close).dropna()
        l_ret = log_returns(aapl_close).dropna()
        cum_ret = cumulative_returns(s_ret)
        roll_ret = rolling_returns(aapl_close, window=21).dropna()

        self.assertEqual(len(s_ret), len(l_ret))
        self.assertGreater(cum_ret.iloc[-1], 0.0)
        self.assertEqual(len(roll_ret), len(aapl_close) - 21)

        cagr_geom = annualized_return(s_ret, geometric=True)
        cagr_arith = annualized_return(s_ret, geometric=False)
        self.assertIsInstance(cagr_geom, float)
        self.assertIsInstance(cagr_arith, float)

    def test_range_volatilities_and_cone(self):
        aapl_ohlc = self.df_ohlcv["AAPL"]
        vol_cc = close_to_close_volatility(aapl_ohlc["Close"], window=21).dropna()
        vol_p = parkinson_volatility(aapl_ohlc, window=21).dropna()
        vol_gk = garman_klass_volatility(aapl_ohlc, window=21).dropna()
        vol_rs = rogers_satchell_volatility(aapl_ohlc, window=21).dropna()
        vol_yz = yang_zhang_volatility(aapl_ohlc, window=21).dropna()

        self.assertTrue(np.all(vol_cc > 0.0))
        self.assertTrue(np.all(vol_p > 0.0))
        self.assertTrue(np.all(vol_gk > 0.0))
        self.assertTrue(np.all(vol_rs > 0.0))
        self.assertTrue(np.all(vol_yz > 0.0))

        cone = volatility_cone(aapl_ohlc, windows=[10, 21, 63])
        self.assertIn("P25", cone.columns)
        self.assertIn("P75", cone.columns)
        self.assertIn("Median", cone.columns)

    def test_gaussian_hmm_regimes(self):
        spy_ret = simple_returns(self.df_ohlcv["SPY"]["Close"]).dropna()
        hmm = GaussianHMMRegimeDetector(n_states=3, max_iter=30, seed=42)
        hmm.fit(spy_ret)
        states = hmm.predict(spy_ret)
        self.assertEqual(len(states), len(spy_ret))
        self.assertTrue(set(np.unique(states)).issubset({0, 1, 2}))
        # Verify monotonic mean sorting
        self.assertLessEqual(hmm.means_[0], hmm.means_[1])
        self.assertLessEqual(hmm.means_[1], hmm.means_[2])

        # Test TrendVol filter
        regimes = TrendVolRegimeFilter.classify(self.df_ohlcv["SPY"]["Close"], sma_window=50, vol_window=10)
        self.assertTrue(set(regimes.unique()).issubset({"BULL", "BEAR", "NEUTRAL"}))


if __name__ == "__main__":
    unittest.main()
