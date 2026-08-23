"""Unit tests for Data Loader and Market Data Generator."""

import unittest
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

from quant_risk_models.data.loader import (
    generate_sample_market_data,
    load_portfolio_data,
    TICKERS,
    DEFAULT_PORTFOLIO_WEIGHTS,
)


class TestDataLoader(unittest.TestCase):
    def test_generate_sample_market_data(self):
        prices_df, returns_df, port_ret = generate_sample_market_data(
            start_date='2020-01-01',
            end_date='2022-12-31',
            seed=42
        )
        
        self.assertGreater(len(returns_df), 500)
        self.assertEqual(len(prices_df), len(returns_df))
        self.assertEqual(len(port_ret), len(returns_df))
        
        # Check all tickers present in returns
        for t in TICKERS:
            self.assertIn(t, returns_df.columns)
            
        # Check OHLC integrity (High >= Low, High >= Close, High >= Open)
        for t in TICKERS:
            ohlc = prices_df[t]
            self.assertTrue((ohlc['High'] >= ohlc['Low']).all())
            self.assertTrue((ohlc['High'] >= ohlc['Close'] * 0.999).all())
            self.assertTrue((ohlc['Low'] <= ohlc['Close'] * 1.001).all())

    def test_load_portfolio_data(self):
        prices, returns, port_ret, weights = load_portfolio_data(seed=42)
        self.assertEqual(len(weights), len(DEFAULT_PORTFOLIO_WEIGHTS))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        
        ann_vol = port_ret.std() * (252 ** 0.5)
        ann_ret = port_ret.mean() * 252
        
        # Verify realistic target parameters
        self.assertAlmostEqual(ann_vol, 0.1862, places=2)
        self.assertAlmostEqual(ann_ret, 0.1234, places=2)


if __name__ == '__main__':
    unittest.main()
