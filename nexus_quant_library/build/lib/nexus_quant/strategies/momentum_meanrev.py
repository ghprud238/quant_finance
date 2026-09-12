"""
Strategies: Moving Average Mean Reversion & Momentum Trend Following.
"""

from typing import Tuple
import numpy as np
import pandas as pd


class MeanReversionStrategy:
    """Bollinger Bands & Z-score normalized mean reversion trading."""

    @staticmethod
    def generate_signals(
        prices: pd.Series, lookback: int = 20, z_entry: float = 1.8, z_exit: float = 0.3
    ) -> pd.DataFrame:
        sma = prices.rolling(lookback).mean()
        std = prices.rolling(lookback).std(ddof=1)
        z = (prices - sma) / std.replace(0, np.nan)

        position = pd.Series(index=prices.index, data=0.0)
        curr_pos = 0.0
        for t in range(len(prices)):
            val = z.iloc[t]
            if np.isnan(val):
                continue
            # Long entry (oversold)
            if val <= -z_entry:
                curr_pos = 1.0
            # Short entry (overbought)
            elif val >= z_entry:
                curr_pos = -1.0
            # Exit mean reversion
            elif abs(val) <= z_exit:
                curr_pos = 0.0
            position.iloc[t] = curr_pos

        return pd.DataFrame({"Price": prices, "SMA": sma, "Z_Score": z, "Position": position})


class MomentumStrategy:
    """Dual Moving Average & Time-Series Momentum (TSMOM) with volatility targeting."""

    @staticmethod
    def generate_signals(
        prices: pd.Series,
        fast_window: int = 20,
        slow_window: int = 100,
        target_vol: float = 0.12,
    ) -> pd.DataFrame:
        fast_ma = prices.rolling(fast_window).mean()
        slow_ma = prices.rolling(slow_window).mean()
        raw_signal = np.sign(fast_ma - slow_ma).fillna(0.0)

        # Volatility targeting
        ret = prices.pct_change()
        realized_vol = ret.rolling(fast_window).std(ddof=1) * np.sqrt(252.0)
        vol_scalar = (target_vol / realized_vol.replace(0, np.nan)).clip(0.2, 2.0).fillna(1.0)

        target_position = (raw_signal * vol_scalar).clip(-1.5, 1.5)

        return pd.DataFrame({
            "Price": prices,
            "Fast_MA": fast_ma,
            "Slow_MA": slow_ma,
            "Raw_Signal": raw_signal,
            "Position": target_position,
        })
