"""Project 12: Momentum Trading Strategy.

Implements cross-sectional and time-series momentum techniques:
1. Dual Moving Average Trend Crossover (Fast vs. Slow MA)
2. Time-Series Momentum (TSMOM / 12-1 Month Lookback)
3. Moving Average Convergence Divergence (MACD 12/26/9)
4. Donchian Channel Trend Breakout
5. Multi-Indicator Composite Trend Following
"""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class MomentumResult:
    """Structured container for Momentum strategy output."""
    price: pd.Series
    fast_ma: pd.Series
    slow_ma: pd.Series
    tsmom_return: Optional[pd.Series]
    macd_line: Optional[pd.Series]
    macd_signal_line: Optional[pd.Series]
    macd_hist: Optional[pd.Series]
    donchian_high: Optional[pd.Series]
    donchian_low: Optional[pd.Series]
    raw_signal: pd.Series
    position: pd.Series
    trade_action: pd.Series
    entries_long: pd.Series
    entries_short: pd.Series
    exits: pd.Series

    def to_dataframe(self) -> pd.DataFrame:
        """Combines all strategy series into a single pandas DataFrame."""
        df = pd.DataFrame({
            'Price': self.price,
            'Fast_MA': self.fast_ma,
            'Slow_MA': self.slow_ma,
            'Raw_Signal': self.raw_signal,
            'Position': self.position,
            'Trade_Action': self.trade_action,
            'Entry_Long': self.entries_long,
            'Entry_Short': self.entries_short,
            'Exit': self.exits,
        })
        if self.tsmom_return is not None:
            df['TSMOM_Return'] = self.tsmom_return
        if self.macd_line is not None:
            df['MACD'] = self.macd_line
            df['MACD_Signal'] = self.macd_signal_line
            df['MACD_Hist'] = self.macd_hist
        if self.donchian_high is not None:
            df['Donchian_High'] = self.donchian_high
            df['Donchian_Low'] = self.donchian_low
        return df


def compute_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates MACD Line, Signal Line, and MACD Histogram."""
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line
    return macd_line, macd_signal_line, macd_hist


def compute_tsmom_returns(
    prices: pd.Series,
    lookback: int = 252,
    lag: int = 21,
) -> pd.Series:
    """Calculates Time-Series Momentum (TSMOM) 12-1 month return.

    R_{12-1, t} = (P_{t - lag} / P_{t - lookback}) - 1
    """
    lagged_p = prices.shift(lag)
    base_p = prices.shift(lookback)
    tsmom = (lagged_p / base_p.replace(0, np.nan)) - 1.0
    return tsmom


def compute_donchian_channels(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """Calculates Donchian Channel High and Low over rolling window."""
    donchian_high = high.shift(1).rolling(window=window, min_periods=window).max()
    donchian_low = low.shift(1).rolling(window=window, min_periods=window).min()
    return donchian_high, donchian_low


class MomentumTradingStrategy:
    """Project 12: Momentum & Trend-Following Strategy Engine.

    Supports multiple momentum modalities:
    - 'crossover': Dual Moving Average Crossover (Fast MA > Slow MA)
    - 'tsmom': 12-1 Month Time-Series Momentum (sign of trailing excess return)
    - 'macd': Moving Average Convergence Divergence histogram crossovers
    - 'donchian': Upper/Lower Channel breakout (Turtle system)
    - 'composite': Multi-factor ensemble combining crossover, TSMOM, and MACD
    """

    def __init__(
        self,
        mode: str = 'crossover',
        fast_window: int = 20,
        slow_window: int = 50,
        tsmom_lookback: int = 252,
        tsmom_lag: int = 21,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        donchian_window: int = 20,
        allow_short: bool = True,
        ma_type: str = 'SMA',
    ) -> None:
        valid_modes = {'crossover', 'tsmom', 'macd', 'donchian', 'composite'}
        if mode.lower() not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}")
        if fast_window >= slow_window:
            raise ValueError(f"fast_window ({fast_window}) must be strictly less than slow_window ({slow_window})")

        self.mode = mode.lower()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.tsmom_lookback = tsmom_lookback
        self.tsmom_lag = tsmom_lag
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.donchian_window = donchian_window
        self.allow_short = allow_short
        self.ma_type = ma_type.upper()

    def compute_indicators(
        self,
        prices: Union[pd.Series, np.ndarray],
        high: Optional[Union[pd.Series, np.ndarray]] = None,
        low: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Calculates all technical trend and momentum indicators."""
        if isinstance(prices, np.ndarray):
            price_s = pd.Series(prices)
        else:
            price_s = prices.copy()

        # Dual Moving Averages
        if self.ma_type == 'EMA':
            fast_ma = price_s.ewm(span=self.fast_window, adjust=False).mean()
            slow_ma = price_s.ewm(span=self.slow_window, adjust=False).mean()
        else:
            fast_ma = price_s.rolling(window=self.fast_window, min_periods=self.fast_window).mean()
            slow_ma = price_s.rolling(window=self.slow_window, min_periods=self.slow_window).mean()

        # TSMOM
        tsmom = compute_tsmom_returns(price_s, lookback=self.tsmom_lookback, lag=self.tsmom_lag)

        # MACD
        macd_line, macd_sig, macd_hist = compute_macd(
            price_s, fast_period=self.macd_fast, slow_period=self.macd_slow, signal_period=self.macd_signal
        )

        # Donchian
        donch_high = None
        donch_low = None
        if high is not None and low is not None:
            high_s = pd.Series(high) if isinstance(high, np.ndarray) else high
            low_s = pd.Series(low) if isinstance(low, np.ndarray) else low
            donch_high, donch_low = compute_donchian_channels(high_s, low_s, window=self.donchian_window)
        else:
            donch_high, donch_low = compute_donchian_channels(price_s, price_s, window=self.donchian_window)

        return {
            'price': price_s,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'tsmom_return': tsmom,
            'macd_line': macd_line,
            'macd_signal_line': macd_sig,
            'macd_hist': macd_hist,
            'donchian_high': donch_high,
            'donchian_low': donch_low,
        }

    def generate_signals(
        self,
        prices: Union[pd.Series, np.ndarray],
        high: Optional[Union[pd.Series, np.ndarray]] = None,
        low: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> MomentumResult:
        """Generates momentum positions, signals, and order transitions."""
        ind = self.compute_indicators(prices, high=high, low=low)
        price_s = ind['price']
        n = len(price_s)
        index = price_s.index if isinstance(price_s, pd.Series) else pd.RangeIndex(n)

        raw_signals = np.zeros(n, dtype=float)
        positions = np.zeros(n, dtype=float)
        trade_actions = ['HOLD'] * n
        entries_long = np.zeros(n, dtype=bool)
        entries_short = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)

        fast_ma = ind['fast_ma'].values
        slow_ma = ind['slow_ma'].values
        tsmom = ind['tsmom_return'].values
        macd_hist = ind['macd_hist'].values
        donch_h = ind['donchian_high'].values if ind['donchian_high'] is not None else None
        donch_l = ind['donchian_low'].values if ind['donchian_low'] is not None else None
        p_arr = price_s.values

        for t in range(n):
            if self.mode == 'crossover':
                if np.isnan(fast_ma[t]) or np.isnan(slow_ma[t]):
                    sig = 0.0
                elif fast_ma[t] > slow_ma[t]:
                    sig = 1.0
                elif fast_ma[t] < slow_ma[t]:
                    sig = -1.0 if self.allow_short else 0.0
                else:
                    sig = 0.0

            elif self.mode == 'tsmom':
                if np.isnan(tsmom[t]):
                    sig = 0.0
                elif tsmom[t] > 0:
                    sig = 1.0
                elif tsmom[t] < 0:
                    sig = -1.0 if self.allow_short else 0.0
                else:
                    sig = 0.0

            elif self.mode == 'macd':
                if np.isnan(macd_hist[t]):
                    sig = 0.0
                elif macd_hist[t] > 0:
                    sig = 1.0
                elif macd_hist[t] < 0:
                    sig = -1.0 if self.allow_short else 0.0
                else:
                    sig = 0.0

            elif self.mode == 'donchian':
                if donch_h is None or np.isnan(donch_h[t]) or np.isnan(donch_l[t]):
                    sig = 0.0
                elif p_arr[t] >= donch_h[t]:
                    sig = 1.0
                elif p_arr[t] <= donch_l[t]:
                    sig = -1.0 if self.allow_short else 0.0
                else:
                    sig = positions[t - 1] if t > 0 else 0.0

            elif self.mode == 'composite':
                # Ensemble voting
                votes = []
                if not (np.isnan(fast_ma[t]) or np.isnan(slow_ma[t])):
                    votes.append(1.0 if fast_ma[t] > slow_ma[t] else (-1.0 if self.allow_short else 0.0))
                if not np.isnan(tsmom[t]):
                    votes.append(1.0 if tsmom[t] > 0 else (-1.0 if self.allow_short else 0.0))
                if not np.isnan(macd_hist[t]):
                    votes.append(1.0 if macd_hist[t] > 0 else (-1.0 if self.allow_short else 0.0))

                if len(votes) >= 2:
                    avg_vote = np.mean(votes)
                    if avg_vote >= 0.33:
                        sig = 1.0
                    elif avg_vote <= -0.33:
                        sig = -1.0 if self.allow_short else 0.0
                    else:
                        sig = 0.0
                else:
                    sig = 0.0
            else:
                sig = 0.0

            raw_signals[t] = sig

            # Position update and action logging
            prev_pos = positions[t - 1] if t > 0 else 0.0
            if sig == 1.0 and prev_pos <= 0.0:
                trade_actions[t] = 'BUY' if prev_pos == 0.0 else 'REVERSE_LONG'
                entries_long[t] = True
            elif sig == -1.0 and prev_pos >= 0.0:
                trade_actions[t] = 'SELL_SHORT' if prev_pos == 0.0 else 'REVERSE_SHORT'
                entries_short[t] = True
            elif sig == 0.0 and prev_pos != 0.0:
                trade_actions[t] = 'EXIT_LONG' if prev_pos > 0 else 'EXIT_SHORT'
                exits[t] = True
            else:
                trade_actions[t] = 'HOLD'

            positions[t] = sig

        return MomentumResult(
            price=price_s,
            fast_ma=ind['fast_ma'],
            slow_ma=ind['slow_ma'],
            tsmom_return=ind['tsmom_return'],
            macd_line=ind['macd_line'],
            macd_signal_line=ind['macd_signal_line'],
            macd_hist=ind['macd_hist'],
            donchian_high=ind['donchian_high'],
            donchian_low=ind['donchian_low'],
            raw_signal=pd.Series(raw_signals, index=index, name='Raw_Signal'),
            position=pd.Series(positions, index=index, name='Position'),
            trade_action=pd.Series(trade_actions, index=index, name='Trade_Action'),
            entries_long=pd.Series(entries_long, index=index, name='Entry_Long'),
            entries_short=pd.Series(entries_short, index=index, name='Entry_Short'),
            exits=pd.Series(exits, index=index, name='Exit'),
        )
