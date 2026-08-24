"""Project 11: Moving Average Mean Reversion Strategy.

Implements statistical mean-reversion trading using rolling Moving Averages (SMA/EMA),
Bollinger Bands, Normalized Z-score tracking, and optional RSI oscillator confirmation.
"""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class MeanReversionResult:
    """Structured container for Mean Reversion strategy output."""
    price: pd.Series
    ma: pd.Series
    rolling_std: pd.Series
    upper_band: pd.Series
    lower_band: pd.Series
    z_score: pd.Series
    rsi: Optional[pd.Series]
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
            'MA': self.ma,
            'Rolling_Std': self.rolling_std,
            'Upper_Band': self.upper_band,
            'Lower_Band': self.lower_band,
            'Z_Score': self.z_score,
            'Raw_Signal': self.raw_signal,
            'Position': self.position,
            'Trade_Action': self.trade_action,
            'Entry_Long': self.entries_long,
            'Entry_Short': self.entries_short,
            'Exit': self.exits,
        })
        if self.rsi is not None:
            df['RSI'] = self.rsi
        return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's Relative Strength Index (RSI).

    Parameters
    ----------
    prices : pd.Series
        Asset close price series.
    period : int, default 14
        Lookback period for average gains and losses.

    Returns
    -------
    pd.Series
        RSI values bounded between [0, 100].
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder exponential moving average smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle zero loss edge cases
    rsi = rsi.fillna(100.0).where(avg_gain != 0, 0.0)
    return rsi


def compute_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    ma_type: str = 'SMA',
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates Bollinger Bands (Middle, Upper, Lower).

    Parameters
    ----------
    prices : pd.Series
        Price series.
    window : int, default 20
        Rolling lookback period.
    num_std : float, default 2.0
        Standard deviation multiplier.
    ma_type : str, default 'SMA'
        'SMA' or 'EMA'.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (Middle MA, Upper Band, Lower Band)
    """
    if ma_type.upper() == 'EMA':
        ma = prices.ewm(span=window, adjust=False).mean()
    else:
        ma = prices.rolling(window=window, min_periods=window).mean()

    rolling_std = prices.rolling(window=window, min_periods=window).std(ddof=1)
    upper_band = ma + (num_std * rolling_std)
    lower_band = ma - (num_std * rolling_std)
    return ma, upper_band, lower_band


def compute_zscore(
    prices: pd.Series,
    window: int = 20,
    ma_type: str = 'SMA',
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes rolling price Z-score relative to moving average.

    Z_t = (P_t - MA_t) / sigma_t
    """
    if ma_type.upper() == 'EMA':
        ma = prices.ewm(span=window, adjust=False).mean()
    else:
        ma = prices.rolling(window=window, min_periods=window).mean()

    std = prices.rolling(window=window, min_periods=window).std(ddof=1)
    # Avoid zero division
    std_safe = std.replace(0, np.nan)
    z_score = (prices - ma) / std_safe
    return z_score, ma, std


class MovingAverageMeanReversionStrategy:
    """Project 11: Systematic Moving Average Mean Reversion Strategy.

    Identifies statistical overbought and oversold conditions where prices
    deviate significantly from their rolling equilibrium (Moving Average).
    Generates mean-reverting entry and exit orders based on Z-score thresholds
    and Bollinger Band band-touches, with optional RSI momentum exhaustion filtering.
    """

    def __init__(
        self,
        lookback_window: int = 20,
        ma_type: str = 'SMA',
        num_std: float = 2.0,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        stop_loss_z: float = 3.5,
        use_rsi_filter: bool = False,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        allow_short: bool = True,
    ) -> None:
        if lookback_window < 2:
            raise ValueError(f"lookback_window must be >= 2, got {lookback_window}")
        if z_entry <= z_exit:
            raise ValueError(f"z_entry ({z_entry}) must be strictly greater than z_exit ({z_exit})")
        if z_entry <= 0 or z_exit < 0:
            raise ValueError("z_entry and z_exit must be positive values")

        self.lookback_window = lookback_window
        self.ma_type = ma_type.upper()
        self.num_std = num_std
        self.z_entry = float(z_entry)
        self.z_exit = float(z_exit)
        self.stop_loss_z = float(stop_loss_z)
        self.use_rsi_filter = use_rsi_filter
        self.rsi_period = rsi_period
        self.rsi_oversold = float(rsi_oversold)
        self.rsi_overbought = float(rsi_overbought)
        self.allow_short = allow_short

    def compute_indicators(self, prices: Union[pd.Series, np.ndarray]) -> Dict[str, pd.Series]:
        """Calculates all mathematical indicators for mean-reversion analysis."""
        if isinstance(prices, np.ndarray):
            prices_series = pd.Series(prices)
        else:
            prices_series = prices.copy()

        z_score, ma, rolling_std = compute_zscore(
            prices_series,
            window=self.lookback_window,
            ma_type=self.ma_type,
        )
        upper_band = ma + (self.num_std * rolling_std)
        lower_band = ma - (self.num_std * rolling_std)

        rsi_series = None
        if self.use_rsi_filter:
            rsi_series = compute_rsi(prices_series, period=self.rsi_period)

        return {
            'price': prices_series,
            'ma': ma,
            'rolling_std': rolling_std,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'z_score': z_score,
            'rsi': rsi_series,
        }

    def generate_signals(self, prices: Union[pd.Series, np.ndarray]) -> MeanReversionResult:
        """Generates discrete trading signals, positions, and order actions.

        State-Machine Execution Logic:
        - Long Entry: Z_t <= -z_entry (and RSI <= rsi_oversold if filter active)
        - Short Entry: Z_t >= +z_entry (and RSI >= rsi_overbought if filter active)
        - Long Exit: Z_t >= -z_exit (mean reversion toward zero) or Z_t <= -stop_loss_z
        - Short Exit: Z_t <= +z_exit (mean reversion toward zero) or Z_t >= +stop_loss_z

        Returns
        -------
        MeanReversionResult
            Dataclass containing full indicator history and trade execution signals.
        """
        ind = self.compute_indicators(prices)
        price_s = ind['price']
        z_s = ind['z_score'].values
        rsi_s = ind['rsi'].values if ind['rsi'] is not None else None
        n = len(price_s)

        positions = np.zeros(n, dtype=float)
        raw_signals = np.zeros(n, dtype=float)
        trade_actions = ['HOLD'] * n
        entries_long = np.zeros(n, dtype=bool)
        entries_short = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)

        current_pos = 0.0

        for t in range(n):
            z = z_s[t]
            if np.isnan(z):
                positions[t] = 0.0
                continue

            rsi_val = rsi_s[t] if rsi_s is not None else None
            rsi_ok_long = (rsi_val is None) or (rsi_val <= self.rsi_oversold)
            rsi_ok_short = (rsi_val is None) or (rsi_val >= self.rsi_overbought)

            # Raw indicator signal
            if z <= -self.z_entry and rsi_ok_long:
                raw_signals[t] = 1.0
            elif z >= self.z_entry and rsi_ok_short:
                raw_signals[t] = -1.0 if self.allow_short else 0.0
            else:
                raw_signals[t] = 0.0

            # State transition logic
            if current_pos == 0.0:
                # Flat state: look for entry
                if z <= -self.z_entry and rsi_ok_long:
                    current_pos = 1.0
                    trade_actions[t] = 'BUY'
                    entries_long[t] = True
                elif z >= self.z_entry and rsi_ok_short and self.allow_short:
                    current_pos = -1.0
                    trade_actions[t] = 'SELL_SHORT'
                    entries_short[t] = True

            elif current_pos == 1.0:
                # Long state: look for exit or stop-loss or reversal
                if z >= -self.z_exit or z <= -self.stop_loss_z:
                    # Target reached or Stop loss triggered
                    current_pos = 0.0
                    trade_actions[t] = 'EXIT_LONG'
                    exits[t] = True
                elif z >= self.z_entry and rsi_ok_short and self.allow_short:
                    # Direct reversal to short
                    current_pos = -1.0
                    trade_actions[t] = 'REVERSE_SHORT'
                    entries_short[t] = True

            elif current_pos == -1.0:
                # Short state: look for exit or stop-loss or reversal
                if z <= self.z_exit or z >= self.stop_loss_z:
                    current_pos = 0.0
                    trade_actions[t] = 'EXIT_SHORT'
                    exits[t] = True
                elif z <= -self.z_entry and rsi_ok_long:
                    # Direct reversal to long
                    current_pos = 1.0
                    trade_actions[t] = 'REVERSE_LONG'
                    entries_long[t] = True

            positions[t] = current_pos

        index = price_s.index if isinstance(price_s, pd.Series) else pd.RangeIndex(n)

        return MeanReversionResult(
            price=price_s,
            ma=ind['ma'],
            rolling_std=ind['rolling_std'],
            upper_band=ind['upper_band'],
            lower_band=ind['lower_band'],
            z_score=ind['z_score'],
            rsi=ind['rsi'],
            raw_signal=pd.Series(raw_signals, index=index, name='Raw_Signal'),
            position=pd.Series(positions, index=index, name='Position'),
            trade_action=pd.Series(trade_actions, index=index, name='Trade_Action'),
            entries_long=pd.Series(entries_long, index=index, name='Entry_Long'),
            entries_short=pd.Series(entries_short, index=index, name='Entry_Short'),
            exits=pd.Series(exits, index=index, name='Exit'),
        )
