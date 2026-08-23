"""
Heuristic Trend and Volatility Regime Filter.

Classifies market environments into Bull, Bear, and Neutral regimes based on
Moving Average trend status (e.g. 200-day SMA) and realized volatility (e.g. 21-day rolling vol).
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


class TrendVolRegimeFilter:
    """
    Trend & Realized Volatility Market Regime Filter.

    Classification Rules:
    - Bull Regime: Price > SMA AND Realized Volatility <= Median Volatility
      (Orderly uptrend with low/moderate volatility)
    - Bear Regime: Price < SMA AND Realized Volatility > Median Volatility
      (Downtrend with heightened uncertainty and elevated volatility)
    - Neutral Regime: Otherwise
      (e.g., volatile blow-off uptrend or low-volatility bottom consolidation)
    """

    def __init__(
        self,
        sma_window: int = 200,
        vol_window: int = 21,
        vol_threshold: Union[str, float] = "median",
        annualization_factor: int = 252,
    ):
        """
        Initialize the TrendVolRegimeFilter.

        Parameters
        ----------
        sma_window : int, default 200
            Window size in trading days for simple moving average.
        vol_window : int, default 21
            Window size in trading days for rolling realized volatility.
        vol_threshold : str or float, default 'median'
            Volatility threshold criterion: 'median', a quantile float (e.g. 0.50),
            or an absolute annualized volatility value.
        annualization_factor : int, default 252
            Trading periods per year.
        """
        self.sma_window = sma_window
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.annualization_factor = annualization_factor

    def classify(
        self,
        prices: Union[pd.Series, pd.DataFrame, np.ndarray],
        returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Classify price history into Bull, Bear, and Neutral regimes.

        Parameters
        ----------
        prices : pd.Series, pd.DataFrame, or np.ndarray
            Asset price series.
        returns : pd.Series, pd.DataFrame, or np.ndarray, optional
            Asset return series. If None, computed as simple percentage returns of prices.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with Price, Return, SMA, Rolling_Vol, Vol_Threshold, Regime, Regime_Code.
        """
        if isinstance(prices, pd.DataFrame):
            price_s = prices.iloc[:, 0].copy()
        elif isinstance(prices, pd.Series):
            price_s = prices.copy()
        else:
            price_s = pd.Series(np.asarray(prices).ravel())

        price_s = price_s.dropna()
        if len(price_s) < max(self.sma_window, self.vol_window):
            raise ValueError(
                f"Price series length ({len(price_s)}) is shorter than maximum window "
                f"({max(self.sma_window, self.vol_window)})."
            )

        if returns is not None:
            if isinstance(returns, (pd.Series, pd.DataFrame)):
                ret_s = returns.iloc[:, 0] if isinstance(returns, pd.DataFrame) else returns
                ret_s = ret_s.reindex(price_s.index)
            else:
                ret_s = pd.Series(np.asarray(returns).ravel(), index=price_s.index)
        else:
            ret_s = price_s.pct_change()

        # Compute SMA and Rolling Realized Volatility (Annualized)
        sma = price_s.rolling(window=self.sma_window).mean()
        rolling_vol = ret_s.rolling(window=self.vol_window).std() * np.sqrt(self.annualization_factor)

        # Volatility threshold
        valid_vol = rolling_vol.dropna()
        if isinstance(self.vol_threshold, str) and self.vol_threshold.lower() == "median":
            threshold_val = float(valid_vol.median())
        elif isinstance(self.vol_threshold, (int, float)):
            if 0.0 < self.vol_threshold < 1.0:
                threshold_val = float(valid_vol.quantile(self.vol_threshold))
            else:
                threshold_val = float(self.vol_threshold)
        else:
            threshold_val = float(valid_vol.median())

        # Regime assignment
        regimes = pd.Series("Neutral", index=price_s.index)
        regime_codes = pd.Series(1, index=price_s.index, dtype=int)  # 0: Bear, 1: Neutral, 2: Bull

        # Valid mask (where both SMA and rolling_vol are available)
        valid_mask = (~sma.isna()) & (~rolling_vol.isna())

        bull_mask = valid_mask & (price_s > sma) & (rolling_vol <= threshold_val)
        bear_mask = valid_mask & (price_s < sma) & (rolling_vol > threshold_val)

        regimes[bull_mask] = "Bull"
        regime_codes[bull_mask] = 2

        regimes[bear_mask] = "Bear"
        regime_codes[bear_mask] = 0

        result_df = pd.DataFrame(
            {
                "Price": price_s,
                "Return": ret_s,
                f"SMA_{self.sma_window}": sma,
                f"Rolling_Vol_{self.vol_window}": rolling_vol,
                "Vol_Threshold": threshold_val,
                "Regime": regimes,
                "Regime_Code": regime_codes,
            },
            index=price_s.index,
        )

        return result_df

    def fit_predict(
        self,
        prices: Union[pd.Series, pd.DataFrame, np.ndarray],
        returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Alias for classify()."""
        return self.classify(prices, returns)

    def regime_metrics(
        self,
        prices: Union[pd.Series, pd.DataFrame, np.ndarray],
        returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Calculate conditional performance metrics across identified regimes.

        Parameters
        ----------
        prices : pd.Series, pd.DataFrame, or np.ndarray
            Asset prices.
        returns : pd.Series, pd.DataFrame, or np.ndarray, optional
            Asset returns.
        risk_free_rate : float, default 0.0
            Annualized risk-free rate.

        Returns
        -------
        metrics_df : pd.DataFrame
            Summary table of conditional returns, volatilities, and Sharpe ratios.
        """
        df = self.classify(prices, returns)
        # Drop warmup rows where SMA/Vol is NaN
        warmup_cols = [f"SMA_{self.sma_window}", f"Rolling_Vol_{self.vol_window}"]
        valid_df = df.dropna(subset=warmup_cols)

        ann = self.annualization_factor
        sqrt_ann = np.sqrt(ann)
        T_total = len(valid_df)

        metrics = []
        for regime_name in ["Bear", "Neutral", "Bull"]:
            subset = valid_df[valid_df["Regime"] == regime_name]
            count = len(subset)
            freq = (count / T_total) * 100.0 if T_total > 0 else 0.0

            if count > 0:
                ret_subset = subset["Return"].dropna()
                mean_daily = float(ret_subset.mean()) if len(ret_subset) > 0 else 0.0
                vol_daily = float(ret_subset.std(ddof=1)) if len(ret_subset) > 1 else 0.0
            else:
                mean_daily = 0.0
                vol_daily = 0.0

            ann_return = mean_daily * ann
            ann_vol = vol_daily * sqrt_ann
            sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-12 else 0.0

            # Average duration calculation: consecutive runs
            durations = self._calculate_durations(valid_df["Regime"] == regime_name)
            avg_duration = float(np.mean(durations)) if len(durations) > 0 else 0.0

            metrics.append({
                "Regime": regime_name,
                "Observations": count,
                "Frequency_Pct": freq,
                "Avg_Duration_Days": avg_duration,
                "Daily_Mean_Return": mean_daily,
                "Annualized_Return": ann_return,
                "Daily_Volatility": vol_daily,
                "Annualized_Volatility": ann_vol,
                "Sharpe_Ratio": sharpe,
            })

        return pd.DataFrame(metrics).set_index("Regime")

    @staticmethod
    def _calculate_durations(boolean_series: pd.Series) -> list:
        """Calculate lengths of consecutive True streaks."""
        durations = []
        current_streak = 0
        for val in boolean_series:
            if val:
                current_streak += 1
            else:
                if current_streak > 0:
                    durations.append(current_streak)
                    current_streak = 0
        if current_streak > 0:
            durations.append(current_streak)
        return durations
