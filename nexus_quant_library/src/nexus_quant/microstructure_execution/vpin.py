"""
Volume Synchronized Probability of Toxicity (VPIN) & High-Frequency Flow Toxicity Engine.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class VPINResult:
    """Output container for VPIN microstructure diagnostics."""
    vpin_series: pd.Series
    bucket_prices: np.ndarray
    bucket_volumes: np.ndarray
    buy_volumes: np.ndarray
    sell_volumes: np.ndarray
    imbalance_series: np.ndarray
    mean_vpin: float
    max_vpin: float
    current_vpin: float
    toxicity_regime: str
    alerts_triggered_count: int


class VPINEngine:
    """
    Volume Synchronized Probability of Toxicity (VPIN) Engine.
    Reference: Easley, López de Prado, and O'Hara (2011, 2012).
    """

    def __init__(
        self,
        n_buckets: int = 50,
        sigma_window: int = 20,
        alert_threshold_95: float = 0.70,
        alert_threshold_99: float = 0.85,
    ):
        self.n_buckets = n_buckets
        self.sigma_window = sigma_window
        self.alert_threshold_95 = alert_threshold_95
        self.alert_threshold_99 = alert_threshold_99

    def compute_vpin_from_ticks(
        self,
        tick_prices: np.ndarray,
        tick_volumes: np.ndarray,
        bucket_size: Optional[float] = None,
    ) -> VPINResult:
        """
        Slice tick-level trades into constant-volume buckets and calculate rolling VPIN.
        """
        total_volume = np.sum(tick_volumes)
        if bucket_size is None:
            # Default to dividing total volume into 200 buckets
            v_bucket = total_volume / 200.0
        else:
            v_bucket = bucket_size

        v_bucket = max(v_bucket, 1.0)

        # 1. Volume Clock Slicing
        bucket_prices_list = []
        bucket_volumes_list = []
        current_bucket_vol = 0.0
        current_bucket_vwap_num = 0.0

        for p, v in zip(tick_prices, tick_volumes):
            rem_v = v
            while rem_v > 0:
                needed = v_bucket - current_bucket_vol
                take = min(rem_v, needed)
                current_bucket_vol += take
                current_bucket_vwap_num += p * take
                rem_v -= take

                if current_bucket_vol >= v_bucket - 1e-8:
                    bucket_prices_list.append(current_bucket_vwap_num / v_bucket)
                    bucket_volumes_list.append(v_bucket)
                    current_bucket_vol = 0.0
                    current_bucket_vwap_num = 0.0

        b_prices = np.array(bucket_prices_list)
        n_b = len(b_prices)
        if n_b < self.n_buckets + 2:
            # If not enough buckets, return basic result
            return VPINResult(
                vpin_series=pd.Series([0.5]),
                bucket_prices=b_prices,
                bucket_volumes=np.array(bucket_volumes_list),
                buy_volumes=np.array([]),
                sell_volumes=np.array([]),
                imbalance_series=np.array([]),
                mean_vpin=0.5,
                max_vpin=0.5,
                current_vpin=0.5,
                toxicity_regime="LOW_SAMPLE",
                alerts_triggered_count=0,
            )

        # 2. Bulk Volume Classification (BVC)
        price_diffs = np.diff(b_prices)
        rolling_sigma = pd.Series(price_diffs).rolling(self.sigma_window, min_periods=2).std().fillna(np.std(price_diffs) or 1.0).values

        # Buy volume: V_tau^B = V * Phi(Delta P / sigma)
        z_scores = price_diffs / np.maximum(rolling_sigma, 1e-6)
        buy_vols = v_bucket * norm.cdf(z_scores)
        sell_vols = v_bucket - buy_vols

        # 3. Rolling VPIN across N buckets: VPIN = sum(|V_B - V_S|) / (N * V)
        abs_imbalance = np.abs(buy_vols - sell_vols)
        rolling_imb_sum = pd.Series(abs_imbalance).rolling(self.n_buckets, min_periods=self.n_buckets).sum().dropna().values
        vpin_vals = rolling_imb_sum / (self.n_buckets * v_bucket)
        vpin_vals = np.clip(vpin_vals, 0.0, 1.0)

        vpin_series = pd.Series(vpin_vals)
        mean_v = float(np.mean(vpin_vals))
        max_v = float(np.max(vpin_vals))
        curr_v = float(vpin_vals[-1])

        alerts = int(np.sum(vpin_vals >= self.alert_threshold_95))

        if curr_v >= self.alert_threshold_99:
            regime = "CRITICAL_TOXICITY_FLASH_CRASH_RISK"
        elif curr_v >= self.alert_threshold_95:
            regime = "HIGH_TOXICITY_ADVERSE_SELECTION"
        else:
            regime = "NORMAL_BALANCED_FLOW"

        return VPINResult(
            vpin_series=vpin_series,
            bucket_prices=b_prices,
            bucket_volumes=np.array(bucket_volumes_list),
            buy_volumes=buy_vols,
            sell_volumes=sell_vols,
            imbalance_series=abs_imbalance,
            mean_vpin=mean_v,
            max_vpin=max_v,
            current_vpin=curr_v,
            toxicity_regime=regime,
            alerts_triggered_count=alerts,
        )
