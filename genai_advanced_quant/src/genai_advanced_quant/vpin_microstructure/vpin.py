"""Volume Synchronized Probability of Toxicity (VPIN) Engine.

Implementation of Easley, López de Prado, and O'Hara (2011, 2012)
market microstructure framework for measuring order flow toxicity,
adverse selection, and flash-crash risk in high-frequency trading.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


@dataclass
class VolumeBucket:
    """Individual volume bucket on the volume clock."""
    bucket_idx: int
    start_time: Any
    end_time: Any
    volume: float
    vwap: float
    buy_volume: float
    sell_volume: float
    delta_p: float
    imbalance: float
    vpin: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_idx": self.bucket_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "volume": self.volume,
            "vwap": self.vwap,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "delta_p": self.delta_p,
            "imbalance": self.imbalance,
            "vpin": self.vpin,
        }


@dataclass
class ToxicityAlert:
    """Alert triggered when VPIN crosses historical risk percentiles."""
    timestamp: Any
    bucket_idx: int
    vpin_value: float
    percentile_rank: float
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "bucket_idx": self.bucket_idx,
            "vpin_value": self.vpin_value,
            "percentile_rank": self.percentile_rank,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class VPINResult:
    """Comprehensive container for VPIN computation results and diagnostics."""
    vpin_series: pd.Series
    buckets_df: pd.DataFrame
    alerts: List[ToxicityAlert]
    bucket_size: float
    n_buckets: int
    mean_vpin: float
    max_vpin: float
    min_vpin: float
    current_vpin: float
    current_toxicity_level: str
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        """Returns structured DataFrame summary of VPIN metrics."""
        rows = [
            {"Metric": "Volume Bucket Size (V)", "Value": f"{self.bucket_size:,.0f} shares"},
            {"Metric": "Rolling Window (N Buckets)", "Value": f"{self.n_buckets} buckets"},
            {"Metric": "Total Volume Buckets", "Value": f"{len(self.buckets_df):,}"},
            {"Metric": "Mean VPIN", "Value": f"{self.mean_vpin:.4f} ({self.mean_vpin:.2%})"},
            {"Metric": "Max VPIN (Peak Toxicity)", "Value": f"{self.max_vpin:.4f} ({self.max_vpin:.2%})"},
            {"Metric": "Min VPIN", "Value": f"{self.min_vpin:.4f} ({self.min_vpin:.2%})"},
            {"Metric": "Current Real-Time VPIN", "Value": f"{self.current_vpin:.4f} ({self.current_vpin:.2%})"},
            {"Metric": "Current Toxicity Regime", "Value": self.current_toxicity_level},
            {"Metric": "Total Toxicity Alerts", "Value": f"{len(self.alerts)} alerts"},
        ]
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns consolidated DataFrame containing price, volumes, imbalances and VPIN."""
        return self.buckets_df.copy()


class VPINEngine:
    """Volume Synchronized Probability of Toxicity (VPIN) Calculation Engine.

    References:
        Easley, D., López de Prado, M., & O'Hara, M. (2011).
        The Microstructure of the 'Flash Crash': Flow Toxicity,
        Liquidity Deficits, and the Volume-Synchronized Probability of Toxicity.
        Journal of Portfolio Management, 37(2), 118-128.

        Easley, D., López de Prado, M., & O'Hara, M. (2012).
        Flow Toxicity and Liquidity in a High-Frequency World.
        The Review of Financial Studies, 25(5), 1457-1493.
    """

    def __init__(
        self,
        bucket_size: Optional[float] = None,
        n_buckets: int = 50,
        sigma_window: int = 20,
        alert_threshold_95: float = 95.0,
        alert_threshold_99: float = 99.0,
    ) -> None:
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self.sigma_window = sigma_window
        self.alert_threshold_95 = alert_threshold_95
        self.alert_threshold_99 = alert_threshold_99

    def compute_volume_buckets(
        self,
        trades_df: pd.DataFrame,
        bucket_size: float,
    ) -> pd.DataFrame:
        if trades_df.empty:
            raise ValueError("trades_df cannot be empty.")
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}")

        prices = trades_df["price"].values.astype(float)
        volumes = trades_df["volume"].values.astype(float)
        timestamps = trades_df["timestamp"].values

        n_trades = len(trades_df)
        buckets: List[Dict[str, Any]] = []

        curr_bucket_idx = 0
        curr_vol = 0.0
        curr_dollar_vol = 0.0
        start_time = timestamps[0]

        for i in range(n_trades):
            p = prices[i]
            v = volumes[i]
            t = timestamps[i]

            remaining_v = v

            while remaining_v > 0:
                needed_v = bucket_size - curr_vol

                if remaining_v < needed_v:
                    curr_vol += remaining_v
                    curr_dollar_vol += p * remaining_v
                    remaining_v = 0.0
                else:
                    curr_dollar_vol += p * needed_v
                    curr_vol += needed_v
                    remaining_v -= needed_v

                    vwap = curr_dollar_vol / bucket_size
                    buckets.append({
                        "bucket_idx": curr_bucket_idx,
                        "start_time": start_time,
                        "end_time": t,
                        "volume": bucket_size,
                        "vwap": vwap,
                    })

                    curr_bucket_idx += 1
                    curr_vol = 0.0
                    curr_dollar_vol = 0.0
                    start_time = t

        if not buckets:
            raise ValueError(f"Total volume ({volumes.sum()}) is less than bucket_size ({bucket_size}).")

        df_buckets = pd.DataFrame(buckets)
        df_buckets["delta_p"] = df_buckets["vwap"].diff().fillna(0.0)
        return df_buckets

    def bulk_volume_classification(
        self,
        buckets_df: pd.DataFrame,
        sigma_window: int = 20,
    ) -> pd.DataFrame:
        df = buckets_df.copy()
        v = df["volume"].values
        delta_p = df["delta_p"].values

        rolling_std = df["delta_p"].rolling(window=sigma_window, min_periods=2).std()
        sample_std = np.std(delta_p[delta_p != 0]) if np.any(delta_p != 0) else 1.0
        if sample_std == 0:
            sample_std = 1.0
        rolling_std = rolling_std.fillna(sample_std).replace(0.0, sample_std)

        z = delta_p / rolling_std.values
        buy_fractions = stats.norm.cdf(z)
        buy_vols = v * buy_fractions
        sell_vols = v * (1.0 - buy_fractions)
        imbalances = np.abs(buy_vols - sell_vols)

        df["sigma_dp"] = rolling_std.values
        df["z_score"] = z
        df["buy_volume"] = buy_vols
        df["sell_volume"] = sell_vols
        df["imbalance"] = imbalances

        return df

    def compute_vpin(
        self,
        trades_df: pd.DataFrame,
        bucket_size: Optional[float] = None,
        n_buckets: Optional[int] = None,
        sigma_window: Optional[int] = None,
        alert_percentile_95: Optional[float] = None,
        alert_percentile_99: Optional[float] = None,
    ) -> VPINResult:
        trades_df = trades_df.rename(columns={c: str(c).lower() for c in trades_df.columns})
        N = n_buckets if n_buckets is not None else self.n_buckets
        sig_win = sigma_window if sigma_window is not None else self.sigma_window
        p95_thresh = alert_percentile_95 if alert_percentile_95 is not None else self.alert_threshold_95
        p99_thresh = alert_percentile_99 if alert_percentile_99 is not None else self.alert_threshold_99

        total_vol = trades_df["volume"].sum()
        if bucket_size is None:
            if self.bucket_size is not None:
                V = self.bucket_size
            else:
                V = max(100.0, total_vol / 400.0)
        else:
            V = bucket_size

        buckets_df = self.compute_volume_buckets(trades_df, bucket_size=V)
        bvc_df = self.bulk_volume_classification(buckets_df, sigma_window=sig_win)

        rolling_imbalance = bvc_df["imbalance"].rolling(window=N, min_periods=1).sum()
        rolling_total_vol = bvc_df["volume"].rolling(window=N, min_periods=1).sum()

        vpin_series = rolling_imbalance / rolling_total_vol
        bvc_df["vpin"] = vpin_series.values

        vpin_values = vpin_series.values
        percentiles = np.zeros(len(vpin_values))
        alerts: List[ToxicityAlert] = []

        for i in range(len(vpin_values)):
            curr_vpin = vpin_values[i]
            hist = vpin_values[: i + 1]
            pct = (np.sum(hist <= curr_vpin) / len(hist)) * 100.0
            percentiles[i] = pct

            if i >= N:
                t = bvc_df["end_time"].iloc[i]
                if pct >= p99_thresh:
                    alerts.append(
                        ToxicityAlert(
                            timestamp=t,
                            bucket_idx=i,
                            vpin_value=curr_vpin,
                            percentile_rank=pct,
                            severity="CRITICAL",
                            message=f"CRITICAL TOXICITY WARNING: VPIN reached {curr_vpin:.2%} (Percentile: {pct:.1f}%). High probability of toxic informed trading / flash-crash risk.",
                        )
                    )
                elif pct >= p95_thresh:
                    alerts.append(
                        ToxicityAlert(
                            timestamp=t,
                            bucket_idx=i,
                            vpin_value=curr_vpin,
                            percentile_rank=pct,
                            severity="WARNING",
                            message=f"ELEVATED TOXICITY ALERT: VPIN at {curr_vpin:.2%} (Percentile: {pct:.1f}%). Liquidity providers advised to widen spreads.",
                        )
                    )

        bvc_df["percentile_rank"] = percentiles
        latest_vpin = vpin_values[-1]
        latest_pct = percentiles[-1]

        if latest_pct >= p99_thresh:
            current_regime = "CRITICAL (Severe Toxicity / Flash Crash Risk)"
        elif latest_pct >= p95_thresh:
            current_regime = "ELEVATED (High Adverse Selection Risk)"
        elif latest_pct >= 75.0:
            current_regime = "MODERATE (Active Informed Trading)"
        else:
            current_regime = "NORMAL (Balanced Retail & Market-Making Flow)"

        mean_vpin = float(np.mean(vpin_values))
        max_vpin = float(np.max(vpin_values))
        min_vpin = float(np.min(vpin_values))

        summary_metrics = {
            "bucket_size": V,
            "n_buckets": N,
            "mean_vpin": mean_vpin,
            "max_vpin": max_vpin,
            "min_vpin": min_vpin,
            "current_vpin": latest_vpin,
            "current_percentile": latest_pct,
            "current_regime": current_regime,
            "total_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a.severity == "CRITICAL"),
            "warning_alerts": sum(1 for a in alerts if a.severity == "WARNING"),
        }

        return VPINResult(
            vpin_series=pd.Series(vpin_values, index=bvc_df["end_time"], name="VPIN"),
            buckets_df=bvc_df,
            alerts=alerts,
            bucket_size=V,
            n_buckets=N,
            mean_vpin=mean_vpin,
            max_vpin=max_vpin,
            min_vpin=min_vpin,
            current_vpin=latest_vpin,
            current_toxicity_level=current_regime,
            summary_metrics=summary_metrics,
        )

    @staticmethod
    def generate_sample_trade_flow(
        n_trades: int = 12000,
        initial_price: float = 100.0,
        base_volume: float = 100.0,
        inject_flash_crash: bool = True,
        seed: int = 42,
    ) -> pd.DataFrame:
        np.random.seed(seed)

        start_date = pd.Timestamp("2024-05-06 09:30:00")
        time_deltas = np.random.exponential(scale=0.5, size=n_trades)
        timestamps = start_date + pd.to_timedelta(np.cumsum(time_deltas), unit="s")

        prices = np.zeros(n_trades)
        volumes = np.zeros(n_trades)
        sides = []

        curr_p = initial_price
        dt = 1.0 / (252 * 6.5 * 3600)
        volatility = 0.25

        crash_start = int(n_trades * 0.50)
        crash_end = int(n_trades * 0.70)

        for i in range(n_trades):
            if inject_flash_crash and crash_start <= i < crash_end:
                p_sell = 0.85
                is_sell = np.random.rand() < p_sell
                side = "sell" if is_sell else "buy"
                v = np.random.lognormal(mean=np.log(base_volume * 3.0), sigma=0.6)
                price_shock = -np.abs(np.random.normal(0.0008, 0.0015)) if is_sell else np.random.normal(0.0001, 0.0005)
                curr_p = max(10.0, curr_p * (1.0 + price_shock))
            elif inject_flash_crash and crash_end <= i < crash_end + int(n_trades * 0.10):
                p_buy = 0.75
                is_buy = np.random.rand() < p_buy
                side = "buy" if is_buy else "sell"
                v = np.random.lognormal(mean=np.log(base_volume * 1.5), sigma=0.5)
                price_shock = np.abs(np.random.normal(0.0005, 0.0010)) if is_buy else -np.abs(np.random.normal(0.0002, 0.0005))
                curr_p = curr_p * (1.0 + price_shock)
            else:
                side = "buy" if np.random.rand() > 0.50 else "sell"
                v = np.random.lognormal(mean=np.log(base_volume), sigma=0.4)
                ret = np.random.normal(0.0, volatility * np.sqrt(dt * 100))
                curr_p = max(10.0, curr_p * (1.0 + ret))

            prices[i] = round(curr_p, 2)
            volumes[i] = round(v, 0)
            sides.append(side)

        return pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        })
