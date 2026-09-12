"""
Multi-Source News, Social Media & Reconstructed Crypto Fear & Greed Index.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SentimentIndexResult:
    composite_index: pd.Series
    volatility_score: pd.Series
    momentum_score: pd.Series
    social_score: pd.Series
    whale_velocity_score: pd.Series
    regimes: pd.Series

    def summary(self) -> str:
        latest = self.composite_index.iloc[-1]
        regime = self.regimes.iloc[-1]
        return f"""Fear & Greed Index Snapshot:
  - Current FGI Level:               {latest:.1f} / 100
  - Market Sentiment State:          {regime}
  - 30-Day Average Sentiment:        {self.composite_index.tail(30).mean():.1f}"""


class MultiSourceSentimentEngine:
    """Reconstructs the institutional 6-component Fear & Greed Index and sentiment alpha signals."""

    def __init__(self):
        pass

    def compute_fear_greed_index(
        self,
        price_series: pd.Series,
        volume_series: pd.Series,
        volatility_series: Optional[pd.Series] = None,
        social_sentiment_series: Optional[pd.Series] = None,
        whale_inflow_series: Optional[pd.Series] = None,
    ) -> SentimentIndexResult:
        df = pd.DataFrame(index=price_series.index)

        ret = price_series.pct_change()
        if volatility_series is None:
            vol = ret.rolling(21).std() * np.sqrt(252)
        else:
            vol = volatility_series
        vol_pct = 100.0 - vol.rank(pct=True) * 100.0
        df["vol_score"] = vol_pct.fillna(50.0)

        mom = price_series.pct_change(30)
        mom_pct = mom.rank(pct=True) * 100.0
        df["mom_score"] = mom_pct.fillna(50.0)

        if social_sentiment_series is None:
            social = (ret.rolling(5).mean() / (ret.rolling(21).std() + 1e-6)).clip(-3, 3)
            social_pct = (social + 3.0) / 6.0 * 100.0
        else:
            social_pct = (social_sentiment_series.clip(-1, 1) + 1.0) * 50.0
        df["social_score"] = social_pct.fillna(50.0)

        sma50 = price_series.rolling(50).mean()
        trend_score = ((price_series / sma50 - 1.0).clip(-0.2, 0.2) + 0.2) / 0.4 * 100.0
        df["dom_score"] = trend_score.fillna(50.0)

        vol_growth = volume_series.rolling(7).mean() / volume_series.rolling(30).mean()
        search_score = ((vol_growth - 1.0).clip(-0.5, 0.5) + 0.5) * 100.0
        df["search_score"] = search_score.fillna(50.0)

        if whale_inflow_series is None:
            whale_score = df["mom_score"]
        else:
            whale_score = 100.0 - whale_inflow_series.rank(pct=True) * 100.0
        df["whale_score"] = whale_score.fillna(50.0)

        fgi = (
            0.25 * df["vol_score"]
            + 0.25 * df["mom_score"]
            + 0.15 * df["social_score"]
            + 0.10 * df["dom_score"]
            + 0.10 * df["search_score"]
            + 0.15 * df["whale_score"]
        )

        regimes = pd.Series("NEUTRAL", index=fgi.index)
        regimes[fgi <= 25.0] = "EXTREME_FEAR"
        regimes[(fgi > 25.0) & (fgi <= 45.0)] = "FEAR"
        regimes[(fgi >= 55.0) & (fgi < 75.0)] = "GREED"
        regimes[fgi >= 75.0] = "EXTREME_GREED"

        return SentimentIndexResult(
            composite_index=fgi,
            volatility_score=df["vol_score"],
            momentum_score=df["mom_score"],
            social_score=df["social_score"],
            whale_velocity_score=df["whale_score"],
            regimes=regimes,
        )
