"""Social Media, News & Crypto Fear/Greed LLM Market Sentiment Engine (Project 48).

Implements multi-source financial and crypto sentiment ingestion, Aspect-Based
Sentiment Analysis (ABSA), Crypto Fear & Greed Index reconstruction from 6 components,
lead-lag cross-correlation diagnostics, and systematic sentiment trading strategy backtesting.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import math
import numpy as np
import pandas as pd
from scipy import stats


class SentimentAspect(str, Enum):
    """Aspect categories for financial sentiment analysis."""
    MACRO = "macro"
    CENTRAL_BANK = "central_bank"
    CRYPTO = "crypto"
    REGULATORY = "regulatory"
    GENERAL = "general"


class FearGreedRegime(str, Enum):
    """Categorical classification of Fear & Greed Index."""
    EXTREME_FEAR = "EXTREME_FEAR"      # 0 - 24
    FEAR = "FEAR"                      # 25 - 44
    NEUTRAL = "NEUTRAL"                # 45 - 55
    GREED = "GREED"                    # 56 - 75
    EXTREME_GREED = "EXTREME_GREED"    # 76 - 100


@dataclass
class SentimentRecord:
    """Represents a scored text snippet or article."""
    text: str
    source: str                          # "news", "twitter", "reddit", "macro"
    timestamp: pd.Timestamp
    aspect: SentimentAspect
    polarity: float                      # Range [-1.0, 1.0]
    confidence: float                    # Range [0.0, 1.0]
    positive_score: float
    negative_score: float
    subjectivity: float = 0.5
    raw_tokens: List[str] = field(default_factory=list)


@dataclass
class FearGreedComponents:
    """Decomposed components of the reconstructed Crypto Fear & Greed Index."""
    timestamp: pd.Timestamp
    volatility_score: float              # 25% weight (lower vol -> higher greed)
    momentum_volume_score: float         # 25% weight (higher upward momentum -> higher greed)
    social_sentiment_score: float        # 15% weight (bullish social tone -> higher greed)
    dominance_score: float               # 10% weight (lower BTC dominance -> altcoin greed)
    search_trends_score: float           # 10% weight (search query volume)
    whale_velocity_score: float          # 15% weight (whale accumulation vs exchange dumps)
    composite_index: float               # Range [0.0, 100.0]
    regime: FearGreedRegime


@dataclass
class LeadLagResult:
    """Results of lead-lag cross-correlation analysis."""
    lags: np.ndarray
    pearson_correlations: np.ndarray
    spearman_correlations: np.ndarray
    p_values: np.ndarray
    peak_lag: int
    peak_correlation: float
    is_sentiment_leading: bool
    summary_dataframe: pd.DataFrame


@dataclass
class SentimentStrategyResult:
    """Backtest results for a systematic sentiment-driven trading strategy."""
    dates: pd.DatetimeIndex
    cumulative_returns: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    signals: pd.DataFrame
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    turnover_annual: float
    metrics_table: pd.DataFrame


# Financial & Crypto Domain Lexicon
FINANCIAL_LEXICON = {
    # Positive financial & crypto terms
    "bullish": 2.5, "rally": 2.2, "surge": 2.0, "breakout": 2.2, "growth": 1.8,
    "profit": 1.9, "outperform": 2.0, "gain": 1.5, "upgrade": 1.7, "accumulation": 1.8,
    "inflow": 1.6, "adoption": 1.8, "partnership": 1.5, "approval": 2.4, "dovish": 1.9,
    "stimulus": 2.0, "expansion": 1.7, "resilient": 1.6, "recovery": 1.7, "ath": 2.5,
    "all-time-high": 2.5, "institutional": 1.5, "inflows": 1.6, "hodl": 1.2, "moon": 1.5,
    "gem": 1.3, "staking": 1.1, "dividend": 1.4, "record": 1.6, "liquidity": 1.2,
    "optimistic": 1.6, "momentum": 1.5, "beat": 1.7, "strong": 1.4, "tailwind": 1.6,

    # Negative financial & crypto terms
    "bearish": -2.5, "crash": -2.8, "dump": -2.4, "plunge": -2.3, "slump": -2.0,
    "collapse": -2.8, "loss": -1.8, "underperform": -2.0, "downgrade": -1.8, "selloff": -2.2,
    "outflow": -1.6, "outflows": -1.6, "liquidation": -2.3, "liquidations": -2.3,
    "recession": -2.5, "inflation": -1.5, "hawkish": -1.8, "tightening": -1.7,
    "rate-hike": -1.9, "hike": -1.6, "lawsuit": -2.2, "sec": -1.4, "cftc": -1.2,
    "subpoena": -2.4, "fraud": -3.0, "insolvency": -3.0, "bankruptcy": -3.0, "hack": -2.8,
    "exploit": -2.7, "rug": -2.8, "rugpull": -3.0, "fud": -1.5, "panic": -2.5,
    "default": -2.6, "ban": -2.4, "prohibition": -2.2, "investigation": -2.0,
    "headwind": -1.6, "weak": -1.4, "miss": -1.6, "decline": -1.5, "drop": -1.4,
}

ASPECT_KEYWORDS = {
    SentimentAspect.MACRO: {
        "inflation", "gdp", "recession", "employment", "unemployment", "cpi", "pce",
        "yield", "treasury", "bonds", "macro", "growth", "slowdown", "soft-landing",
        "stagflation", "deficit", "commodities", "oil", "dollar", "dxy"
    },
    SentimentAspect.CENTRAL_BANK: {
        "fed", "fomc", "powell", "ecb", "lagarde", "rate", "hike", "cut", "dovish",
        "hawkish", "qt", "qe", "balance-sheet", "tightening", "easing", "interest-rate",
        "basis-points", "bps", "terminal-rate", "pivot"
    },
    SentimentAspect.CRYPTO: {
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto", "defi", "nft",
        "halving", "miner", "hashrate", "mempool", "wallet", "onchain", "whale",
        "altcoin", "layer1", "layer2", "staking", "validator", "liquidity-pool"
    },
    SentimentAspect.REGULATORY: {
        "sec", "cftc", "gensler", "lawsuit", "subpoena", "court", "judge", "etf",
        "approval", "denial", "compliance", "aml", "kyc", "sanction", "ban", "legal",
        "investigation", "license", "enforcement", "finra", "doj"
    }
}

NEGATION_TERMS = {
    "not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely",
    "cannot", "doesn't", "don't", "didn't", "won't", "wouldn't", "isn't", "aren't"
}

INTENSIFIER_TERMS = {
    "very": 1.3, "extremely": 1.6, "massively": 1.5, "hugely": 1.4, "substantially": 1.3,
    "critically": 1.4, "highly": 1.3, "slightly": 0.7, "somewhat": 0.8, "moderately": 0.9
}


class MultiSourceSentimentEngine:
    """Engine for multi-source financial and crypto sentiment analysis and alpha modeling."""

    def __init__(
        self,
        custom_lexicon: Optional[Dict[str, float]] = None,
        fear_greed_weights: Optional[Dict[str, float]] = None,
    ):
        self.lexicon = FINANCIAL_LEXICON.copy()
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)

        # Default standard Fear & Greed weights
        self.fg_weights = {
            "volatility": 0.25,
            "momentum_volume": 0.25,
            "social_sentiment": 0.15,
            "whale_velocity": 0.15,
            "dominance": 0.10,
            "search_trends": 0.10,
        }
        if fear_greed_weights:
            self.fg_weights.update(fear_greed_weights)
            # Normalize weights to sum to 1.0
            total_w = sum(self.fg_weights.values())
            self.fg_weights = {k: v / total_w for k, v in self.fg_weights.items()}

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into lowercase words removing punctuation."""
        clean_text = re.sub(r"[^a-zA-Z0-9\s-]", " ", text.lower())
        tokens = [t.strip() for t in clean_text.split() if len(t.strip()) > 1]
        return tokens

    def classify_aspect(self, tokens: List[str]) -> SentimentAspect:
        """Classifies text into primary financial aspect using keyword overlap."""
        aspect_counts = {aspect: 0 for aspect in SentimentAspect if aspect != SentimentAspect.GENERAL}
        
        for token in tokens:
            for aspect, kws in ASPECT_KEYWORDS.items():
                if token in kws:
                    aspect_counts[aspect] += 1

        max_aspect = max(aspect_counts, key=aspect_counts.get)
        if aspect_counts[max_aspect] > 0:
            return max_aspect
        return SentimentAspect.GENERAL

    def analyze_text(
        self,
        text: str,
        source: str = "news",
        timestamp: Optional[pd.Timestamp] = None,
        target_aspect: Optional[SentimentAspect] = None,
    ) -> SentimentRecord:
        """Analyzes sentiment polarity of text with negation and intensifier handling."""
        tokens = self.tokenize(text)
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        aspect = target_aspect if target_aspect else self.classify_aspect(tokens)

        pos_score = 0.0
        neg_score = 0.0
        n_scored = 0

        negated = False
        intensifier = 1.0

        for i, token in enumerate(tokens):
            if token in NEGATION_TERMS:
                negated = True
                continue
            if token in INTENSIFIER_TERMS:
                intensifier = INTENSIFIER_TERMS[token]
                continue

            if token in self.lexicon:
                base_weight = self.lexicon[token] * intensifier
                if negated:
                    base_weight = -0.75 * base_weight
                    negated = False
                intensifier = 1.0

                if base_weight > 0:
                    pos_score += base_weight
                else:
                    neg_score += abs(base_weight)
                n_scored += 1

        total_magnitude = pos_score + neg_score
        if total_magnitude > 0:
            raw_diff = pos_score - neg_score
            polarity = float(np.clip(raw_diff / (total_magnitude + 1.0), -1.0, 1.0))
            confidence = float(np.clip(n_scored / (len(tokens) + 1e-5) * 5.0, 0.1, 1.0))
        else:
            polarity = 0.0
            confidence = 0.1

        return SentimentRecord(
            text=text,
            source=source,
            timestamp=timestamp,
            aspect=aspect,
            polarity=polarity,
            confidence=confidence,
            positive_score=pos_score,
            negative_score=neg_score,
            raw_tokens=tokens,
        )

    def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Ingests a list of document dicts with keys (text, source, timestamp)."""
        records = []
        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source", "news")
            ts = pd.to_datetime(doc.get("timestamp", pd.Timestamp.now()))
            aspect_override = doc.get("aspect")
            
            res = self.analyze_text(text=text, source=source, timestamp=ts, target_aspect=aspect_override)
            records.append({
                "Timestamp": res.timestamp,
                "Source": res.source,
                "Aspect": res.aspect.value,
                "Polarity": res.polarity,
                "Confidence": res.confidence,
                "Positive_Score": res.positive_score,
                "Negative_Score": res.negative_score,
                "Text": res.text,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("Timestamp").reset_index(drop=True)
        return df

    def compute_fear_greed_index(
        self,
        volatility_series: pd.Series,
        price_series: pd.Series,
        volume_series: pd.Series,
        social_sentiment_series: Optional[pd.Series] = None,
        btc_dominance_series: Optional[pd.Series] = None,
        search_trends_series: Optional[pd.Series] = None,
        whale_net_inflow_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Reconstructs the multi-component Crypto Fear & Greed Index time series (0 to 100)."""
        idx = price_series.index
        n_bars = len(price_series)

        # 1. Volatility Component (25%): Lower realized volatility relative to 90d rolling -> Greed (100)
        ret = price_series.pct_change().fillna(0.0)
        rolling_vol_30 = ret.rolling(30, min_periods=5).std() * np.sqrt(365)
        rolling_vol_90 = ret.rolling(90, min_periods=10).std() * np.sqrt(365)
        vol_ratio = (rolling_vol_30 / (rolling_vol_90 + 1e-6)).fillna(1.0)
        vol_score = np.clip(100.0 - (vol_ratio - 0.5) * 50.0, 0.0, 100.0)

        # 2. Market Momentum / Volume Component (25%): Price relative to 30d/90d MA & Volume surge
        ma_30 = price_series.rolling(30, min_periods=5).mean()
        ma_90 = price_series.rolling(90, min_periods=10).mean()
        mom_ratio = (price_series / (ma_90 + 1e-6)).fillna(1.0)
        vol_growth = (volume_series / (volume_series.rolling(30, min_periods=5).mean() + 1e-6)).fillna(1.0)
        mom_score = np.clip((mom_ratio - 0.7) / 0.6 * 60.0 + (vol_growth - 0.5) * 20.0, 0.0, 100.0)

        # 3. Social Media Sentiment (15%): Default to polarity scaled to [0, 100]
        if social_sentiment_series is not None:
            soc_score = np.clip((social_sentiment_series + 1.0) * 50.0, 0.0, 100.0)
        else:
            soc_score = np.clip(50.0 + ret.rolling(7).mean() * 1000.0, 10.0, 90.0)

        # 4. Bitcoin Dominance (10%): Rising BTC dominance implies altcoin fear
        if btc_dominance_series is not None:
            dom_change = btc_dominance_series.pct_change(30).fillna(0.0)
            dom_score = np.clip(50.0 - dom_change * 300.0, 0.0, 100.0)
        else:
            dom_score = pd.Series(50.0, index=idx)

        # 5. Search Trends (10%): Google Trends volume growth
        if search_trends_series is not None:
            search_score = np.clip(search_trends_series, 0.0, 100.0)
        else:
            search_score = np.clip(50.0 + ret.rolling(14).mean() * 500.0, 15.0, 85.0)

        # 6. Whale / On-Chain Velocity (15%): Whale net accumulation vs exchange inflow dumps
        if whale_net_inflow_series is not None:
            whale_score = np.clip(50.0 + whale_net_inflow_series * 50.0, 0.0, 100.0)
        else:
            whale_score = np.clip(50.0 + ret.rolling(5).mean() * 400.0, 10.0, 90.0)

        # Composite Weighted Index Calculation
        w = self.fg_weights
        composite = (
            w["volatility"] * vol_score +
            w["momentum_volume"] * mom_score +
            w["social_sentiment"] * soc_score +
            w["dominance"] * dom_score +
            w["search_trends"] * search_score +
            w["whale_velocity"] * whale_score
        )
        composite = pd.Series(composite, index=idx).fillna(50.0)
        composite = np.clip(composite, 0.0, 100.0)
        composite = pd.Series(np.clip(composite, 0.0, 100.0), index=idx)

        # Classify Regimes
        regimes = []
        for val in composite:
            if val < 25.0:
                regimes.append(FearGreedRegime.EXTREME_FEAR.value)
            elif val < 45.0:
                regimes.append(FearGreedRegime.FEAR.value)
            elif val <= 55.0:
                regimes.append(FearGreedRegime.NEUTRAL.value)
            elif val <= 75.0:
                regimes.append(FearGreedRegime.GREED.value)
            else:
                regimes.append(FearGreedRegime.EXTREME_GREED.value)

        result_df = pd.DataFrame({
            "Composite_FGI": np.round(composite, 1),
            "Regime": regimes,
            "Volatility_Score": np.round(vol_score, 1),
            "Momentum_Score": np.round(mom_score, 1),
            "Social_Score": np.round(soc_score, 1),
            "Dominance_Score": np.round(dom_score, 1),
            "Search_Score": np.round(search_score, 1),
            "Whale_Score": np.round(whale_score, 1),
        }, index=idx)

        return result_df

    def compute_lead_lag_correlation(
        self,
        sentiment_series: pd.Series,
        return_series: pd.Series,
        max_lag: int = 10,
    ) -> LeadLagResult:
        """Computes cross-correlation between sentiment and forward asset returns across lags."""
        valid_idx = sentiment_series.dropna().index.intersection(return_series.dropna().index)
        s = sentiment_series.loc[valid_idx]
        r = return_series.loc[valid_idx]

        lags = np.arange(-max_lag, max_lag + 1)
        pearson_cors = []
        spearman_cors = []
        p_vals = []

        n = len(valid_idx)
        for lag in lags:
            if lag < 0:
                shifted_s = s.iloc[-lag:]
                shifted_r = r.iloc[:lag]
            elif lag > 0:
                shifted_s = s.iloc[:-lag]
                shifted_r = r.iloc[lag:]
            else:
                shifted_s = s
                shifted_r = r

            if len(shifted_s) > 10:
                p_r, p_val = stats.pearsonr(shifted_s, shifted_r)
                s_r, _ = stats.spearmanr(shifted_s, shifted_r)
            else:
                p_r, p_val, s_r = 0.0, 1.0, 0.0

            pearson_cors.append(p_r)
            spearman_cors.append(s_r)
            p_vals.append(p_val)

        pearson_cors = np.array(pearson_cors)
        spearman_cors = np.array(spearman_cors)
        p_vals = np.array(p_vals)

        abs_cors = np.abs(pearson_cors)
        peak_idx = int(np.argmax(abs_cors))
        peak_lag = int(lags[peak_idx])
        peak_corr = float(pearson_cors[peak_idx])
        is_leading = peak_lag > 0

        summary_df = pd.DataFrame({
            "Lag_Days": lags,
            "Pearson_Correlation": np.round(pearson_cors, 4),
            "Spearman_Rank_Correlation": np.round(spearman_cors, 4),
            "p_Value": p_vals,
            "Significant_5pct": p_vals < 0.05,
        })

        return LeadLagResult(
            lags=lags,
            pearson_correlations=pearson_cors,
            spearman_correlations=spearman_cors,
            p_values=p_vals,
            peak_lag=peak_lag,
            peak_correlation=peak_corr,
            is_sentiment_leading=is_leading,
            summary_dataframe=summary_df,
        )

    def backtest_sentiment_strategy(
        self,
        prices_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        threshold_long: float = 60.0,
        threshold_short: float = 40.0,
        rebalance_freq: int = 1,
        transaction_cost_bps: float = 5.0,
        risk_free_rate: float = 0.02,
    ) -> SentimentStrategyResult:
        """Backtests systematic sentiment-driven cross-asset or single-asset trading strategy."""
        if isinstance(prices_df, pd.Series):
            prices_df = prices_df.to_frame(name="Asset")
        if isinstance(sentiment_df, pd.Series):
            sentiment_df = sentiment_df.to_frame(name="Asset")

        common_idx = prices_df.index.intersection(sentiment_df.index)
        prices = prices_df.loc[common_idx]
        sent = sentiment_df.loc[common_idx]

        returns = prices.pct_change().fillna(0.0)
        n_days, n_assets = prices.shape

        signals = pd.DataFrame(0.0, index=common_idx, columns=prices.columns)
        for col in prices.columns:
            if col in sent.columns:
                s_col = sent[col]
            else:
                s_col = sent.iloc[:, 0]

            signals.loc[s_col >= threshold_long, col] = 1.0
            signals.loc[s_col <= threshold_short, col] = -1.0

        weights = pd.DataFrame(0.0, index=common_idx, columns=prices.columns)
        for t in range(0, n_days, rebalance_freq):
            sig_row = signals.iloc[t]
            n_long = (sig_row > 0).sum()
            n_short = (sig_row < 0).sum()

            w_row = pd.Series(0.0, index=prices.columns)
            if n_long > 0:
                w_row[sig_row > 0] = 0.5 / n_long if n_short > 0 else 1.0 / n_long
            if n_short > 0:
                w_row[sig_row < 0] = -0.5 / n_short if n_long > 0 else -1.0 / n_short

            end_t = min(t + rebalance_freq, n_days)
            for step in range(t, end_t):
                weights.iloc[step] = w_row

        lagged_weights = weights.shift(1).fillna(0.0)
        turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
        cost_drag = turnover * (transaction_cost_bps / 10000.0)

        gross_daily_returns = (lagged_weights * returns).sum(axis=1)
        net_daily_returns = gross_daily_returns - cost_drag

        cumulative_equity = (1.0 + net_daily_returns).cumprod()

        n_years = max(1.0 / 252.0, len(net_daily_returns) / 252.0)
        cagr = float((cumulative_equity.iloc[-1]) ** (1.0 / n_years) - 1.0)
        ann_vol = float(net_daily_returns.std() * np.sqrt(252))

        rf_daily = risk_free_rate / 252.0
        excess_ret = net_daily_returns - rf_daily
        sharpe = float(np.sqrt(252) * excess_ret.mean() / (net_daily_returns.std() + 1e-6))

        downside_ret = net_daily_returns[net_daily_returns < 0]
        downside_vol = float(downside_ret.std() * np.sqrt(252)) if len(downside_ret) > 0 else ann_vol
        sortino = float(np.sqrt(252) * excess_ret.mean() / (downside_vol + 1e-6))

        peaks = cumulative_equity.cummax()
        drawdowns = (cumulative_equity - peaks) / peaks
        max_dd = float(drawdowns.min())
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0

        win_rate = float((net_daily_returns > 0).sum() / (len(net_daily_returns) + 1e-6))
        pos_sum = net_daily_returns[net_daily_returns > 0].sum()
        neg_sum = abs(net_daily_returns[net_daily_returns < 0].sum())
        profit_factor = float(pos_sum / (neg_sum + 1e-6))
        turnover_ann = float(turnover.mean() * 252.0)

        metrics_df = pd.DataFrame({
            "Metric": [
                "CAGR (Annual Return)",
                "Annualized Volatility",
                "Sharpe Ratio (Rf=2%)",
                "Sortino Ratio",
                "Calmar Ratio",
                "Maximum Drawdown",
                "Daily Win Rate",
                "Profit Factor",
                "Annualized Turnover",
                "Annual Cost Drag (bps)"
            ],
            "Value": [
                f"{cagr:+.2%}",
                f"{ann_vol:.2%}",
                f"{sharpe:.2f}",
                f"{sortino:.2f}",
                f"{calmar:.2f}",
                f"{max_dd:.2%}",
                f"{win_rate:.1%}",
                f"{profit_factor:.2f}",
                f"{turnover_ann:.2f}",
                f"{turnover_ann * transaction_cost_bps:.1f} bps"
            ]
        })

        return SentimentStrategyResult(
            dates=common_idx,
            cumulative_returns=cumulative_equity,
            daily_returns=net_daily_returns,
            positions=lagged_weights,
            signals=signals,
            cagr=cagr,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            turnover_annual=turnover_ann,
            metrics_table=metrics_df,
        )
