"""
Central Bank Multilingual Hawk/Dove Monetary Policy NLP & Taylor Rule Residualization Engine.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Multilingual Domain Lexicons
HAWKISH_LEXICON = {
    "inflation", "overheating", "tightening", "rate hike", "restrictive", "upside risk",
    "wage pressure", "demand surge", "tapering", "excess demand", "above target", "elevated",
    "curb inflation", "persistent", "vigilant", "unanchored", "hawkish", "subir tasas",
    "inflación elevada", "presiones inflacionarias", "aperto monetário", "aperto",
    "alta de juros", "risco fiscal", "利上げ", "インフレ懸念", "金融引き締め", "物価上昇"
}

DOVISH_LEXICON = {
    "easing", "rate cut", "accommodative", "headwinds", "downside risk", "slack",
    "disinflation", "slowdown", "recession", "below target", "support growth", "transitory",
    "liquidity injection", "stimulus", "dovish", "patience", "labor market weakness",
    "bajar tasas", "desaceleración", "estímulo", "corte de juros", "afrouxamento",
    "desinflação", "crescimento fraco", "利下げ", "金融緩和", "景気減速", "物価目標未達"
}

TOPIC_KEYWORDS = {
    "Inflation": ["inflation", "cpi", "pce", "prices", "cost", "wage", "inflación", "物価", "インフレ"],
    "Labor": ["employment", "unemployment", "job", "payroll", "labor", "empleo", "desempleo", "雇用", "失業"],
    "Growth": ["gdp", "growth", "output", "demand", "expansion", "recession", "pib", "crecimiento", "景気", "成長"],
    "Balance_Sheet": ["balance sheet", "qt", "qe", "asset purchase", "liquidity", "liquidité", "資産買い入れ", "流動性"],
}


@dataclass
class StanceResult:
    """Output container for central bank policy statement NLP analysis."""
    central_bank: str
    date: str
    hawk_dove_score: float
    stance_category: str
    hawkish_count: int
    dovish_count: int
    total_tokens: int
    topic_distribution: Dict[str, float]
    taylor_rule_residual: Optional[float] = None
    expected_yield_impact_bps: float = 0.0


class CentralBankStanceIndexer:
    """
    NLP engine for parsing central bank communications and indexing policy stances.
    Evaluates net hawkishness, topic focus, and Taylor Rule policy shocks.
    """

    def __init__(self, smoothing_epsilon: float = 1.0):
        self.smoothing_epsilon = smoothing_epsilon

    def clean_text(self, text: str) -> str:
        """Normalize text and remove special characters while preserving multilingual characters."""
        return text.lower().strip()

    def score_statement(
        self,
        text: str,
        central_bank: str = "FED",
        date: str = "2026-01-01",
        actual_policy_rate: Optional[float] = None,
        cpi_inflation: Optional[float] = None,
        inflation_target: float = 2.0,
        gdp_growth: Optional[float] = None,
        potential_gdp: float = 2.0,
    ) -> StanceResult:
        """Score text for Hawk/Dove stance and evaluate Taylor Rule gap."""
        cleaned = self.clean_text(text)
        words = re.findall(r"\w+", cleaned)
        total_tokens = len(words)

        hawk_count = 0
        for kw in HAWKISH_LEXICON:
            hawk_count += len(re.findall(r"\b" + re.escape(kw) + r"\b", cleaned))

        dove_count = 0
        for kw in DOVISH_LEXICON:
            dove_count += len(re.findall(r"\b" + re.escape(kw) + r"\b", cleaned))

        # Normalized Hawk-Dove Score [-1.0, +1.0]
        score = (hawk_count - dove_count) / (hawk_count + dove_count + self.smoothing_epsilon)
        score = float(np.clip(score, -1.0, 1.0))

        if score > 0.15:
            stance = "HAWKISH"
        elif score < -0.15:
            stance = "DOVISH"
        else:
            stance = "NEUTRAL"

        # Topic segmentation
        topic_counts: Dict[str, float] = {}
        for topic, kws in TOPIC_KEYWORDS.items():
            t_count = 0
            for kw in kws:
                t_count += len(re.findall(r"\b" + re.escape(kw) + r"\b", cleaned))
            topic_counts[topic] = float(t_count)
        tot_topic = sum(topic_counts.values())
        if tot_topic > 0:
            topic_dist = {k: v / tot_topic for k, v in topic_counts.items()}
        else:
            topic_dist = {k: 1.0 / len(topic_counts) for k in topic_counts}

        # Taylor Rule Gap: i_target = r_star + pi + 0.5*(pi - pi_star) + 0.5*(y - y_star)
        taylor_residual = None
        yield_impact_bps = score * 15.0  # approximate 15 bps shift per unit score
        if actual_policy_rate is not None and cpi_inflation is not None and gdp_growth is not None:
            r_star = 0.50 if central_bank in ["BOJ", "ECB"] else 1.50
            taylor_target = r_star + cpi_inflation + 0.5 * (cpi_inflation - inflation_target) + 0.5 * (gdp_growth - potential_gdp)
            taylor_gap = actual_policy_rate - taylor_target
            # Combined forward guidance tone surprise + macro rule residual
            taylor_residual = float(0.6 * taylor_gap + 0.4 * (score * 2.0))

        return StanceResult(
            central_bank=central_bank,
            date=date,
            hawk_dove_score=score,
            stance_category=stance,
            hawkish_count=hawk_count,
            dovish_count=dove_count,
            total_tokens=total_tokens,
            topic_distribution=topic_dist,
            taylor_rule_residual=taylor_residual,
            expected_yield_impact_bps=float(yield_impact_bps),
        )

    def analyze_corpus(self, statement_records: List[Dict[str, Union[str, float]]]) -> pd.DataFrame:
        """Batch process central bank statements into structured DataFrame."""
        records = []
        for rec in statement_records:
            res = self.score_statement(
                text=str(rec["text"]),
                central_bank=str(rec.get("central_bank", "FED")),
                date=str(rec.get("date", "2026-01-01")),
                actual_policy_rate=float(rec["policy_rate"]) if "policy_rate" in rec else None,
                cpi_inflation=float(rec["cpi_inflation"]) if "cpi_inflation" in rec else None,
                inflation_target=float(rec.get("inflation_target", 2.0)),
                gdp_growth=float(rec["gdp_growth"]) if "gdp_growth" in rec else None,
            )
            records.append({
                "Central_Bank": res.central_bank,
                "Date": res.date,
                "Hawk_Dove_Score": res.hawk_dove_score,
                "Stance": res.stance_category,
                "Hawkish_Tokens": res.hawkish_count,
                "Dovish_Tokens": res.dovish_count,
                "Top_Topic": max(res.topic_distribution.items(), key=lambda x: x[1])[0],
                "Taylor_Residual": res.taylor_rule_residual,
                "Expected_Yield_Impact_bps": res.expected_yield_impact_bps,
            })
        return pd.DataFrame(records)
