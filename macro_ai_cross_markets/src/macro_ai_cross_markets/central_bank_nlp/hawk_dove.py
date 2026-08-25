"""Multilingual Central Bank LLM & Hawk/Dove Monetary Policy Indexer (Project 46).

Implements NLP stance extraction, topic decomposition, Taylor rule residualization,
and market impact predictive modeling for G10 & EM central banks.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import re
import math
import numpy as np
import pandas as pd
from ..data.loader import PolicyStatement


HAWKISH_TERMS_EN = {
    "raise", "raised", "raising", "hike", "hiked", "hiking", "increase", "increased",
    "tighten", "tightening", "tightened", "elevated", "restrictive", "firming",
    "overheat", "overheating", "pressures", "vigilance", "vigilant", "persevere",
    "perseverance", "persistence", "persistent", "committed", "withdrawal",
    "acceleration", "exceeded", "broaden", "broadened", "strengthen", "upside",
    "unanchored", "overheating", "frontload", "frontloads", "frontloading"
}

DOVISH_TERMS_EN = {
    "cut", "cuts", "cutting", "lower", "lowered", "lowering", "reduce", "reduced",
    "easing", "accommodative", "accommodation", "downside", "slowdown", "slowed",
    "slack", "moderate", "moderated", "moderating", "soft", "softened", "softening",
    "deteriorated", "subdued", "recession", "headwinds", "support", "supporting",
    "patience", "patient", "patiently", "flexibility", "flexibilization", "aid",
    "stimulate", "stimulus", "liquidity", "prudent", "relief", "cooling"
}

HAWKISH_TERMS_JA = {"利上げ", "引き上げ", "インフレ圧力", "物価上昇", "金融引き締め", "上限", "タカ派", "タカ的"}
DOVISH_TERMS_JA = {"金融緩和", "緩和", "マイナス金利", "買い入れ", "粘り強く", "ハト派", "ハト的", "景気減速"}

HAWKISH_TERMS_PT = {"elevar", "elevacao", "aumento", "alta", "aperto", "contracionista", "vigilancia", "pressao", "ancorar"}
DOVISH_TERMS_PT = {"reduzir", "reducao", "corte", "flexibilizacao", "afrouxamento", "estimulo", "desaceleracao", "acomodaticio"}

HAWKISH_TERMS_ES = {"incrementar", "incremento", "aumento", "alza", "restrictiva", "presiones", "inflacion", "alza"}
DOVISH_TERMS_ES = {"reducir", "reduccion", "recorte", "acomodaticia", "flexibilizacion", "desaceleracion", "apoyo"}

HAWKISH_TERMS_ZH = {"加息", "上调", "紧缩", "通胀压力", "升值", "过热"}
DOVISH_TERMS_ZH = {"降息", "下调", "充裕", "逆周期", "信贷支持", "实体经济", "降准", "宽松", "刺激"}


@dataclass
class StanceScoreResult:
    """Results of Hawk/Dove stance extraction for a policy statement."""
    central_bank: str
    date: str
    policy_rate: float
    rate_decision: str
    hawk_dove_score: float  # in [-1.0, +1.0], +1.0 = Max Hawk, -1.0 = Max Dove
    hawkish_count: int
    dovish_count: int
    total_tokens: int
    topic_weights: Dict[str, float]
    predicted_2y_yield_move_bps: float
    predicted_fx_move_pct: float
    language: str


@dataclass
class TaylorRuleResult:
    """Results of Taylor Rule benchmark and policy gap residualization."""
    country: str
    current_policy_rate: float
    taylor_rule_rate: float
    taylor_gap: float  # Policy Rate - Taylor Rate (bps / %)
    neutral_rate_r_star: float
    inflation_rate: float
    inflation_target: float
    output_gap_pct: float
    residual_hawk_dove_score: float


def clean_and_tokenize(text: str) -> List[str]:
    """Cleans text and extracts lowercase word tokens."""
    text_clean = re.sub(r'[^a-zA-Z0-9一-鿿぀-ゟ゠-ヿáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]', ' ', text.lower())
    tokens = [t.strip() for t in text_clean.split() if len(t.strip()) > 1]
    return tokens


class CentralBankStanceIndexer:
    """Natural Language Processing & LLM indexer for central bank policy stances."""

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.hawkish_lexicon_en = HAWKISH_TERMS_EN
        self.dovish_lexicon_en = DOVISH_TERMS_EN

    def score_statement(self, stmt: PolicyStatement) -> StanceScoreResult:
        """Scores a policy statement on the Hawk-Dove continuous spectrum [-1.0, +1.0]."""
        text_en = stmt.english_translation if stmt.english_translation else stmt.statement_text
        tokens = clean_and_tokenize(text_en)

        # Count hawkish and dovish signals
        n_hawk = sum(1 for t in tokens if t in self.hawkish_lexicon_en)
        n_dove = sum(1 for t in tokens if t in self.dovish_lexicon_en)

        # Also check native language terms if present
        if stmt.official_language == "ja":
            for w in HAWKISH_TERMS_JA:
                if w in stmt.statement_text:
                    n_hawk += 2
            for w in DOVISH_TERMS_JA:
                if w in stmt.statement_text:
                    n_dove += 2
        elif stmt.official_language == "pt":
            for w in HAWKISH_TERMS_PT:
                if w in stmt.statement_text.lower():
                    n_hawk += 1
            for w in DOVISH_TERMS_PT:
                if w in stmt.statement_text.lower():
                    n_dove += 1
        elif stmt.official_language == "es":
            for w in HAWKISH_TERMS_ES:
                if w in stmt.statement_text.lower():
                    n_hawk += 1
            for w in DOVISH_TERMS_ES:
                if w in stmt.statement_text.lower():
                    n_dove += 1
        elif stmt.official_language == "zh":
            for w in HAWKISH_TERMS_ZH:
                if w in stmt.statement_text:
                    n_hawk += 2
            for w in DOVISH_TERMS_ZH:
                if w in stmt.statement_text:
                    n_dove += 2

        # Rate decision bonus
        if stmt.rate_decision == "HIKE":
            n_hawk += 2
        elif stmt.rate_decision == "CUT":
            n_dove += 2

        total_signals = n_hawk + n_dove
        if total_signals == 0:
            score = 0.0
        else:
            score = (n_hawk - n_dove) / (total_signals + self.epsilon)

        score = float(np.clip(score, -1.0, +1.0))

        # Topic breakdown
        topics = {
            "Inflation": sum(1 for t in tokens if t in ["inflation", "prices", "cpi", "costs", "pressures"]),
            "Labor": sum(1 for t in tokens if t in ["employment", "unemployment", "jobs", "wages", "labor"]),
            "Growth": sum(1 for t in tokens if t in ["growth", "activity", "expansion", "demand", "recession"]),
            "Balance_Sheet": sum(1 for t in tokens if t in ["holdings", "securities", "liquidity", "ycc", "balance"]),
        }
        tot_topic = max(1, sum(topics.values()))
        topic_weights = {k: round(v / tot_topic, 3) for k, v in topics.items()}

        # Market predictive response (Empirical rule: +1.0 Hawk shock = +15bps on 2Y yield, +0.85% FX appreciation)
        pred_yield_bps = float(score * 18.5)
        pred_fx_pct = float(score * 0.95)

        return StanceScoreResult(
            central_bank=stmt.central_bank,
            date=stmt.date,
            policy_rate=stmt.policy_rate,
            rate_decision=stmt.rate_decision,
            hawk_dove_score=round(score, 4),
            hawkish_count=n_hawk,
            dovish_count=n_dove,
            total_tokens=len(tokens),
            topic_weights=topic_weights,
            predicted_2y_yield_move_bps=round(pred_yield_bps, 2),
            predicted_fx_move_pct=round(pred_fx_pct, 2),
            language=stmt.official_language,
        )

    def analyze_corpus(self, statements: List[PolicyStatement]) -> pd.DataFrame:
        """Scores an entire corpus of statements and returns a summary DataFrame."""
        results = [self.score_statement(s) for s in statements]
        rows = []
        for r in results:
            rows.append({
                "Central_Bank": r.central_bank,
                "Date": r.date,
                "Policy_Rate": r.policy_rate,
                "Decision": r.rate_decision,
                "Hawk_Dove_Score": r.hawk_dove_score,
                "Stance": "HAWKISH" if r.hawk_dove_score > 0.15 else ("DOVISH" if r.hawk_dove_score < -0.15 else "NEUTRAL"),
                "Pred_2Y_Move_bps": r.predicted_2y_yield_move_bps,
                "Pred_FX_Move_%": r.predicted_fx_move_pct,
                "Top_Topic": max(r.topic_weights.items(), key=lambda x: x[1])[0],
                "Language": r.language,
            })
        return pd.DataFrame(rows)


class TaylorRuleModel:
    """Taylor (1993) Monetary Policy Benchmark and Policy Gap Residualizer."""

    def __init__(
        self,
        alpha_inflation: float = 0.5,
        beta_output: float = 0.5,
    ):
        self.alpha_inflation = alpha_inflation
        self.beta_output = beta_output

    def calculate_taylor_rule(
        self,
        country: str,
        current_policy_rate: float,
        inflation_rate: float,
        inflation_target: float = 2.0,
        r_star: float = 0.5,
        output_gap_pct: float = 0.0,
        hawk_dove_nlp_score: float = 0.0,
    ) -> TaylorRuleResult:
        """Calculates theoretical Taylor Rule rate and computes residual NLP stance."""
        # Classic Taylor Rule: i* = r* + pi + alpha*(pi - pi*) + beta*(y - y*)
        taylor_rate = r_star + inflation_rate + self.alpha_inflation * (inflation_rate - inflation_target) + self.beta_output * output_gap_pct
        taylor_rate = float(np.clip(taylor_rate, -1.0, 60.0))

        taylor_gap = current_policy_rate - taylor_rate  # Positive = Policy tighter than Taylor Rule

        # Residualize NLP stance against economic Taylor gap
        # If policy is already tight (+gap), hawkish rhetoric is expected; residual captures genuine surprise
        beta_gap = 0.08
        residual_stance = hawk_dove_nlp_score - beta_gap * (taylor_gap / 2.0)
        residual_stance = float(np.clip(residual_stance, -1.0, +1.0))

        return TaylorRuleResult(
            country=country,
            current_policy_rate=current_policy_rate,
            taylor_rule_rate=round(taylor_rate, 2),
            taylor_gap=round(taylor_gap, 2),
            neutral_rate_r_star=r_star,
            inflation_rate=inflation_rate,
            inflation_target=inflation_target,
            output_gap_pct=output_gap_pct,
            residual_hawk_dove_score=round(residual_stance, 4),
        )
