"""
Financial LLM: SEC 10-K Semantic Drift & The "Lazy Prices" Alpha Anomaly.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd


class SimpleTfidfVectorizer:
    """Lightweight pure Python/NumPy TF-IDF vectorizer with sublinear TF."""

    def __init__(self, stop_words: Optional[Set[str]] = None):
        self.stop_words = stop_words or {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
            "by", "of", "from", "as", "is", "was", "are", "were", "be", "been", "that",
            "this", "which", "it", "our", "we", "have", "has", "had", "will", "would"
        }
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    def tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return [w for w in words if w not in self.stop_words]

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        tokenized = [self.tokenize(doc) for doc in corpus]
        unique_words = sorted(list(set(w for doc in tokenized for w in doc)))
        self.vocab = {w: i for i, w in enumerate(unique_words)}
        
        n_docs = len(corpus)
        n_vocab = len(unique_words)
        if n_vocab == 0:
            return np.zeros((n_docs, 1))

        # DF
        df = np.zeros(n_vocab)
        for doc in tokenized:
            doc_set = set(doc)
            for w in doc_set:
                if w in self.vocab:
                    df[self.vocab[w]] += 1
        
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        # TF-IDF
        matrix = np.zeros((n_docs, n_vocab))
        for i, doc in enumerate(tokenized):
            counts: Dict[str, int] = {}
            for w in doc:
                counts[w] = counts.get(w, 0) + 1
            for w, count in counts.items():
                if w in self.vocab:
                    j = self.vocab[w]
                    tf = 1.0 + np.log(count)
                    matrix[i, j] = tf * self.idf[j]
            # L2 normalize
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        return matrix


@dataclass
class FilingDriftReport:
    ticker: str
    year: int
    cosine_drift_total: float
    cosine_drift_mda: float
    cosine_drift_risk: float
    jaccard_distance: float
    sentiment_score: float
    category: str


class SemanticDriftEngine:
    """Extracts year-over-year textual modifications in SEC 10-K disclosures."""

    def __init__(self, high_drift_threshold: float = 0.15, lazy_threshold: float = 0.04):
        self.high_drift_threshold = high_drift_threshold
        self.lazy_threshold = lazy_threshold

    def compute_cosine_drift(self, text_prior: str, text_current: str) -> float:
        if not text_prior.strip() or not text_current.strip():
            return 0.0
        vec = SimpleTfidfVectorizer()
        matrix = vec.fit_transform([text_prior, text_current])
        v1, v2 = matrix[0], matrix[1]
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_sim = dot / (norm1 * norm2)
        return float(np.clip(1.0 - cos_sim, 0.0, 2.0))

    def compute_jaccard_distance(self, text_prior: str, text_current: str) -> float:
        set1 = set(text_prior.lower().split())
        set2 = set(text_current.lower().split())
        union = set1.union(set2)
        if not union:
            return 0.0
        return float(1.0 - (len(set1.intersection(set2)) / len(union)))

    def analyze_filing_pair(self, ticker: str, year: int, text_prior: Dict[str, str], text_curr: Dict[str, str]) -> FilingDriftReport:
        drift_mda = self.compute_cosine_drift(text_prior.get("mda", ""), text_curr.get("mda", ""))
        drift_risk = self.compute_cosine_drift(text_prior.get("risk_factors", ""), text_curr.get("risk_factors", ""))
        
        full_prior = text_prior.get("mda", "") + " " + text_prior.get("risk_factors", "")
        full_curr = text_curr.get("mda", "") + " " + text_curr.get("risk_factors", "")
        drift_total = self.compute_cosine_drift(full_prior, full_curr)
        jaccard = self.compute_jaccard_distance(full_prior, full_curr)

        neg_words = {"litigation", "investigation", "loss", "decline", "risk", "uncertainty", "impairment", "breach"}
        pos_words = {"growth", "profit", "expansion", "record", "innovative", "gain", "efficiency", "strength"}
        tokens = re.findall(r"\b[a-zA-Z]+\b", full_curr.lower())
        n_neg = sum(1 for w in tokens if w in neg_words)
        n_pos = sum(1 for w in tokens if w in pos_words)
        sent_score = float((n_pos - n_neg) / max(n_pos + n_neg, 1))

        if drift_total >= self.high_drift_threshold:
            category = "HIGH_DRIFT"
        elif drift_total <= self.lazy_threshold:
            category = "LAZY_DISCLOSURE"
        else:
            category = "MODERATE_DRIFT"

        return FilingDriftReport(
            ticker=ticker,
            year=year,
            cosine_drift_total=drift_total,
            cosine_drift_mda=drift_mda,
            cosine_drift_risk=drift_risk,
            jaccard_distance=jaccard,
            sentiment_score=sent_score,
            category=category,
        )


class LazyPricesStrategy:
    """Constructs dollar-neutral portfolios going Long Lazy Disclosers and Short High-Drift firms."""

    def __init__(self, quantile_cutoff: float = 0.30):
        self.quantile_cutoff = quantile_cutoff

    def generate_positions(self, drift_df: pd.DataFrame) -> pd.DataFrame:
        df = drift_df.copy()
        q_low = df["cosine_drift_total"].quantile(self.quantile_cutoff)
        q_high = df["cosine_drift_total"].quantile(1.0 - self.quantile_cutoff)

        df["Weight"] = 0.0
        df["Recommendation"] = "HOLD"

        long_mask = df["cosine_drift_total"] <= q_low
        short_mask = df["cosine_drift_total"] >= q_high

        if long_mask.sum() > 0:
            df.loc[long_mask, "Weight"] = 0.5 / long_mask.sum()
            df.loc[long_mask, "Recommendation"] = "LONG (LAZY_ALPHA)"
        if short_mask.sum() > 0:
            df.loc[short_mask, "Weight"] = -0.5 / short_mask.sum()
            df.loc[short_mask, "Recommendation"] = "SHORT (HIGH_DRIFT_RISK)"

        return df
