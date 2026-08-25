"""Financial LLM & SEC 10-K Semantic Drift Alpha Engine (Project 31).

Implements the 'Lazy Prices' research framework (Cohen, Malloy, Nguyen 2020) and
Loughran-McDonald (2011) financial domain sentiment and uncertainty analytics.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import re
import math
from collections import Counter
import numpy as np
import pandas as pd

# Loughran-McDonald Financial Sentiment Lexicon (Representative High-Signal Terms)
LM_NEGATIVE_WORDS = {
    "abandon", "abandoned", "abandonment", "adverse", "adversely", "against",
    "allege", "alleged", "allegation", "annul", "annulment", "arbitration",
    "arrearage", "bankruptcy", "breach", "breached", "cancel", "cancellation",
    "casualty", "catastrophe", "cease", "cessation", "claim", "closure",
    "collusion", "compliance", "concede", "concern", "condemn", "condemnation",
    "conflict", "contingency", "corrective", "crash", "crisis", "curtail",
    "curtailment", "damage", "damages", "debt", "default", "defect",
    "deficiency", "deficit", "delay", "delays", "delinquency", "delist",
    "depress", "depressed", "depreciation", "deteriorate", "deterioration",
    "detrimental", "diminish", "diminution", "disadvantage", "disapproval",
    "discipline", "disclaim", "disclose", "discontinuance", "discontinued",
    "discourage", "discredit", "dispute", "disputed", "disruption", "disruptions",
    "dissolution", "distress", "distressed", "divest", "divestiture",
    "doubt", "downward", "embargo", "embargoes", "emergency", "enforce",
    "enforcement", "erosion", "error", "errors", "eviction", "expropriation",
    "fail", "failed", "failure", "fine", "fines", "forfeit", "forfeiture",
    "fraud", "fraudulent", "freeze", "grievance", "halt", "hazard",
    "impair", "impaired", "impairment", "impede", "inability", "inadequate",
    "inadvisable", "incompetent", "indictment", "infringe", "infringement",
    "injunction", "insolvency", "insolvent", "investigation", "investigations",
    "judgment", "judgments", "lapse", "lawsuit", "lawsuits", "legal",
    "liability", "liabilities", "liquidate", "liquidation", "litigate",
    "litigation", "litigations", "loss", "losses", "lost", "malfeasance",
    "misleading", "mismanagement", "misrepresent", "misrepresentation",
    "noncompliance", "obsolete", "obsolescence", "penalize", "penalty",
    "penalties", "peril", "petition", "recession", "restructure",
    "restructuring", "revoke", "revocation", "risk", "risks", "sanction",
    "sanctions", "scarcity", "scrutiny", "settlement", "shortage", "shortages",
    "shutdown", "stagnant", "stagnation", "subpoena", "terminate",
    "termination", "threat", "threaten", "uncertain", "uncertainty",
    "uncertainties", "unfavorable", "unforeseen", "unlawful", "unstable",
    "volatile", "volatility", "vulnerability", "vulnerable", "warn", "warning"
}

LM_POSITIVE_WORDS = {
    "achieve", "achieved", "achievement", "advancement", "advantage",
    "advantageous", "attain", "attained", "benefit", "benefited",
    "beneficial", "boost", "breakthrough", "collaborate", "collaboration",
    "deliver", "delivered", "efficiency", "efficient", "enable",
    "enabled", "enhance", "enhanced", "enhancement", "exceed",
    "exceeded", "excel", "excellence", "expand", "expanded",
    "expansion", "gain", "gained", "gains", "grow", "growth",
    "innovate", "innovation", "innovative", "leader", "leadership",
    "milestone", "momentum", "optimize", "optimized", "optimization",
    "outperform", "outperformed", "partner", "partnership", "productive",
    "productivity", "profit", "profitable", "profitability", "progress",
    "record", "resilient", "resilience", "robust", "scalable",
    "strength", "strengthen", "strengthened", "strong", "succeed",
    "success", "successful", "surpass", "surpassed", "sustainable",
    "traction", "triumph", "upgrade", "upgraded", "upside", "win", "won"
}

LM_UNCERTAINTY_WORDS = {
    "almost", "alter", "altering", "alteration", "ambiguous", "anomaly",
    "anticipate", "anticipated", "appear", "approximate", "approximately",
    "arbitrary", "assume", "assumed", "assumption", "believe", "cautious",
    "clarify", "conceivable", "conditional", "confusion", "contingent",
    "could", "depend", "dependence", "dependent", "deviate", "differ",
    "differing", "doubt", "estimate", "estimated", "estimation",
    "fluctuate", "fluctuating", "fluctuation", "fluctuations", "forecast",
    "foreseeable", "gamble", "hidden", "hypothetical", "imprecise",
    "indefinite", "indeterminate", "instability", "intangible", "likelihood",
    "may", "maybe", "might", "nearly", "occasionally", "ordinarily",
    "pending", "perceive", "perceived", "perhaps", "plausible",
    "possible", "possibly", "potential", "potentially", "predict",
    "predicted", "preliminary", "presume", "probable", "probably",
    "random", "reassess", "reconsider", "revise", "revised",
    "risk", "risks", "rough", "roughly", "rumor", "seldom",
    "somewhat", "speculate", "speculation", "tentative", "uncertain",
    "uncertainty", "uncertainties", "unclear", "undetermined",
    "unexpected", "unforeseen", "unpredictable", "unresolved", "untested",
    "variable", "vary", "varying", "volatile", "volatility"
}

LM_LITIGIOUS_WORDS = {
    "allege", "allegation", "appellate", "arbitrate", "arbitration",
    "claimant", "counsel", "court", "damages", "defendant",
    "deposition", "dispute", "enjoin", "felony", "injunction",
    "investigation", "jurisdiction", "justice", "lawsuit", "lawsuits",
    "legal", "litigate", "litigation", "plaintiff", "prosecute",
    "regulation", "regulatory", "settlement", "statute", "statutory",
    "subpoena", "testimony", "tort", "trial", "verdict"
}

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am",
    "an", "and", "any", "are", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do",
    "does", "doesn't", "doing", "don't", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his",
    "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
    "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}


def clean_and_tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Cleans raw financial text into lowercase alphanumeric token list."""
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    tokens = [t for t in cleaned.split() if len(t) > 2]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


class SimpleTfidfVectorizer:
    """Pure Python/NumPy TF-IDF Vectorizer with Sublinear TF Scaling."""
    
    def __init__(self, min_df: int = 1, max_features: Optional[int] = 5000):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])
        
    def fit(self, raw_documents: List[str]) -> "SimpleTfidfVectorizer":
        doc_tokens = [clean_and_tokenize(doc) for doc in raw_documents]
        n_docs = len(doc_tokens)
        
        df_counts: Counter = Counter()
        for tokens in doc_tokens:
            unique_terms = set(tokens)
            df_counts.update(unique_terms)
            
        filtered_terms = [t for t, cnt in df_counts.items() if cnt >= self.min_df]
        if self.max_features is not None and len(filtered_terms) > self.max_features:
            filtered_terms = [t for t, _ in df_counts.most_common(self.max_features)]
            
        filtered_terms.sort()
        self.vocabulary_ = {t: i for i, t in enumerate(filtered_terms)}
        
        # Smooth IDF: ln((1 + N) / (1 + df)) + 1
        n_features = len(self.vocabulary_)
        self.idf_ = np.zeros(n_features)
        for term, idx in self.vocabulary_.items():
            df_val = df_counts[term]
            self.idf_[idx] = math.log((1.0 + n_docs) / (1.0 + df_val)) + 1.0
            
        return self
        
    def transform(self, raw_documents: List[str]) -> np.ndarray:
        n_docs = len(raw_documents)
        n_features = len(self.vocabulary_)
        if n_features == 0:
            return np.zeros((n_docs, 1))
            
        matrix = np.zeros((n_docs, n_features))
        for doc_idx, doc in enumerate(raw_documents):
            tokens = clean_and_tokenize(doc)
            tf_counts = Counter(tokens)
            for term, count in tf_counts.items():
                if term in self.vocabulary_:
                    col_idx = self.vocabulary_[term]
                    # Sublinear TF scaling: 1 + ln(count)
                    tf_scaled = 1.0 + math.log(count)
                    matrix[doc_idx, col_idx] = tf_scaled * self.idf_[col_idx]
                    
            # L2 normalization
            norm = np.linalg.norm(matrix[doc_idx])
            if norm > 0:
                matrix[doc_idx] /= norm
                
        return matrix
        
    def fit_transform(self, raw_documents: List[str]) -> np.ndarray:
        return self.fit(raw_documents).transform(raw_documents)


@dataclass
class DriftAnalysisResult:
    """Structured result of multi-year SEC 10-K semantic drift analysis."""
    ticker: str
    year_prior: int
    year_current: int
    cosine_dissimilarity_total: float
    jaccard_distance_total: float
    cosine_dissimilarity_mda: float
    cosine_dissimilarity_risk: float
    sentiment_prior: float
    sentiment_current: float
    sentiment_change: float
    uncertainty_prior: float
    uncertainty_current: float
    uncertainty_change: float
    litigious_ratio_current: float
    drift_category: str  # "HIGH_DRIFT", "MODERATE_DRIFT", "LAZY_DISCLOSURE"


class SemanticDriftEngine:
    """Calculates semantic drift and financial domain sentiment shifts in corporate filings."""
    
    def __init__(self, high_drift_threshold: float = 0.15, lazy_threshold: float = 0.04):
        self.high_drift_threshold = high_drift_threshold
        self.lazy_threshold = lazy_threshold
        
    @staticmethod
    def jaccard_distance(tokens_a: List[str], tokens_b: List[str]) -> float:
        """Computes Jaccard distance: 1 - |A intersect B| / |A union B|."""
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        if union == 0:
            return 0.0
        return 1.0 - (intersection / union)
        
    @staticmethod
    def cosine_dissimilarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes Cosine Dissimilarity: 1 - cos(a, b)."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        sim = max(-1.0, min(1.0, sim))
        return 1.0 - sim

    @staticmethod
    def loughran_mcdonald_sentiment(tokens: List[str]) -> Dict[str, float]:
        """Computes domain-specific Loughran-McDonald sentiment metrics."""
        if not tokens:
            return {"sentiment": 0.0, "uncertainty": 0.0, "litigious": 0.0, "negative_pct": 0.0, "positive_pct": 0.0}
            
        n_total = len(tokens)
        n_neg = sum(1 for t in tokens if t in LM_NEGATIVE_WORDS)
        n_pos = sum(1 for t in tokens if t in LM_POSITIVE_WORDS)
        n_unc = sum(1 for t in tokens if t in LM_UNCERTAINTY_WORDS)
        n_lit = sum(1 for t in tokens if t in LM_LITIGIOUS_WORDS)
        
        # Sentiment score bounded in [-1, +1]
        denom = n_pos + n_neg
        sentiment = (n_pos - n_neg) / denom if denom > 0 else 0.0
        
        return {
            "sentiment": sentiment,
            "uncertainty": n_unc / n_total,
            "litigious": n_lit / n_total,
            "negative_pct": (n_neg / n_total) * 100.0,
            "positive_pct": (n_pos / n_total) * 100.0,
        }

    def analyze_pair(
        self,
        ticker: str,
        doc_prior: Dict[str, str],
        doc_current: Dict[str, str],
        year_prior: int,
        year_current: int,
    ) -> DriftAnalysisResult:
        """Analyzes semantic drift between two annual 10-K filings of the same firm."""
        text_prior_full = doc_prior.get("Full_Text", "")
        text_curr_full = doc_current.get("Full_Text", "")
        
        text_prior_mda = doc_prior.get("Item_7_MDA", "")
        text_curr_mda = doc_current.get("Item_7_MDA", "")
        
        text_prior_risk = doc_prior.get("Item_1A_Risk_Factors", "")
        text_curr_risk = doc_current.get("Item_1A_Risk_Factors", "")
        
        # TF-IDF Embedding
        vec = SimpleTfidfVectorizer()
        corpus = [text_prior_full, text_curr_full, text_prior_mda, text_curr_mda, text_prior_risk, text_curr_risk]
        vecs = vec.fit_transform(corpus)
        
        cos_full = self.cosine_dissimilarity(vecs[0], vecs[1])
        cos_mda = self.cosine_dissimilarity(vecs[2], vecs[3])
        cos_risk = self.cosine_dissimilarity(vecs[4], vecs[5])
        
        tok_prior = clean_and_tokenize(text_prior_full)
        tok_curr = clean_and_tokenize(text_curr_full)
        jaccard_dist = self.jaccard_distance(tok_prior, tok_curr)
        
        # Sentiment
        sent_prior = self.loughran_mcdonald_sentiment(tok_prior)
        sent_curr = self.loughran_mcdonald_sentiment(tok_curr)
        
        if cos_full >= self.high_drift_threshold:
            category = "HIGH_DRIFT"
        elif cos_full <= self.lazy_threshold:
            category = "LAZY_DISCLOSURE"
        else:
            category = "MODERATE_DRIFT"
            
        return DriftAnalysisResult(
            ticker=ticker,
            year_prior=year_prior,
            year_current=year_current,
            cosine_dissimilarity_total=cos_full,
            jaccard_distance_total=jaccard_dist,
            cosine_dissimilarity_mda=cos_mda,
            cosine_dissimilarity_risk=cos_risk,
            sentiment_prior=sent_prior["sentiment"],
            sentiment_current=sent_curr["sentiment"],
            sentiment_change=sent_curr["sentiment"] - sent_prior["sentiment"],
            uncertainty_prior=sent_prior["uncertainty"],
            uncertainty_current=sent_curr["uncertainty"],
            uncertainty_change=sent_curr["uncertainty"] - sent_prior["uncertainty"],
            litigious_ratio_current=sent_curr["litigious"],
            drift_category=category
        )

    def analyze_universe(
        self,
        filings: Dict[str, Dict[int, Dict[str, str]]],
        target_year: int = 2023,
    ) -> pd.DataFrame:
        """Runs cross-sectional semantic drift analysis across a universe of corporate filings."""
        records = []
        for ticker, year_dict in filings.items():
            prior_year = target_year - 1
            if prior_year in year_dict and target_year in year_dict:
                res = self.analyze_pair(ticker, year_dict[prior_year], year_dict[target_year], prior_year, target_year)
                records.append({
                    "Ticker": ticker,
                    "Year": target_year,
                    "Cosine_Drift_Total": res.cosine_dissimilarity_total,
                    "Cosine_Drift_MDA": res.cosine_dissimilarity_mda,
                    "Cosine_Drift_Risk": res.cosine_dissimilarity_risk,
                    "Jaccard_Distance": res.jaccard_distance_total,
                    "Sentiment_Score": res.sentiment_current,
                    "Sentiment_Change": res.sentiment_change,
                    "Uncertainty_Score": res.uncertainty_current,
                    "Litigious_Score": res.litigious_ratio_current,
                    "Category": res.drift_category,
                })
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("Cosine_Drift_Total", ascending=False).reset_index(drop=True)
        return df


class LazyPricesStrategy:
    """Implements the Cohen, Malloy & Nguyen (2020) Lazy Prices anomaly.
    
    Generates alpha by going Long Low-Drift (unchanged/verbatim disclosures) and
    Short High-Drift (heavy edits / complex narrative revisions) firms.
    """
    
    def __init__(self, quantile_cutoff: float = 0.30):
        self.quantile_cutoff = quantile_cutoff
        
    def generate_positions(self, drift_df: pd.DataFrame) -> pd.DataFrame:
        """Generates dollar-neutral alpha portfolio weights from cross-sectional drift scores."""
        if drift_df.empty:
            return pd.DataFrame()
            
        df = drift_df.copy()
        n = len(df)
        k = max(1, int(n * self.quantile_cutoff))
        
        # Sort by total drift
        df = df.sort_values("Cosine_Drift_Total", ascending=True).reset_index(drop=True)
        
        df["Weight"] = 0.0
        # Long Low-Drift (first k assets)
        df.loc[:k-1, "Weight"] = 0.5 / k
        # Short High-Drift (last k assets)
        df.loc[n-k:, "Weight"] = -0.5 / k
        
        df["Recommendation"] = "NEUTRAL"
        df.loc[:k-1, "Recommendation"] = "LONG (Lazy Disclosures / Low Risk)"
        df.loc[n-k:, "Recommendation"] = "SHORT (High Semantic Drift / Narrative Shifts)"
        
        return df[["Ticker", "Cosine_Drift_Total", "Sentiment_Score", "Category", "Weight", "Recommendation"]]
