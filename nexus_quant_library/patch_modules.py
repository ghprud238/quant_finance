import os

def w(path, content):
    full = os.path.join('/working_dir/nexus_quant_platform', path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print('Patched', path)

w('src/nexus_quant/ai_alternative_data/ml_predictor.py', '''"""
Financial Machine Learning: Feature Engineering, Fractional Differencing (FFD) & Purged TimeSeries CV.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


def adfuller_test(series: pd.Series, maxlag: int = 1) -> Tuple[float, float]:
    """Lightweight Augmented Dickey-Fuller (ADF) test using pure NumPy/SciPy."""
    ts = series.dropna().values
    n = len(ts)
    if n < 10:
        return 0.0, 1.0

    dy = np.diff(ts)
    y_lag = ts[:-1]
    
    # Regression: dy_t = alpha + beta * y_{t-1} + gamma * dy_{t-1} + eps_t
    if maxlag >= 1 and len(dy) > 2:
        X = np.column_stack([np.ones(len(dy) - 1), y_lag[1:], dy[:-1]])
        y = dy[1:]
    else:
        X = np.column_stack([np.ones(len(dy)), y_lag])
        y = dy

    # OLS: beta_hat = (X'X)^(-1) X'y
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        params = np.linalg.solve(XtX + 1e-8 * np.eye(X.shape[1]), Xty)
        residuals = y - X @ params
        s2 = np.sum(residuals ** 2) / max(len(y) - X.shape[1], 1)
        cov = s2 * np.linalg.inv(XtX + 1e-8 * np.eye(X.shape[1]))
        beta = params[1]
        se_beta = np.sqrt(max(cov[1, 1], 1e-10))
        t_stat = beta / se_beta

        # Approximate p-value from MacKinnon critical values (-3.43 for 1%, -2.86 for 5%, -2.57 for 10%)
        # Standard logistic approximation for ADF p-value
        p_val = float(1.0 / (1.0 + np.exp(-1.5 * (t_stat + 2.86))))
        return float(t_stat), float(np.clip(p_val, 0.0, 1.0))
    except Exception:
        return 0.0, 1.0


def get_weights_ffd(d: float, threshold: float = 1e-4, max_lags: int = 200) -> np.ndarray:
    """Computes weights for fixed-width window fractional differencing."""
    w = [1.0]
    k = 1
    while k < max_lags:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff_fixed_width(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Applies fixed-width window fractional differentiation to a time series."""
    weights = get_weights_ffd(d, threshold=threshold)
    width = len(weights)
    res = {}
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1]
        res[series.index[i]] = np.dot(weights, window)
    return pd.Series(res, name=f"{series.name}_fracdiff_{d:.2f}")


def find_min_d(series: pd.Series, max_d: float = 1.0, step: float = 0.05, p_val_threshold: float = 0.05) -> float:
    """Finds minimal differencing parameter d that achieves stationarity."""
    for d in np.arange(0.0, max_d + step, step):
        fd = frac_diff_fixed_width(series, d)
        if len(fd) > 20:
            _, p_val = adfuller_test(fd.dropna(), maxlag=1)
            if p_val <= p_val_threshold:
                return float(round(d, 2))
    return 1.0


class FinancialFeatureEngineer:
    """Generates stationary technical, volatility, momentum, and fractional differentiation features."""

    def __init__(self, frac_diff_d: Optional[float] = None):
        self.frac_diff_d = frac_diff_d

    def engineer_features(self, df_ohlc: pd.DataFrame, target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Engineers multi-horizon feature matrix X and forward return target y."""
        df = df_ohlc.copy()
        features = pd.DataFrame(index=df.index)

        # Multi-horizon returns & momentum
        close = df["Close"]
        for h in [1, 5, 21, 63, 126]:
            features[f"ret_{h}d"] = close.pct_change(h)

        # Volatility features
        features["vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        if "High" in df.columns and "Low" in df.columns:
            hl = np.log(df["High"] / df["Low"])
            features["parkinson_vol"] = np.sqrt((hl ** 2).rolling(21).mean() / (4 * np.log(2)) * 252)
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            co = np.log(df["Close"] / df["Open"])
            hl = np.log(df["High"] / df["Low"])
            features["garman_klass_vol"] = np.sqrt(
                (0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)).rolling(21).mean() * 252
            )

        # Technical Oscillators
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # Bollinger Z-Score
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features["bollinger_z"] = (close - sma20) / std20.replace(0, np.nan)

        # Moving average ratios
        features["ma_ratio_20_200"] = close.rolling(20).mean() / close.rolling(200).mean() - 1.0

        # Fractional Differencing
        if self.frac_diff_d is None:
            d_opt = find_min_d(close.dropna())
            self.frac_diff_d = d_opt
        
        fd_series = frac_diff_fixed_width(close, self.frac_diff_d)
        features["frac_diff"] = fd_series

        # Target forward returns
        target = close.pct_change(target_horizon).shift(-target_horizon).rename("target_return")

        combined = pd.concat([features, target], axis=1).dropna()
        X = combined.drop(columns=["target_return"])
        y = combined["target_return"]
        return X, y


class PurgedTimeSeriesSplit:
    """Purged and Embargoed K-Fold Cross-Validation for Financial Time Series."""

    def __init__(self, n_splits: int = 5, purge_window: int = 5, embargo_window: int = 5):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False
            
            left_purge_start = max(0, test_start - self.purge_window)
            train_mask[left_purge_start:test_start] = False

            right_embargo_end = min(n_samples, test_end + self.embargo_window)
            train_mask[test_end:right_embargo_end] = False

            train_idx = indices[train_mask]
            splits.append((train_idx, test_idx))
        return splits


@dataclass
class MLPredictorResult:
    information_coefficient: float
    rank_information_coefficient: float
    mse: float
    r_squared: float
    directional_hit_rate: float
    feature_importances: pd.Series
    oof_predictions: pd.Series
    actual_returns: pd.Series

    def summary(self) -> str:
        return f"""Financial ML Performance:
  - Out-of-Sample IC (Pearson):      {self.information_coefficient:+.4f}
  - Rank IC (Spearman):              {self.rank_information_coefficient:+.4f}
  - Directional Hit Rate:            {self.directional_hit_rate:.2%}
  - Mean Squared Error (MSE):        {self.mse:.6f}
  - Out-of-Sample R^2:               {self.r_squared:+.4f}
  - Top 3 Alpha Drivers:             {', '.join(self.feature_importances.head(3).index.tolist())}"""


class MLReturnPredictor:
    """Regularized Financial ML Predictor with Purged K-Fold Cross Validation."""

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0, n_splits: int = 5, purge_window: int = 5):
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.n_splits = n_splits
        self.purge_window = purge_window

    def fit_predict_cv(self, X: pd.DataFrame, y: pd.Series) -> MLPredictorResult:
        cv = PurgedTimeSeriesSplit(n_splits=self.n_splits, purge_window=self.purge_window)
        oof_preds = pd.Series(index=y.index, dtype=float)
        weights_list = []

        X_mat = X.values
        y_vec = y.values

        for train_idx, test_idx in cv.split(X):
            X_tr, y_tr = X_mat[train_idx], y_vec[train_idx]
            X_te = X_mat[test_idx]

            mu, std = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-8
            X_tr_std = (X_tr - mu) / std
            X_te_std = (X_te - mu) / std

            p = X_tr_std.shape[1]
            XtX = X_tr_std.T @ X_tr_std
            Xty = X_tr_std.T @ y_tr
            w = np.linalg.solve(XtX + self.alpha * np.eye(p), Xty)
            weights_list.append(w)

            y_pred = X_te_std @ w
            oof_preds.iloc[test_idx] = y_pred

        valid_mask = ~oof_preds.isna()
        y_true = y[valid_mask]
        y_hat = oof_preds[valid_mask]

        ic, _ = stats.pearsonr(y_true, y_hat)
        rank_ic, _ = stats.spearmanr(y_true, y_hat)
        mse = float(np.mean((y_true - y_hat) ** 2))
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_hat) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        hit_rate = float(np.mean(np.sign(y_true) == np.sign(y_hat)))

        avg_weights = np.mean(weights_list, axis=0)
        feat_imp = pd.Series(np.abs(avg_weights), index=X.columns).sort_values(ascending=False)

        return MLPredictorResult(
            information_coefficient=float(ic),
            rank_information_coefficient=float(rank_ic),
            mse=mse,
            r_squared=r2,
            directional_hit_rate=hit_rate,
            feature_importances=feat_imp,
            oof_predictions=y_hat,
            actual_returns=y_true,
        )
''')

w('src/nexus_quant/ai_alternative_data/sec_semantic_drift.py', '''"""
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
        words = re.findall(r"\\b[a-zA-Z]{2,}\\b", text.lower())
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
        tokens = full_curr.lower().split()
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
''')

w('src/nexus_quant/defi_prediction_markets/__init__.py', '''"""
DeFi AMM Liquidity, Loss-Versus-Rebalancing (LVR), Perpetual Basis & Prediction Market Arbitrage Engine.
"""

from .uniswap_v3 import ConcentratedLiquidityAMM, ConstantProductAMM, UniswapPosition, SwapResult
from .lvr_model import ImpermanentLossCalculator, LossVersusRebalancingEngine, LPSimulationResult
from .perp_basis import PerpetualFundingEngine, CashAndCarryBasisTrader, BasisTradeResult
from .prediction_arbitrage import (
    PredictionMarketArbitrageEngine,
    ArbitrageOpportunity,
    KellyAllocation,
    BinaryOrderBook,
    OrderBookLevel,
)

__all__ = [
    "ConcentratedLiquidityAMM",
    "ConstantProductAMM",
    "UniswapPosition",
    "SwapResult",
    "ImpermanentLossCalculator",
    "LossVersusRebalancingEngine",
    "LPSimulationResult",
    "PerpetualFundingEngine",
    "CashAndCarryBasisTrader",
    "BasisTradeResult",
    "PredictionMarketArbitrageEngine",
    "ArbitrageOpportunity",
    "KellyAllocation",
    "BinaryOrderBook",
    "OrderBookLevel",
]
''')

print("Finished patching modules with pure numpy/scipy!")
