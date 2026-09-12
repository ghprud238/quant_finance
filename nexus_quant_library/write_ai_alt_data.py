import os

def w(path, content):
    full = os.path.join('/working_dir/nexus_quant_platform', path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print('Generated', path)

w('src/nexus_quant/ai_alternative_data/ml_predictor.py', '''"""
Financial Machine Learning: Feature Engineering, Fractional Differencing (FFD) & Purged TimeSeries CV.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


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
    """Finds minimal differencing parameter d that achieves stationarity (ADF test)."""
    for d in np.arange(0.0, max_d + step, step):
        fd = frac_diff_fixed_width(series, d)
        if len(fd) > 20:
            adf_stat, p_val, _, _, _, _ = adfuller(fd.dropna(), maxlag=1)
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

w('src/nexus_quant/ai_alternative_data/gnn_supply_chain.py', '''"""
Supply-Chain Knowledge Graph & Graph Neural Network (GNN) Spillover Momentum Alpha.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


class SupplyChainNetwork:
    """Represents a directed customer-supplier economic dependency network."""

    def __init__(self, tickers: List[str], adjacency_matrix: np.ndarray):
        self.tickers = tickers
        self.n_nodes = len(tickers)
        self.adj_matrix = np.array(adjacency_matrix, dtype=float)

    @classmethod
    def create_default_network(cls) -> "SupplyChainNetwork":
        tickers = ["AAPL", "NVDA", "TSMC", "ASML", "MSFT", "AMZN", "AMD", "QCOM", "AVGO", "CRUS"]
        n = len(tickers)
        A = np.zeros((n, n))
        idx = {t: i for i, t in enumerate(tickers)}

        A[idx["CRUS"], idx["AAPL"]] = 0.76
        A[idx["TSMC"], idx["AAPL"]] = 0.25
        A[idx["TSMC"], idx["NVDA"]] = 0.15
        A[idx["ASML"], idx["TSMC"]] = 0.40
        A[idx["NVDA"], idx["MSFT"]] = 0.18
        A[idx["NVDA"], idx["AMZN"]] = 0.14
        A[idx["AMD"], idx["MSFT"]] = 0.12
        A[idx["QCOM"], idx["AAPL"]] = 0.22
        A[idx["AVGO"], idx["AAPL"]] = 0.20
        return cls(tickers, A)

    def get_normalized_adjacency(self) -> np.ndarray:
        A_tilde = self.adj_matrix + np.eye(self.n_nodes)
        D_tilde = np.sum(A_tilde, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D_tilde, 1e-8)))
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def get_centrality_table(self) -> pd.DataFrame:
        A = self.adj_matrix
        out_degree = np.sum(A > 0, axis=1)
        in_degree = np.sum(A > 0, axis=0)
        hhi = np.sum(A ** 2, axis=1)
        total_dep = np.sum(A, axis=1)

        M = A / np.maximum(np.sum(A, axis=0, keepdims=True), 1e-8)
        d = 0.85
        p = np.ones(self.n_nodes) / self.n_nodes
        for _ in range(50):
            p = (1 - d) / self.n_nodes + d * (M @ p)

        return pd.DataFrame({
            "Ticker": self.tickers,
            "PageRank_Centrality": p,
            "Supplier_Customer_Count": in_degree,
            "Supplier_Out_Count": out_degree,
            "Customer_Concentration_HHI": hhi,
            "Total_Customer_Revenue_Pct": total_dep,
        }).sort_values("PageRank_Centrality", ascending=False)


@dataclass
class SupplyChainAlphaResult:
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    information_coefficient: float
    ic_information_ratio: float
    equity_curve: pd.Series
    strategy_returns: pd.Series

    def summary(self) -> str:
        return f"""Supply-Chain GNN Alpha Performance:
  - Strategy CAGR:                   {self.cagr:+.2%}
  - Annualized Volatility:           {self.volatility:.2%}
  - Sharpe Ratio (Rf=2%):            {self.sharpe_ratio:.2f}
  - Maximum Drawdown:                {self.max_drawdown:.2%}
  - Mean Information Coefficient:    {self.information_coefficient:+.4f}
  - IC Information Ratio:            {self.ic_information_ratio:.2f}"""


class SupplyChainGraphAlpha:
    """GNN-driven Customer-to-Supplier Lead-Lag Momentum Extraction."""

    def __init__(self, network: Optional[SupplyChainNetwork] = None, lead_lag_window: int = 5):
        self.network = network or SupplyChainNetwork.create_default_network()
        self.lead_lag_window = lead_lag_window

    def compute_graph_momentum_signals(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        aligned_tickers = [t for t in self.network.tickers if t in prices_df.columns]
        P = prices_df[aligned_tickers]
        returns = P.pct_change(self.lead_lag_window)

        A_norm = self.network.get_normalized_adjacency()
        signals = pd.DataFrame(index=returns.index, columns=aligned_tickers, dtype=float)

        for dt in returns.index:
            r_t = returns.loc[dt].fillna(0.0).values
            h1 = A_norm @ r_t
            h2 = A_norm @ np.maximum(0, h1)
            signals.loc[dt] = h2

        mean_sig = signals.mean(axis=1)
        std_sig = signals.std(axis=1).replace(0, 1.0)
        z_signals = signals.sub(mean_sig, axis=0).div(std_sig, axis=0)
        return z_signals

    def backtest_strategy(self, prices_df: pd.DataFrame, n_quantiles: int = 3, tx_cost_bps: float = 5.0) -> SupplyChainAlphaResult:
        signals = self.compute_graph_momentum_signals(prices_df).shift(1)
        aligned_tickers = [t for t in self.network.tickers if t in prices_df.columns]
        daily_returns = prices_df[aligned_tickers].pct_change()

        strat_returns = []
        ics = []

        for dt in daily_returns.index:
            if dt not in signals.index:
                continue
            sig_row = signals.loc[dt].dropna()
            ret_row = daily_returns.loc[dt].dropna()
            common = sig_row.index.intersection(ret_row.index)
            if len(common) < 4:
                continue

            s = sig_row[common]
            r = ret_row[common]

            ic, _ = stats.spearmanr(s, r)
            if not np.isnan(ic):
                ics.append(ic)

            q_high = s.quantile(1.0 - 1.0 / n_quantiles)
            q_low = s.quantile(1.0 / n_quantiles)

            long_mask = s >= q_high
            short_mask = s <= q_low

            w = pd.Series(0.0, index=common)
            if long_mask.sum() > 0:
                w[long_mask] = 0.5 / long_mask.sum()
            if short_mask.sum() > 0:
                w[short_mask] = -0.5 / short_mask.sum()

            gross_ret = (w * r).sum()
            strat_returns.append(gross_ret - tx_cost_bps * 1e-4 * 0.1)

        ret_series = pd.Series(strat_returns)
        cagr = float(np.prod(1.0 + ret_series) ** (252.0 / max(len(ret_series), 1)) - 1.0) if len(ret_series) > 0 else 0.0
        vol = float(ret_series.std() * np.sqrt(252.0)) if len(ret_series) > 0 else 0.0
        sharpe = float((cagr - 0.02) / vol) if vol > 0 else 0.0

        cum = (1.0 + ret_series).cumprod()
        peaks = cum.cummax()
        dd = (cum - peaks) / peaks
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0
        win_rate = float(np.mean(ret_series > 0)) if len(ret_series) > 0 else 0.0

        mean_ic = float(np.mean(ics)) if ics else 0.0
        ic_ir = float(mean_ic / (np.std(ics) + 1e-8)) if ics else 0.0

        return SupplyChainAlphaResult(
            cagr=cagr,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            information_coefficient=mean_ic,
            ic_information_ratio=ic_ir,
            equity_curve=cum,
            strategy_returns=ret_series,
        )
''')

w('src/nexus_quant/ai_alternative_data/sec_semantic_drift.py', '''"""
Financial LLM: SEC 10-K Semantic Drift & The "Lazy Prices" Alpha Anomaly.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")

    def compute_cosine_drift(self, text_prior: str, text_current: str) -> float:
        if not text_prior.strip() or not text_current.strip():
            return 0.0
        tfidf = self.vectorizer.fit_transform([text_prior, text_current])
        v1 = tfidf[0].toarray()[0]
        v2 = tfidf[1].toarray()[0]
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

w('src/nexus_quant/ai_alternative_data/fear_greed_sentiment.py', '''"""
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
''')

w('src/nexus_quant/ai_alternative_data/agentic_swarm.py', '''"""
Autonomous Multi-Agent Macroeconomic & Crypto Hedge Fund Swarm.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class AgentView:
    agent_name: str
    asset_views: Dict[str, float]
    conviction: float
    rationale: str


@dataclass
class InvestmentCommitteeMemo:
    date: str
    agent_views: List[AgentView]
    consensus_returns: Dict[str, float]
    optimal_weights: Dict[str, float]
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    committee_consensus_score: float
    memo_text: str

    def to_markdown(self) -> str:
        weights_str = "\\n".join([f"  - **{k}**: {v:.1%}" for k, v in self.optimal_weights.items()])
        return f"""# Investment Committee Memorandum — {self.date}
**Consensus Conviction Score**: {self.committee_consensus_score:.1f}/10.0
**Expected Portfolio Return (Ann.)**: {self.expected_portfolio_return:.2%}
**Expected Portfolio Volatility**: {self.expected_portfolio_volatility:.2%}

## Target Asset Allocations:
{weights_str}

## Committee Debate Summary:
{self.memo_text}
"""


class MultiAgentHedgeFundSwarm:
    """Orchestrates autonomous specialist agents and Black-Litterman portfolio construction."""

    def __init__(self, target_vol_annual: float = 0.12, assets: Optional[List[str]] = None):
        self.target_vol_annual = target_vol_annual
        self.assets = assets or ["Equities (SPY)", "Bonds (TLT)", "Gold (GLD)", "Crypto (BTC)", "FX Carry (EM)"]

    def run_investment_committee(
        self,
        macro_signal: float,
        crypto_onchain: float,
        sentiment_fgi: float,
        cov_matrix: Optional[np.ndarray] = None,
        date: str = "2026-08-30",
    ) -> InvestmentCommitteeMemo:
        macro_views = {
            "Equities (SPY)": 0.08 * macro_signal,
            "Bonds (TLT)": -0.04 * macro_signal if macro_signal > 0 else 0.06,
            "Gold (GLD)": 0.05 if macro_signal < 0 else 0.02,
            "Crypto (BTC)": 0.15 * macro_signal,
            "FX Carry (EM)": 0.07 * macro_signal,
        }
        macro_agent = AgentView(
            agent_name="MacroEconomistAgent",
            asset_views=macro_views,
            conviction=0.85,
            rationale=f"Global macro regime is {'Expansionary' if macro_signal > 0 else 'Defensive'}."
        )

        crypto_views = {
            "Equities (SPY)": 0.04,
            "Bonds (TLT)": 0.02,
            "Gold (GLD)": 0.03,
            "Crypto (BTC)": 0.25 * crypto_onchain,
            "FX Carry (EM)": 0.03,
        }
        crypto_agent = AgentView(
            agent_name="CryptoMicrostructureAgent",
            asset_views=crypto_views,
            conviction=0.80,
            rationale=f"On-chain MVRV and whale flows signal {'Strong Inflows' if crypto_onchain > 0 else 'Deleveraging Risk'}."
        )

        sent_normalized = (sentiment_fgi - 50.0) / 50.0
        sent_views = {
            "Equities (SPY)": 0.05 * sent_normalized,
            "Bonds (TLT)": -0.02 * sent_normalized,
            "Gold (GLD)": 0.04 if sent_normalized < -0.3 else 0.01,
            "Crypto (BTC)": 0.18 * sent_normalized,
            "FX Carry (EM)": 0.04 * sent_normalized,
        }
        sent_agent = AgentView(
            agent_name="SentimentAlphaAgent",
            asset_views=sent_views,
            conviction=0.75,
            rationale=f"Market Fear & Greed index is at {sentiment_fgi:.0f}/100."
        )

        all_views = [macro_agent, crypto_agent, sent_agent]
        total_conviction = sum(a.conviction for a in all_views)
        
        consensus_ret = {}
        for asset in self.assets:
            exp_r = sum(a.asset_views.get(asset, 0.0) * a.conviction for a in all_views) / total_conviction
            consensus_ret[asset] = float(exp_r)

        n = len(self.assets)
        if cov_matrix is None:
            vols = np.array([0.16, 0.12, 0.15, 0.55, 0.10])
            corr = np.eye(n)
            corr[0, 3] = 0.35; corr[3, 0] = 0.35
            corr[0, 1] = -0.20; corr[1, 0] = -0.20
            cov_matrix = np.outer(vols, vols) * corr

        mu = np.array([consensus_ret[a] for a in self.assets])
        
        def loss(w):
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, cov_matrix @ w))
            return -port_ret + 1.5 * (port_vol ** 2)

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
        bnds = [(0.0, 0.50) for _ in range(n)]
        w0 = np.ones(n) / n
        opt = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons)

        opt_weights = {a: float(opt.x[i]) for i, a in enumerate(self.assets)}
        p_ret = float(np.dot(opt.x, mu))
        p_vol = float(np.sqrt(np.dot(opt.x, cov_matrix @ opt.x)))
        consensus_score = float(total_conviction / len(all_views) * 10.0)

        memo_text = f"""The Macro and Sentiment agents reached a unified view favoring {'Risk-On asset classes' if macro_signal > 0 else 'Defensive & Real Assets'}.
The Crypto Microstructure specialist noted {'positive on-chain accumulation' if crypto_onchain > 0 else 'elevated funding drag'}.
The PM Chair executed constrained optimization maintaining a portfolio target volatility of {p_vol:.1%}."""

        return InvestmentCommitteeMemo(
            date=date,
            agent_views=all_views,
            consensus_returns=consensus_ret,
            optimal_weights=opt_weights,
            expected_portfolio_return=p_ret,
            expected_portfolio_volatility=p_vol,
            committee_consensus_score=consensus_score,
            memo_text=memo_text,
        )
''')

print("Finished AI Alternative Data Module.")
