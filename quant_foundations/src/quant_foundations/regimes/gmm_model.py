"""
Gaussian Mixture Model (GMM) for Market Regime Detection.

Performs unsupervised clustering on (Return, Volatility) feature space using
the Expectation-Maximization (EM) algorithm implemented with NumPy and SciPy.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


REGIME_NAMES_2STATE = ["Bear", "Bull"]
REGIME_NAMES_3STATE = ["Bear", "Neutral", "Bull"]


class GMMRegimeDetector:
    """
    Gaussian Mixture Model for Market Regime Clustering.

    Clusters market conditions into distinct regimes within the (Return, Realized Volatility)
    feature plane using the Expectation-Maximization (EM) algorithm.

    Features:
    - Automatic 2D feature extraction from raw price or return series (Return, Rolling Volatility).
    - EM optimization with full or diagonal covariance matrices and variance regularization.
    - Automatic state sorting and labeling (Bear, Neutral, Bull).
    - Soft posterior cluster probabilities and hard regime assignments.
    - Regime-conditional performance statistics (Annualized Return, Volatility, Sharpe).
    """

    def __init__(
        self,
        n_components: int = 3,
        vol_window: int = 21,
        max_iter: int = 150,
        tol: float = 1e-5,
        covariance_type: str = "full",
        random_state: Optional[int] = 42,
        annualization_factor: int = 252,
        var_floor: float = 1e-6,
    ):
        """
        Initialize the GMM Regime Detector.

        Parameters
        ----------
        n_components : int, default 3
            Number of mixture components (typically 2 for Bull/Bear or 3 for Bull/Neutral/Bear).
        vol_window : int, default 21
            Rolling window for calculating realized volatility when 1D returns are provided.
        max_iter : int, default 150
            Maximum number of EM iterations.
        tol : float, default 1e-5
            Log-likelihood convergence threshold.
        covariance_type : str, default 'full'
            Type of covariance matrix: 'full' or 'diag'.
        random_state : int, optional
            Random seed.
        annualization_factor : int, default 252
            Trading periods per year.
        var_floor : float, default 1e-6
            Regularization added to covariance diagonals.
        """
        if n_components < 2:
            raise ValueError("n_components must be at least 2.")

        self.n_components = n_components
        self.vol_window = vol_window
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type.lower()
        self.random_state = random_state
        self.annualization_factor = annualization_factor
        self.var_floor = var_floor

        # Parameters
        self.is_fitted: bool = False
        self.weights_: np.ndarray = np.array([])         # shape (K,)
        self.means_: np.ndarray = np.array([])           # shape (K, d)
        self.covariances_: np.ndarray = np.array([])     # shape (K, d, d)
        self.log_likelihood_: float = -np.inf

        # Outputs
        self.regimes_: np.ndarray = np.array([])
        self.regime_names_: List[str] = []
        self.regime_labels_: List[str] = []
        self.responsibilities_: np.ndarray = np.array([])  # shape (T, K)
        self.feature_names_: List[str] = []
        self.features_: np.ndarray = np.array([])
        self.index_: Optional[pd.Index] = None

    def _extract_features(
        self,
        data: Union[pd.Series, pd.DataFrame, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[pd.Index], List[str]]:
        """Extract (Return, Realized Volatility) 2D features if 1D data is provided."""
        if isinstance(data, pd.DataFrame):
            if data.shape[1] >= 2:
                df = data.dropna()
                return df.values, df.index, list(df.columns)
            else:
                s = data.iloc[:, 0]
        elif isinstance(data, pd.Series):
            s = data
        else:
            arr = np.asarray(data)
            if arr.ndim > 1 and arr.shape[1] >= 2:
                valid_mask = ~np.isnan(arr).any(axis=1)
                return arr[valid_mask], None, [f"Feature_{i+1}" for i in range(arr.shape[1])]
            s = pd.Series(arr.ravel())

        s = s.dropna()
        # Compute rolling volatility
        rolling_vol = s.rolling(window=self.vol_window).std() * np.sqrt(self.annualization_factor)

        feature_df = pd.DataFrame(
            {
                "Return": s,
                "Realized_Vol": rolling_vol,
            },
            index=s.index if hasattr(s, "index") else None,
        ).dropna()

        return feature_df.values, feature_df.index, ["Return", "Realized_Vol"]

    def _init_parameters(self, X: np.ndarray) -> None:
        """Initialize GMM parameters using quantiles on the first feature."""
        N, d = X.shape
        K = self.n_components

        # Weights: uniform
        self.weights_ = np.full(K, 1.0 / K)

        # Sort indices by returns
        sorted_indices = np.argsort(X[:, 0])
        splits = np.array_split(sorted_indices, K)

        self.means_ = np.zeros((K, d))
        self.covariances_ = np.zeros((K, d, d))

        overall_cov = np.cov(X, rowvar=False, ddof=1) if N > 1 else np.eye(d)
        if d == 1:
            overall_cov = np.array([[float(overall_cov)]])

        for k in range(K):
            sub_X = X[splits[k]]
            if len(sub_X) > 1:
                self.means_[k] = np.mean(sub_X, axis=0)
                sub_cov = np.cov(sub_X, rowvar=False, ddof=1)
                if d == 1:
                    sub_cov = np.array([[float(sub_cov)]])
                if self.covariance_type == "diag":
                    sub_cov = np.diag(np.diag(sub_cov))
                self.covariances_[k] = sub_cov + np.eye(d) * self.var_floor
            else:
                self.means_[k] = np.mean(X, axis=0)
                self.covariances_[k] = overall_cov + np.eye(d) * self.var_floor

    def _compute_responsibilities(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        E-step: Compute posterior responsibilities gamma[i, k] = P(Z_i = k | X_i).

        Returns
        -------
        gamma : np.ndarray, shape (N, K)
        log_likelihood : float
        """
        N, d = X.shape
        K = self.n_components
        weighted_densities = np.zeros((N, K))

        for k in range(K):
            cov_k = self.covariances_[k] + np.eye(d) * self.var_floor
            try:
                dist = stats.multivariate_normal(mean=self.means_[k], cov=cov_k, allow_singular=True)
                dens = dist.pdf(X)
            except Exception:
                var_k = np.maximum(np.diag(cov_k), self.var_floor)
                dens = np.ones(N)
                for dim in range(d):
                    dens *= stats.norm.pdf(X[:, dim], loc=self.means_[k, dim], scale=np.sqrt(var_k[dim]))
            weighted_densities[:, k] = np.maximum(self.weights_[k] * dens, 1e-300)

        sum_densities = np.sum(weighted_densities, axis=1, keepdims=True)
        gamma = weighted_densities / np.maximum(sum_densities, 1e-300)
        log_likelihood = float(np.sum(np.log(np.maximum(sum_densities.ravel(), 1e-300))))

        return gamma, log_likelihood

    def fit(
        self,
        data: Union[pd.Series, pd.DataFrame, np.ndarray],
    ) -> "GMMRegimeDetector":
        """
        Fit the Gaussian Mixture Model on the feature dataset using EM.

        Parameters
        ----------
        data : pd.Series, pd.DataFrame, or np.ndarray
            Returns or multi-feature data.

        Returns
        -------
        self : GMMRegimeDetector
            Fitted detector instance.
        """
        X, idx, f_names = self._extract_features(data)
        self.features_ = X
        self.index_ = idx
        self.feature_names_ = f_names
        N, d = X.shape
        K = self.n_components

        self._init_parameters(X)
        prev_log_lik = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            gamma, log_lik = self._compute_responsibilities(X)

            # Check convergence
            if np.abs(log_lik - prev_log_lik) < self.tol:
                break
            prev_log_lik = log_lik

            # M-step
            N_k = np.sum(gamma, axis=0)  # shape (K,)

            # 1. Weights
            self.weights_ = N_k / N

            # 2. Means & Covariances
            for k in range(K):
                denom = max(N_k[k], 1e-300)
                self.means_[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / denom

                diff = X - self.means_[k]
                weighted_diff = gamma[:, k : k + 1] * diff
                cov_k = (weighted_diff.T @ diff) / denom
                if self.covariance_type == "diag":
                    cov_k = np.diag(np.diag(cov_k))
                self.covariances_[k] = cov_k + np.eye(d) * self.var_floor

        # Final evaluation
        gamma, self.log_likelihood_ = self._compute_responsibilities(X)

        # Sort states by return (feature 0)
        order = np.argsort(self.means_[:, 0])
        self.weights_ = self.weights_[order]
        self.means_ = self.means_[order]
        self.covariances_ = self.covariances_[order]
        self.responsibilities_ = gamma[:, order]

        if K == 2:
            self.regime_names_ = REGIME_NAMES_2STATE
        elif K == 3:
            self.regime_names_ = REGIME_NAMES_3STATE
        else:
            self.regime_names_ = [f"Cluster_{i}" for i in range(K)]

        # Hard clustering
        self.regimes_ = np.argmax(self.responsibilities_, axis=1)
        self.regime_labels_ = [self.regime_names_[s] for s in self.regimes_]

        self.is_fitted = True
        return self

    def predict(
        self,
        data: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """
        Predict regime labels for data.

        Returns
        -------
        regimes : pd.Series or np.ndarray
            Predicted regime labels or state indices.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")

        if data is None:
            if self.index_ is not None:
                return pd.Series(self.regime_labels_, index=self.index_, name="Regime")
            return self.regimes_

        X, idx, _ = self._extract_features(data)
        gamma, _ = self._compute_responsibilities(X)
        order = np.argsort(self.means_[:, 0])
        gamma_sorted = gamma[:, order]
        seq = np.argmax(gamma_sorted, axis=1)
        labels = [self.regime_names_[s] for s in seq]

        if idx is not None:
            return pd.Series(labels, index=idx, name="Regime")
        return seq

    def predict_proba(
        self,
        data: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Compute posterior regime probabilities.

        Returns
        -------
        df : pd.DataFrame or np.ndarray
            Matrix of component responsibilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing probabilities.")

        if data is None:
            if self.index_ is not None:
                return pd.DataFrame(
                    self.responsibilities_,
                    index=self.index_,
                    columns=[f"Prob_{name}" for name in self.regime_names_],
                )
            return self.responsibilities_

        X, idx, _ = self._extract_features(data)
        gamma, _ = self._compute_responsibilities(X)
        order = np.argsort(self.means_[:, 0])
        gamma_sorted = gamma[:, order]

        if idx is not None:
            return pd.DataFrame(
                gamma_sorted,
                index=idx,
                columns=[f"Prob_{name}" for name in self.regime_names_],
            )
        return gamma_sorted

    def regime_metrics(
        self,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute regime-conditional metrics.

        Returns
        -------
        df : pd.DataFrame
            Table with regime performance statistics.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        ann = self.annualization_factor
        sqrt_ann = np.sqrt(ann)
        T_total = len(self.regimes_)

        metrics = []
        for state_idx, name in enumerate(self.regime_names_):
            mask = (self.regimes_ == state_idx)
            count = int(np.sum(mask))
            freq = (count / T_total) * 100.0 if T_total > 0 else 0.0

            if count > 0:
                ret_subset = self.features_[mask, 0]
                daily_mean = float(np.mean(ret_subset))
                daily_vol = float(np.std(ret_subset, ddof=1)) if count > 1 else 0.0
            else:
                daily_mean = float(self.means_[state_idx, 0])
                daily_vol = float(np.sqrt(self.covariances_[state_idx, 0, 0]))

            ann_return = daily_mean * ann
            ann_vol = daily_vol * sqrt_ann
            sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-12 else 0.0

            metrics.append({
                "Regime": name,
                "Observations": count,
                "Frequency_Pct": freq,
                "Mixture_Weight": float(self.weights_[state_idx]),
                "Mean_Return_Daily": daily_mean,
                "Annualized_Return": ann_return,
                "Daily_Volatility": daily_vol,
                "Annualized_Volatility": ann_vol,
                "Sharpe_Ratio": sharpe,
            })

        return pd.DataFrame(metrics).set_index("Regime")

    def summary(self) -> Dict:
        """Return comprehensive GMM regime detection summary dictionary."""
        return {
            "n_components": self.n_components,
            "log_likelihood": self.log_likelihood_,
            "weights": {name: float(w) for name, w in zip(self.regime_names_, self.weights_)},
            "means": {name: self.means_[i].tolist() for i, name in enumerate(self.regime_names_)},
            "regime_metrics": self.regime_metrics().to_dict(orient="index"),
        }
