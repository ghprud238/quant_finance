"""
Gaussian Hidden Markov Model (HMM) for Market Regime Detection.

Implements Expectation-Maximization (Baum-Welch algorithm) from scratch with
Viterbi decoding, smoothed posteriors, transition matrix analysis,
stationary distributions, and regime-conditional performance metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


REGIME_NAMES_2STATE = ["Bear", "Bull"]
REGIME_NAMES_3STATE = ["Bear", "Neutral", "Bull"]


class GaussianHMMRegimeDetector:
    """
    Gaussian Hidden Markov Model for Financial Market Regime Detection.

    Identifies latent macro regimes (e.g. Bull, Bear, Neutral/Sideways) from
    asset return series or multi-asset / multi-feature inputs.

    Features:
    - Baum-Welch (EM) parameter estimation with scaling to prevent numerical underflow.
    - Fully vectorized matrix operations for fast, stable training.
    - Viterbi algorithm for maximum a posteriori (MAP) most likely state sequence.
    - Forward-Backward algorithm for smoothed posterior state probabilities.
    - Automatic, consistent state ordering (e.g., State 0: Bear, State 1: Neutral, State 2: Bull).
    - Transition probability matrix, stationary distribution, and expected regime durations.
    - Regime-conditional performance statistics (Annualized Return, Volatility, Sharpe Ratio).
    """

    def __init__(
        self,
        n_states: int = 2,
        max_iter: int = 150,
        tol: float = 1e-6,
        random_state: Optional[int] = 42,
        annualization_factor: int = 252,
        var_floor: float = 1e-6,
    ):
        """
        Initialize the Gaussian HMM regime detector.

        Parameters
        ----------
        n_states : int, default 2
            Number of hidden regimes (typically 2 for Bull/Bear or 3 for Bull/Neutral/Bear).
        max_iter : int, default 150
            Maximum number of Baum-Welch EM iterations.
        tol : float, default 1e-6
            Convergence tolerance for change in log-likelihood.
        random_state : int, optional
            Random seed for parameter initialization.
        annualization_factor : int, default 252
            Trading periods per year.
        var_floor : float, default 1e-6
            Minimum variance floor to prevent numerical singularity.
        """
        if n_states < 2:
            raise ValueError("n_states must be at least 2.")

        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.annualization_factor = annualization_factor
        self.var_floor = var_floor

        # Model parameters
        self.is_fitted: bool = False
        self.start_prob_: np.ndarray = np.array([])          # shape (K,)
        self.trans_mat_: np.ndarray = np.array([])           # shape (K, K)
        self.means_: np.ndarray = np.array([])               # shape (K, d)
        self.covars_: np.ndarray = np.array([])              # shape (K, d, d)
        self.log_likelihood_: float = -np.inf
        self.n_features_: int = 1

        # Fitted outputs
        self.regimes_: np.ndarray = np.array([])             # sequence of state indices
        self.regime_names_: List[str] = []
        self.regime_labels_: List[str] = []                  # sequence of state names
        self.posterior_probs_: np.ndarray = np.array([])     # shape (T, K)
        self.stationary_dist_: np.ndarray = np.array([])     # shape (K,)
        self.expected_durations_: Dict[str, float] = {}
        self.index_: Optional[pd.Index] = None
        self.raw_data_: np.ndarray = np.array([])

    def _init_params(self, X: np.ndarray) -> None:
        """Initialize HMM parameters using quantiles / K-means-like heuristic."""
        rng = np.random.RandomState(self.random_state)
        T, d = X.shape
        K = self.n_states

        # Initial state distribution: uniform
        self.start_prob_ = np.full(K, 1.0 / K)

        # Transition matrix: sticky diagonal prior (0.90 on diag, 0.10 off-diag)
        diag_prob = 0.90
        off_diag = (1.0 - diag_prob) / (K - 1)
        self.trans_mat_ = np.full((K, K), off_diag)
        np.fill_diagonal(self.trans_mat_, diag_prob)

        # Initialize means based on sorted quantiles of the primary feature (returns)
        sorted_indices = np.argsort(X[:, 0])
        splits = np.array_split(sorted_indices, K)

        means = np.zeros((K, d))
        covars = np.zeros((K, d, d))

        overall_cov = np.cov(X, rowvar=False, ddof=1) if T > 1 else np.eye(d)
        if d == 1:
            overall_cov = np.array([[float(overall_cov)]])

        for k in range(K):
            sub_X = X[splits[k]]
            if len(sub_X) > 1:
                means[k] = np.mean(sub_X, axis=0)
                sub_cov = np.cov(sub_X, rowvar=False, ddof=1)
                if d == 1:
                    sub_cov = np.array([[float(sub_cov)]])
                covars[k] = sub_cov + np.eye(d) * self.var_floor
            else:
                means[k] = np.mean(X, axis=0) + rng.normal(0, 0.01, size=d)
                covars[k] = overall_cov + np.eye(d) * self.var_floor

        self.means_ = means
        self.covars_ = covars

    def _compute_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian emission densities B[t, k] = N(X_t; means_[k], covars_[k]).
        Returns array of shape (T, K).
        """
        T, d = X.shape
        K = self.n_states
        B = np.zeros((T, K))

        if d == 1:
            # Vectorized 1D Gaussian for speed
            for k in range(K):
                mu = self.means_[k, 0]
                var = max(float(self.covars_[k, 0, 0]), self.var_floor)
                std = np.sqrt(var)
                dens = (1.0 / (np.sqrt(2.0 * np.pi) * std)) * np.exp(-0.5 * ((X[:, 0] - mu) / std) ** 2)
                B[:, k] = np.maximum(dens, 1e-300)
        else:
            for k in range(K):
                mean_k = self.means_[k]
                cov_k = self.covars_[k] + np.eye(d) * self.var_floor
                try:
                    dist = stats.multivariate_normal(mean=mean_k, cov=cov_k, allow_singular=True)
                    dens = dist.pdf(X)
                except Exception:
                    var_k = np.maximum(np.diag(cov_k), self.var_floor)
                    dens = np.ones(T)
                    for dim in range(d):
                        dens *= stats.norm.pdf(X[:, dim], loc=mean_k[dim], scale=np.sqrt(var_k[dim]))
                B[:, k] = np.maximum(dens, 1e-300)

        return B

    def _forward_backward(self, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Scaled Forward-Backward algorithm for Baum-Welch.

        Returns
        -------
        alpha_hat : np.ndarray, shape (T, K)
        beta_hat : np.ndarray, shape (T, K)
        c : np.ndarray, shape (T,)
        log_likelihood : float
        """
        T, K = B.shape
        alpha_hat = np.zeros((T, K))
        c = np.zeros(T)

        # Forward pass
        alpha_0 = self.start_prob_ * B[0]
        c[0] = 1.0 / np.maximum(np.sum(alpha_0), 1e-300)
        alpha_hat[0] = alpha_0 * c[0]

        for t in range(1, T):
            alpha_t = (alpha_hat[t - 1] @ self.trans_mat_) * B[t]
            c[t] = 1.0 / np.maximum(np.sum(alpha_t), 1e-300)
            alpha_hat[t] = alpha_t * c[t]

        log_likelihood = -float(np.sum(np.log(np.maximum(c, 1e-300))))

        # Backward pass
        beta_hat = np.zeros((T, K))
        beta_hat[T - 1] = c[T - 1]

        for t in range(T - 2, -1, -1):
            beta_t = self.trans_mat_ @ (B[t + 1] * beta_hat[t + 1])
            beta_hat[t] = beta_t * c[t]

        return alpha_hat, beta_hat, c, log_likelihood

    def fit(
        self,
        returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    ) -> "GaussianHMMRegimeDetector":
        """
        Fit the Gaussian HMM model using Baum-Welch EM algorithm.

        Parameters
        ----------
        returns : pd.Series, pd.DataFrame, or np.ndarray
            Asset return series or feature matrix (e.g. Return, Realized Volatility).

        Returns
        -------
        self : GaussianHMMRegimeDetector
            Fitted detector instance.
        """
        X, idx = self._prepare_data(returns)
        self.index_ = idx
        self.raw_data_ = X
        T, d = X.shape
        self.n_features_ = d
        K = self.n_states

        self._init_params(X)
        prev_log_lik = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            B = self._compute_emission_probs(X)
            alpha_hat, beta_hat, c, log_lik = self._forward_backward(B)

            # Check convergence
            if np.abs(log_lik - prev_log_lik) < self.tol:
                break
            prev_log_lik = log_lik

            # Smoothed state posteriors gamma[t, k]
            gamma = alpha_hat * beta_hat / c[:, np.newaxis]
            gamma = gamma / np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-300)

            # Vectorized computation of transition posteriors xi sum
            U = B[1:] * beta_hat[1:]  # shape (T-1, K)
            denom = np.sum(alpha_hat[:-1] * (U @ self.trans_mat_.T), axis=1, keepdims=True)
            U_norm = U / np.maximum(denom, 1e-300)
            xi_sum = self.trans_mat_ * (alpha_hat[:-1].T @ U_norm)

            # M-step updates
            # 1. Start probabilities
            self.start_prob_ = gamma[0] / np.sum(gamma[0])

            # 2. Transition matrix
            gamma_sum_Tminus1 = np.sum(gamma[:-1], axis=0)
            for i in range(K):
                denom_i = max(gamma_sum_Tminus1[i], 1e-300)
                self.trans_mat_[i] = xi_sum[i] / denom_i
                self.trans_mat_[i] /= np.sum(self.trans_mat_[i])

            # 3. Means & Covariances
            gamma_sum = np.sum(gamma, axis=0)
            for k in range(K):
                denom_k = max(gamma_sum[k], 1e-300)
                self.means_[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / denom_k

                diff = X - self.means_[k]  # shape (T, d)
                weighted_diff = gamma[:, k : k + 1] * diff
                cov_k = (weighted_diff.T @ diff) / denom_k
                self.covars_[k] = cov_k + np.eye(d) * self.var_floor

        # Final pass
        B = self._compute_emission_probs(X)
        alpha_hat, beta_hat, c, self.log_likelihood_ = self._forward_backward(B)
        gamma = alpha_hat * beta_hat / c[:, np.newaxis]
        gamma = gamma / np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-300)

        # Sort and label states consistently
        self._sort_and_label_states(X, gamma)

        # Viterbi decoding for optimal regime sequence
        self.regimes_ = self._viterbi_decode(X)
        self.regime_labels_ = [self.regime_names_[s] for s in self.regimes_]

        # Compute stationary distribution & expected durations
        self._compute_stationary_and_durations()

        self.is_fitted = True
        return self

    def _sort_and_label_states(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """
        Sort states so that regimes are consistently ordered:
        - 2-state: State 0: Bear (lowest return / highest vol), State 1: Bull
        - 3-state: State 0: Bear, State 1: Neutral, State 2: Bull
        """
        K = self.n_states
        # Order by mean return (feature 0)
        order = np.argsort(self.means_[:, 0])

        # Permute parameters according to sorted order
        self.start_prob_ = self.start_prob_[order]
        self.trans_mat_ = self.trans_mat_[order][:, order]
        self.means_ = self.means_[order]
        self.covars_ = self.covars_[order]
        self.posterior_probs_ = gamma[:, order]

        if K == 2:
            self.regime_names_ = REGIME_NAMES_2STATE
        elif K == 3:
            self.regime_names_ = REGIME_NAMES_3STATE
        else:
            self.regime_names_ = [f"Regime_{i}" for i in range(K)]

    def _viterbi_decode(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm for MAP hidden state sequence (in log-space).
        """
        T, d = X.shape
        K = self.n_states
        B = self._compute_emission_probs(X)
        log_B = np.log(np.maximum(B, 1e-300))
        log_A = np.log(np.maximum(self.trans_mat_, 1e-300))
        log_pi = np.log(np.maximum(self.start_prob_, 1e-300))

        viterbi = np.zeros((T, K))
        backpointer = np.zeros((T, K), dtype=int)

        viterbi[0] = log_pi + log_B[0]

        for t in range(1, T):
            for j in range(K):
                scores = viterbi[t - 1] + log_A[:, j]
                best_prev = int(np.argmax(scores))
                viterbi[t, j] = scores[best_prev] + log_B[t, j]
                backpointer[t, j] = best_prev

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[T - 1] = int(np.argmax(viterbi[T - 1]))

        for t in range(T - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def _compute_stationary_and_durations(self) -> None:
        """Compute the stationary distribution and expected regime durations."""
        K = self.n_states
        A = self.trans_mat_

        # Solve (A^T - I + 1 1^T) pi = 1
        M = A.T - np.eye(K) + np.ones((K, K))
        try:
            stat_dist = np.linalg.solve(M, np.ones(K))
            stat_dist = np.maximum(stat_dist, 0.0)
            stat_dist /= np.sum(stat_dist)
        except np.linalg.LinAlgError:
            stat_dist = np.full(K, 1.0 / K)

        self.stationary_dist_ = stat_dist

        # Expected duration: 1 / (1 - P_ii)
        self.expected_durations_ = {}
        for i, name in enumerate(self.regime_names_):
            p_stay = float(A[i, i])
            if p_stay < 1.0:
                duration = 1.0 / (1.0 - p_stay)
            else:
                duration = float("inf")
            self.expected_durations_[name] = duration

    def _prepare_data(
        self,
        returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[pd.Index]]:
        """Clean and extract numpy array from input returns."""
        if isinstance(returns, pd.DataFrame):
            df = returns.dropna()
            idx = df.index
            X = df.values
        elif isinstance(returns, pd.Series):
            s = returns.dropna()
            idx = s.index
            X = s.values.reshape(-1, 1)
        else:
            arr = np.asarray(returns)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            valid_mask = ~np.isnan(arr).any(axis=1)
            X = arr[valid_mask]
            idx = None

        if len(X) < 10:
            raise ValueError("Input series must contain at least 10 valid data points.")

        return X, idx

    def predict(
        self,
        returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """
        Predict the most likely hidden regime sequence (Viterbi path).

        Parameters
        ----------
        returns : pd.Series, pd.DataFrame, or np.ndarray, optional
            If None, returns fitted sequence.

        Returns
        -------
        regimes : pd.Series or np.ndarray
            Predicted regime labels or state indices.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")

        if returns is None:
            if self.index_ is not None:
                return pd.Series(self.regime_labels_, index=self.index_, name="Regime")
            return self.regimes_

        X, idx = self._prepare_data(returns)
        seq = self._viterbi_decode(X)
        labels = [self.regime_names_[s] for s in seq]

        if idx is not None:
            return pd.Series(labels, index=idx, name="Regime")
        return seq

    def predict_proba(
        self,
        returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Compute smoothed posterior regime probabilities P(S_t = k | Y).

        Returns
        -------
        posterior_probs : pd.DataFrame or np.ndarray
            Matrix of shape (T, K) with regime probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing probabilities.")

        if returns is None:
            if self.index_ is not None:
                return pd.DataFrame(
                    self.posterior_probs_,
                    index=self.index_,
                    columns=[f"Prob_{name}" for name in self.regime_names_],
                )
            return self.posterior_probs_

        X, idx = self._prepare_data(returns)
        B = self._compute_emission_probs(X)
        alpha_hat, beta_hat, c, _ = self._forward_backward(B)
        gamma = alpha_hat * beta_hat / c[:, np.newaxis]
        gamma = gamma / np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-300)

        if idx is not None:
            return pd.DataFrame(
                gamma,
                index=idx,
                columns=[f"Prob_{name}" for name in self.regime_names_],
            )
        return gamma

    def transition_matrix_df(self) -> pd.DataFrame:
        """Return transition probability matrix as a labeled DataFrame."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        return pd.DataFrame(
            self.trans_mat_,
            index=[f"From_{name}" for name in self.regime_names_],
            columns=[f"To_{name}" for name in self.regime_names_],
        )

    def regime_metrics(
        self,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute regime-conditional metrics (Annualized Return, Volatility, Sharpe Ratio).

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Annualized risk-free rate.

        Returns
        -------
        df : pd.DataFrame
            Table with regime performance statistics.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        ann = self.annualization_factor
        sqrt_ann = np.sqrt(ann)
        T = len(self.regimes_)

        metrics = []
        for state_idx, name in enumerate(self.regime_names_):
            mask = (self.regimes_ == state_idx)
            count = int(np.sum(mask))
            freq = (count / T) * 100.0 if T > 0 else 0.0

            if count > 0:
                ret_subset = self.raw_data_[mask, 0]
                daily_mean = float(np.mean(ret_subset))
                daily_vol = float(np.std(ret_subset, ddof=1)) if count > 1 else 0.0
            else:
                daily_mean = float(self.means_[state_idx, 0])
                daily_vol = float(np.sqrt(self.covars_[state_idx, 0, 0]))

            ann_return = daily_mean * ann
            ann_vol = daily_vol * sqrt_ann
            sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-12 else 0.0

            metrics.append({
                "Regime": name,
                "Observations": count,
                "Frequency_Pct": freq,
                "Stationary_Prob": float(self.stationary_dist_[state_idx]),
                "Expected_Duration_Days": self.expected_durations_[name],
                "Daily_Mean_Return": daily_mean,
                "Annualized_Return": ann_return,
                "Daily_Volatility": daily_vol,
                "Annualized_Volatility": ann_vol,
                "Sharpe_Ratio": sharpe,
            })

        return pd.DataFrame(metrics).set_index("Regime")

    def summary(self, risk_free_rate: float = 0.0) -> Dict:
        """Return comprehensive regime analysis summary dictionary."""
        return {
            "n_states": self.n_states,
            "log_likelihood": self.log_likelihood_,
            "regime_metrics": self.regime_metrics(risk_free_rate).to_dict(orient="index"),
            "transition_matrix": self.transition_matrix_df().to_dict(),
            "expected_durations": self.expected_durations_,
        }
