"""
Foundations: Gaussian Hidden Markov Model (HMM) & Regime Detection Models.
"""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


class GaussianHMMRegimeDetector:
    """
    Gaussian Hidden Markov Model for market regime classification.
    Implements Baum-Welch (EM) parameter estimation and Viterbi state decoding.
    States are sorted monotonically by return/volatility:
    - State 0: Bear (Low return / High volatility)
    - State 1: Neutral (Moderate return / Moderate volatility)
    - State 2: Bull (High return / Low volatility)
    """

    def __init__(self, n_states: int = 3, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.means_: Optional[np.ndarray] = None
        self.covars_: Optional[np.ndarray] = None
        self.trans_mat_: Optional[np.ndarray] = None
        self.start_prob_: Optional[np.ndarray] = None

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "GaussianHMMRegimeDetector":
        """Fit Gaussian HMM to historical returns using Baum-Welch EM algorithm."""
        rng = np.random.RandomState(self.seed)
        x = np.asarray(returns, dtype=float).flatten()
        x = x[~np.isnan(x)]
        n = len(x)

        # Initialize parameters
        quantiles = np.linspace(0, 1, self.n_states + 1)
        self.means_ = np.array([np.quantile(x, (quantiles[i] + quantiles[i+1])/2) for i in range(self.n_states)])
        self.covars_ = np.full(self.n_states, np.var(x))
        self.trans_mat_ = np.full((self.n_states, self.n_states), 0.1 / (self.n_states - 1))
        np.fill_diagonal(self.trans_mat_, 0.9)
        self.start_prob_ = np.full(self.n_states, 1.0 / self.n_states)

        prev_ll = -np.inf

        for it in range(self.max_iter):
            # Emission probabilities B[t, k] = N(x_t; mu_k, sigma_k)
            B = np.zeros((n, self.n_states))
            for k in range(self.n_states):
                sd = np.sqrt(np.maximum(self.covars_[k], 1e-8))
                B[:, k] = stats.norm.pdf(x, loc=self.means_[k], scale=sd)
            B = np.maximum(B, 1e-300)

            # Forward pass (scaled)
            alpha = np.zeros((n, self.n_states))
            c = np.zeros(n)
            alpha[0] = self.start_prob_ * B[0]
            c[0] = np.sum(alpha[0])
            alpha[0] /= c[0]

            for t in range(1, n):
                alpha[t] = (alpha[t-1] @ self.trans_mat_) * B[t]
                c[t] = np.sum(alpha[t])
                if c[t] == 0:
                    c[t] = 1e-300
                alpha[t] /= c[t]

            log_likelihood = np.sum(np.log(c))

            # Backward pass (scaled)
            beta = np.zeros((n, self.n_states))
            beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                beta[t] = self.trans_mat_ @ (B[t+1] * beta[t+1]) / c[t+1]

            # Posteriors gamma and xi
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True)

            xi = np.zeros((n - 1, self.n_states, self.n_states))
            for t in range(n - 1):
                numerator = alpha[t, :, None] * self.trans_mat_ * B[t+1, None, :] * beta[t+1, None, :]
                xi[t] = numerator / np.sum(numerator)

            # M-step: Update parameters
            self.start_prob_ = gamma[0]
            self.trans_mat_ = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]
            self.trans_mat_ /= np.sum(self.trans_mat_, axis=1, keepdims=True)

            for k in range(self.n_states):
                weight = np.sum(gamma[:, k])
                if weight > 1e-8:
                    self.means_[k] = np.sum(gamma[:, k] * x) / weight
                    self.covars_[k] = np.sum(gamma[:, k] * (x - self.means_[k])**2) / weight
                    self.covars_[k] = max(self.covars_[k], 1e-8)

            if abs(log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = log_likelihood

        # Monotonic sorting: Sort states by mean return ascending
        sort_idx = np.argsort(self.means_)
        self.means_ = self.means_[sort_idx]
        self.covars_ = self.covars_[sort_idx]
        self.start_prob_ = self.start_prob_[sort_idx]
        self.trans_mat_ = self.trans_mat_[sort_idx][:, sort_idx]

        return self

    def predict(self, returns: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Decode most likely sequence of hidden states via Viterbi algorithm."""
        x = np.asarray(returns, dtype=float).flatten()
        n = len(x)
        if n == 0 or self.means_ is None:
            return np.array([])

        B = np.zeros((n, self.n_states))
        for k in range(self.n_states):
            sd = np.sqrt(self.covars_[k])
            B[:, k] = stats.norm.pdf(x, loc=self.means_[k], scale=sd)
        B = np.maximum(B, 1e-300)

        viterbi = np.zeros((n, self.n_states))
        backpointer = np.zeros((n, self.n_states), dtype=int)

        viterbi[0] = np.log(np.maximum(self.start_prob_, 1e-300)) + np.log(B[0])

        for t in range(1, n):
            for k in range(self.n_states):
                trans_prob = viterbi[t-1] + np.log(np.maximum(self.trans_mat_[:, k], 1e-300))
                best_prev = np.argmax(trans_prob)
                viterbi[t, k] = trans_prob[best_prev] + np.log(B[t, k])
                backpointer[t, k] = best_prev

        # Backtrack
        path = np.zeros(n, dtype=int)
        path[-1] = np.argmax(viterbi[-1])
        for t in range(n - 2, -1, -1):
            path[t] = backpointer[t+1, path[t+1]]

        return path


class TrendVolRegimeFilter:
    """Rule-based regime filter combining 200d SMA trend and 21d rolling volatility."""

    @staticmethod
    def classify(prices: pd.Series, sma_window: int = 200, vol_window: int = 21) -> pd.Series:
        sma = prices.rolling(sma_window).mean()
        ret = prices.pct_change()
        vol = ret.rolling(vol_window).std() * np.sqrt(252)
        median_vol = vol.rolling(sma_window).median()

        regime = pd.Series(index=prices.index, data="NEUTRAL")
        bull_mask = (prices > sma) & (vol <= median_vol)
        bear_mask = (prices < sma) & (vol > median_vol)

        regime[bull_mask] = "BULL"
        regime[bear_mask] = "BEAR"
        return regime
