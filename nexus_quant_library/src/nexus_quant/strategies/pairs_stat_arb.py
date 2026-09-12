"""
Strategies: Statistical Arbitrage, Cointegration & Online Kalman Filter Pairs Trading.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import scipy.stats as stats


class PairsStatArbEngine:
    """Kalman Filter dynamic hedge ratio estimation and statistical arbitrage spread trading."""

    def __init__(self, delta: float = 1e-4, observation_noise: float = 1e-3):
        self.delta = delta
        self.observation_noise = observation_noise

    def fit_kalman_filter(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        Online state-space Kalman Filter:
        State: [alpha_t, beta_t]^T
        Measurement: y_t = alpha_t + beta_t * x_t + v_t
        """
        n = len(y)
        y_vals = y.values
        x_vals = x.values

        # State covariance P and noise Q
        P = np.eye(2) * 1.0
        Q = np.eye(2) * self.delta / (1.0 - self.delta)
        R = self.observation_noise

        state = np.zeros(2) # [alpha, beta]
        alphas = np.zeros(n)
        betas = np.zeros(n)
        spreads = np.zeros(n)
        variances = np.zeros(n)

        for t in range(n):
            # Measurement matrix H = [1, x_t]
            H = np.array([1.0, x_vals[t]])

            # Prediction
            P = P + Q

            # Innovation
            y_pred = H @ state
            error = y_vals[t] - y_pred
            Q_t = float(H @ P @ H.T + R)

            # Kalman gain
            K = (P @ H) / Q_t

            # Update
            state = state + K * error
            P = P - np.outer(K, H @ P)

            alphas[t] = state[0]
            betas[t] = state[1]
            spreads[t] = error
            variances[t] = Q_t

        z_scores = spreads / np.sqrt(np.maximum(variances, 1e-8))

        return pd.DataFrame({
            "Alpha": alphas,
            "Beta": betas,
            "Spread": spreads,
            "Z_Score": z_scores,
        }, index=y.index)

    def generate_trading_positions(
        self, y: pd.Series, x: pd.Series, z_entry: float = 1.8, z_exit: float = 0.3
    ) -> pd.DataFrame:
        """Generate dollar-neutral pair positions (Long Spread vs Short Spread)."""
        kf_df = self.fit_kalman_filter(y, x)
        z = kf_df["Z_Score"]
        betas = kf_df["Beta"]

        pos_y = pd.Series(index=y.index, data=0.0)
        pos_x = pd.Series(index=x.index, data=0.0)

        curr_state = 0 # -1 short spread, +1 long spread
        for t in range(len(z)):
            val = z.iloc[t]
            b = betas.iloc[t]
            if val <= -z_entry:
                curr_state = 1
            elif val >= z_entry:
                curr_state = -1
            elif abs(val) <= z_exit:
                curr_state = 0

            if curr_state == 1:
                # Long Asset Y, Short beta * Asset X
                norm = 1.0 + abs(b)
                pos_y.iloc[t] = 1.0 / norm
                pos_x.iloc[t] = -b / norm
            elif curr_state == -1:
                # Short Asset Y, Long beta * Asset X
                norm = 1.0 + abs(b)
                pos_y.iloc[t] = -1.0 / norm
                pos_x.iloc[t] = b / norm

        return pd.DataFrame({
            "Weight_Y": pos_y,
            "Weight_X": pos_x,
            "Spread_Z": z,
            "Beta": betas,
        }, index=y.index)
