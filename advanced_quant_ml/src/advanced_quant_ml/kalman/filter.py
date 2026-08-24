"""Kalman Filter Engine for Dynamic Pairs Trading & Statistical Arbitrage.

Project 23: State-space recursive regression for tracking time-varying hedge ratios
and generating mean-reverting spread trading signals.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class KalmanFilterResult:
    """Stores the filtered state trajectories and innovation metrics."""
    alpha: pd.Series
    beta: pd.Series
    spread: pd.Series
    innovation_std: pd.Series
    z_score: pd.Series
    kalman_gain: pd.DataFrame
    state_cov: pd.DataFrame
    dates: pd.DatetimeIndex

    def to_dataframe(self) -> pd.DataFrame:
        """Exports state and spread trajectories to a pandas DataFrame."""
        return pd.DataFrame(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "spread": self.spread,
                "innovation_std": self.innovation_std,
                "z_score": self.z_score,
            },
            index=self.dates,
        )


@dataclass
class KalmanStrategyResult:
    """Stores backtest metrics, positions, and trades for Kalman pairs strategy."""
    positions: pd.Series
    weight_asset1: pd.Series
    weight_asset2: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    cumulative_returns: pd.Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    entries_long: pd.Series
    entries_short: pd.Series
    exits: pd.Series
    metrics: Dict[str, Any]
    filter_result: KalmanFilterResult

    def summary_table(self) -> pd.DataFrame:
        """Returns a formatted performance summary table."""
        return pd.DataFrame(
            [{"Metric": k, "Value": f"{v:+.2%}" if ("Return" in k or "Drawdown" in k or "Rate" in k or "CAGR" in k) else (f"{v:.2f}" if isinstance(v, float) else str(v))}
             for k, v in self.metrics.items()]
        )


class KalmanFilterPairs:
    """Dynamic State-Space Linear Regression via Kalman Filtering.

    Formulation:
      State Equation:       theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q)
                            theta_t = [alpha_t, beta_t]^T
      Measurement Equation: y_t     = H_t theta_t + v_t,  v_t ~ N(0, R)
                            H_t     = [1, x_t]
    """

    def __init__(
        self,
        delta: float = 1e-4,
        observation_cov: Union[float, str] = 1e-3,
        initial_state: Optional[np.ndarray] = None,
        initial_state_cov: Optional[np.ndarray] = None,
        auto_init_ols: bool = True,
    ):
        """Initializes Kalman Filter parameters.

        Args:
            delta: System process noise scaling (transition variance).
                   Q_t = (delta / (1 - delta)) * P_{t-1} or constant diag.
            observation_cov: Measurement noise variance R (or 'auto' for sample variance).
            initial_state: Initial state vector [alpha_0, beta_0]^T.
            initial_state_cov: Initial covariance matrix P_0. Defaults to 1.0 * I.
            auto_init_ols: If True and initial_state is None, initializes via early OLS window.
        """
        self.delta = delta
        self.observation_cov = observation_cov
        self.initial_state = initial_state
        self.initial_state_cov = initial_state_cov
        self.auto_init_ols = auto_init_ols

    def filter(
        self,
        y: Union[pd.Series, np.ndarray],
        x: Union[pd.Series, np.ndarray],
    ) -> KalmanFilterResult:
        """Executes recursive forward Kalman filter on asset pairs.

        Args:
            y: Dependent asset prices (Asset 1).
            x: Independent asset prices (Asset 2).

        Returns:
            KalmanFilterResult containing time-varying hedge ratios and z-scores.
        """
        if isinstance(y, pd.Series):
            dates = y.index
            y_arr = y.values.astype(float)
        else:
            y_arr = np.asarray(y, dtype=float)
            dates = pd.date_range("2020-01-01", periods=len(y_arr), freq="B")

        if isinstance(x, pd.Series):
            x_arr = x.values.astype(float)
        else:
            x_arr = np.asarray(x, dtype=float)

        n = len(y_arr)
        if len(x_arr) != n:
            raise ValueError(f"Price series lengths mismatch: len(y)={n}, len(x)={len(x_arr)}")

        # Initialize theta_0 and P_0
        if self.initial_state is not None:
            theta = np.asarray(self.initial_state, dtype=float).copy()
        elif self.auto_init_ols and n >= 20:
            # Fit initial 20 periods with OLS
            init_w = min(n, 30)
            X_init = np.column_stack([np.ones(init_w), x_arr[:init_w]])
            beta_ols = np.linalg.lstsq(X_init, y_arr[:init_w], rcond=None)[0]
            theta = beta_ols.copy()
        else:
            theta = np.array([0.0, 1.0])

        if self.initial_state_cov is not None:
            P = np.asarray(self.initial_state_cov, dtype=float).copy()
        else:
            P = np.eye(2) * 1.0

        if self.observation_cov == "auto" or self.observation_cov is None:
            # Estimate initial measurement variance from early residuals
            init_w = min(n, 30)
            H_init = np.column_stack([np.ones(init_w), x_arr[:init_w]])
            res_init = y_arr[:init_w] - np.dot(H_init, theta)
            R = float(np.var(res_init)) if np.var(res_init) > 1e-6 else 1e-3
        else:
            R = float(self.observation_cov)

        delta = self.delta

        # Storage arrays
        alpha_arr = np.zeros(n)
        beta_arr = np.zeros(n)
        spread_arr = np.zeros(n)
        innov_std_arr = np.zeros(n)
        z_score_arr = np.zeros(n)
        k_gain_arr = np.zeros((n, 2))

        for t in range(n):
            # Measurement vector H_t = [1, x_t]
            H = np.array([1.0, x_arr[t]])

            # 1. State Prediction: theta_{t|t-1} = theta_{t-1|t-1}
            # Adaptive process noise Q_t = (delta / (1 - delta)) * P_{t-1|t-1}
            Q = (delta / (1.0 - delta)) * P
            P_pred = P + Q

            # 2. Innovation / Measurement Prediction
            y_pred = np.dot(H, theta)
            e = y_arr[t] - y_pred  # Spread error
            Q_t = float(np.dot(H, np.dot(P_pred, H)) + R)  # Innovation variance

            # 3. Kalman Gain
            K = np.dot(P_pred, H) / Q_t

            # 4. State Update
            theta = theta + K * e
            P = P_pred - np.outer(K, np.dot(H, P_pred))

            # Store metrics
            alpha_arr[t] = theta[0]
            beta_arr[t] = theta[1]
            spread_arr[t] = e
            innov_std = np.sqrt(max(Q_t, 1e-12))
            innov_std_arr[t] = innov_std
            z_score_arr[t] = e / innov_std
            k_gain_arr[t] = K

        alpha_series = pd.Series(alpha_arr, index=dates, name="alpha")
        beta_series = pd.Series(beta_arr, index=dates, name="beta")
        spread_series = pd.Series(spread_arr, index=dates, name="spread")
        innov_std_series = pd.Series(innov_std_arr, index=dates, name="innovation_std")
        z_score_series = pd.Series(z_score_arr, index=dates, name="z_score")
        k_gain_df = pd.DataFrame(k_gain_arr, index=dates, columns=["K_alpha", "K_beta"])
        cov_df = pd.DataFrame({"P_00": P[0, 0], "P_11": P[1, 1]}, index=dates)

        return KalmanFilterResult(
            alpha=alpha_series,
            beta=beta_series,
            spread=spread_series,
            innovation_std=innov_std_series,
            z_score=z_score_series,
            kalman_gain=k_gain_df,
            state_cov=cov_df,
            dates=dates,
        )


class KalmanPairsStrategy:
    """Mean-Reverting Statistical Arbitrage Strategy using Kalman Z-Scores."""

    def __init__(
        self,
        z_entry: float = 1.5,
        z_exit: float = 0.4,
        stop_loss_z: float = 3.5,
        delta: float = 1e-4,
        observation_cov: Union[float, str] = 1e-3,
        transaction_cost_bps: float = 5.0,
        risk_free_rate: float = 0.02,
    ):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.stop_loss_z = stop_loss_z
        self.delta = delta
        self.observation_cov = observation_cov
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_free_rate = risk_free_rate

    def backtest(
        self,
        y: pd.Series,
        x: pd.Series,
    ) -> KalmanStrategyResult:
        """Executes full backtest of the Kalman Pairs Strategy."""
        kf = KalmanFilterPairs(delta=self.delta, observation_cov=self.observation_cov)
        kf_res = kf.filter(y, x)

        z = kf_res.z_score.values
        beta = kf_res.beta.values
        n = len(z)
        dates = kf_res.dates

        positions = np.zeros(n)
        entries_long = np.zeros(n, dtype=bool)
        entries_short = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)

        current_pos = 0

        for t in range(1, n):
            z_val = z[t]

            if current_pos == 0:
                if z_val <= -self.z_entry:
                    current_pos = 1  # Long spread: Buy y, Short x
                    entries_long[t] = True
                elif z_val >= self.z_entry:
                    current_pos = -1  # Short spread: Short y, Buy x
                    entries_short[t] = True
            elif current_pos == 1:
                # Exit long spread on mean reversion or stop-loss
                if z_val >= -self.z_exit or z_val <= -self.stop_loss_z:
                    current_pos = 0
                    exits[t] = True
            elif current_pos == -1:
                # Exit short spread on mean reversion or stop-loss
                if z_val <= self.z_exit or z_val >= self.stop_loss_z:
                    current_pos = 0
                    exits[t] = True

            positions[t] = current_pos

        pos_series = pd.Series(positions, index=dates, name="Position")

        # Dollar-weighted allocations
        denom = 1.0 + np.abs(beta)
        w1 = (positions * (1.0 / denom))
        w2 = (positions * (-beta / denom))

        w1_series = pd.Series(w1, index=dates, name="Weight_Asset1")
        w2_series = pd.Series(w2, index=dates, name="Weight_Asset2")

        # Returns calculation (lagged execution to eliminate lookahead bias)
        r_y = y.pct_change().fillna(0.0)
        r_x = x.pct_change().fillna(0.0)

        gross_returns = w1_series.shift(1).fillna(0.0) * r_y + w2_series.shift(1).fillna(0.0) * r_x

        # Transaction costs
        dw1 = w1_series.diff().abs().fillna(0.0)
        dw2 = w2_series.diff().abs().fillna(0.0)
        turnover = dw1 + dw2
        costs = turnover * (self.transaction_cost_bps / 10000.0)
        net_returns = gross_returns - costs

        cum_returns = (1.0 + net_returns).cumprod() - 1.0
        equity_curve = 100000.0 * (1.0 + cum_returns)

        # Drawdowns
        hwm = equity_curve.cummax()
        drawdowns = (equity_curve - hwm) / hwm
        max_dd = drawdowns.min()

        # Performance metrics
        total_ret = cum_returns.iloc[-1]
        n_years = max(n / 252.0, 0.1)
        cagr = (1.0 + total_ret) ** (1.0 / n_years) - 1.0
        ann_vol = net_returns.std() * np.sqrt(252)
        sharpe = (cagr - self.risk_free_rate) / max(ann_vol, 1e-6)
        downside_vol = net_returns[net_returns < 0].std() * np.sqrt(252)
        sortino = (cagr - self.risk_free_rate) / max(downside_vol, 1e-6)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
        win_rate = (net_returns[positions != 0] > 0).mean() if (positions != 0).any() else 0.0

        metrics = {
            "Total Return": total_ret,
            "CAGR": cagr,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_dd,
            "Win Rate": win_rate,
            "Total Trades": int(entries_long.sum() + entries_short.sum()),
            "Annualized Turnover": turnover.sum() / n_years,
        }

        return KalmanStrategyResult(
            positions=pos_series,
            weight_asset1=w1_series,
            weight_asset2=w2_series,
            gross_returns=gross_returns,
            net_returns=net_returns,
            cumulative_returns=cum_returns,
            equity_curve=equity_curve,
            drawdown_series=drawdowns,
            entries_long=pd.Series(entries_long, index=dates),
            entries_short=pd.Series(entries_short, index=dates),
            exits=pd.Series(exits, index=dates),
            metrics=metrics,
            filter_result=kf_res,
        )
