"""Project 13: Statistical Arbitrage & Pairs Trading Strategy.

Implements econometric cointegration modeling, dynamic Kalman Filter hedge ratio tracking,
Ornstein-Uhlenbeck (OU) process mean-reversion parameter estimation, and dollar-neutral
spread execution.
"""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CointegrationTestResult:
    """Container for Engle-Granger Cointegration Test Results."""
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_cointegrated: bool
    hedge_ratio_static: float
    intercept_static: float
    residuals: pd.Series


@dataclass
class OrnsteinUhlenbeckParams:
    """Container for Ornstein-Uhlenbeck (OU) SDE Parameters."""
    reversion_speed_theta: float
    long_term_mean_mu: float
    volatility_sigma: float
    half_life_days: float
    r_squared: float


@dataclass
class PairsTradingResult:
    """Container for complete Pairs Trading strategy output."""
    price1: pd.Series
    price2: pd.Series
    hedge_ratio: pd.Series
    spread: pd.Series
    spread_mean: pd.Series
    spread_std: pd.Series
    z_score: pd.Series
    pair_position: pd.Series
    asset1_position: pd.Series
    asset2_position: pd.Series
    dollar_weight1: pd.Series
    dollar_weight2: pd.Series
    trade_action: pd.Series
    entries_long_spread: pd.Series
    entries_short_spread: pd.Series
    exits: pd.Series
    ou_params: Optional[OrnsteinUhlenbeckParams] = None
    coint_test: Optional[CointegrationTestResult] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Combines all pairs series into a unified pandas DataFrame."""
        return pd.DataFrame({
            'Price_1': self.price1,
            'Price_2': self.price2,
            'Hedge_Ratio': self.hedge_ratio,
            'Spread': self.spread,
            'Spread_Mean': self.spread_mean,
            'Spread_Std': self.spread_std,
            'Z_Score': self.z_score,
            'Pair_Position': self.pair_position,
            'Asset1_Position': self.asset1_position,
            'Asset2_Position': self.asset2_position,
            'Weight_1': self.dollar_weight1,
            'Weight_2': self.dollar_weight2,
            'Trade_Action': self.trade_action,
            'Entry_Long_Spread': self.entries_long_spread,
            'Entry_Short_Spread': self.entries_short_spread,
            'Exit': self.exits,
        })


def adf_unit_root_test(series: pd.Series, max_lags: int = 1) -> Tuple[float, float, Dict[str, float]]:
    """Performs Augmented Dickey-Fuller (ADF) test on a time series.

    Regression model: \Delta y_t = lpha + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + e_t
    H0: \gamma = 0 (unit root / non-stationary)
    H1: \gamma < 0 (stationary)

    Returns
    -------
    Tuple[float, float, Dict[str, float]]
        (ADF test statistic, approximate p-value, critical values dictionary)
    """
    y = np.asarray(series.dropna(), dtype=float)
    n = len(y)
    if n < 15:
        return 0.0, 1.0, {'1%': -3.5, '5%': -2.89, '10%': -2.58}

    dy = np.diff(y)
    y_lag = y[:-1]

    if max_lags > 0:
        k = max_lags
        n_obs = len(dy) - k
        cols = [np.ones(n_obs), y_lag[k:]]
        for i in range(1, k + 1):
            cols.append(dy[k - i : -i])
        X = np.column_stack(cols)
        Y = dy[k:]
    else:
        n_obs = len(dy)
        X = np.column_stack([np.ones(n_obs), y_lag])
        Y = dy

    # OLS estimation
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        dof = n_obs - X.shape[1]
        sigma2 = np.sum(res**2) / max(dof, 1)
        var_cov = sigma2 * np.linalg.inv(X.T @ X)
        gamma_se = np.sqrt(max(var_cov[1, 1], 1e-12))
        gamma_hat = beta[1]
        adf_stat = gamma_hat / gamma_se
    except Exception:
        adf_stat = 0.0

    # MacKinnon (1994, 2010) critical values approximation with constant (no trend)
    crit_vals = {
        '1%': -3.434 - 1.798 / n,
        '5%': -2.863 - 1.000 / n,
        '10%': -2.568 - 0.500 / n,
    }

    # MacKinnon approximate p-value interpolation
    if adf_stat <= crit_vals['1%']:
        p_val = max(0.001, 0.01 * (1.0 + (adf_stat - crit_vals['1%']) / crit_vals['1%']))
    elif adf_stat <= crit_vals['5%']:
        p_val = 0.01 + 0.04 * ((adf_stat - crit_vals['1%']) / (crit_vals['5%'] - crit_vals['1%']))
    elif adf_stat <= crit_vals['10%']:
        p_val = 0.05 + 0.05 * ((adf_stat - crit_vals['5%']) / (crit_vals['10%'] - crit_vals['5%']))
    else:
        p_val = min(0.999, 0.10 + 0.90 * max(0.0, (adf_stat - crit_vals['10%']) / (0.0 - crit_vals['10%'])))

    return float(adf_stat), float(p_val), crit_vals


def engle_granger_cointegration_test(
    p1: pd.Series,
    p2: pd.Series,
    significance_level: float = 0.05,
) -> CointegrationTestResult:
    """Performs Engle-Granger Two-Step Cointegration Test on two price series.

    Step 1: OLS regression P_1 = lpha + eta P_2 + \epsilon
    Step 2: ADF test on residuals \epsilon
    """
    df = pd.DataFrame({'P1': p1, 'P2': p2}).dropna()
    y = df['P1'].values
    x = df['P2'].values

    # OLS regression
    slope, intercept, r_val, p_val_reg, std_err = stats.linregress(x, y)
    residuals = pd.Series(y - (intercept + slope * x), index=df.index, name='Residuals')

    adf_stat, p_val_adf, crit_vals = adf_unit_root_test(residuals, max_lags=1)
    is_coint = bool(p_val_adf <= significance_level or adf_stat <= crit_vals['5%'])

    return CointegrationTestResult(
        adf_statistic=adf_stat,
        p_value=p_val_adf,
        critical_values=crit_vals,
        is_cointegrated=is_coint,
        hedge_ratio_static=float(slope),
        intercept_static=float(intercept),
        residuals=residuals,
    )


def fit_ornstein_uhlenbeck(spread: pd.Series, dt: float = 1.0) -> OrnsteinUhlenbeckParams:
    """Fits an Ornstein-Uhlenbeck (OU) Mean Reversion SDE to the spread series.

    dS_t = 	heta (\mu - S_t) dt + \sigma dW_t
    Discrete AR(1): S_t = a + b S_{t-1} + \eta_t
    where b = exp(-	heta dt) pprox 1 - 	heta dt, a = \mu(1 - b).
    Half-life: t_{1/2} = ln(2) / 	heta
    """
    s = spread.dropna().values
    n = len(s)
    if n < 10:
        return OrnsteinUhlenbeckParams(0.1, 0.0, 1.0, 6.93, 0.0)

    s_curr = s[1:]
    s_prev = s[:-1]

    slope, intercept, r_val, p_val, std_err = stats.linregress(s_prev, s_curr)
    residuals = s_curr - (intercept + slope * s_prev)

    b = float(np.clip(slope, 1e-6, 0.9999))
    theta = -np.log(b) / dt
    mu = float(intercept / (1.0 - b))
    half_life = float(np.log(2.0) / max(theta, 1e-6))

    res_var = float(np.var(residuals, ddof=2))
    sigma = float(np.sqrt(res_var * (2.0 * theta) / (1.0 - b**2 + 1e-12)))

    return OrnsteinUhlenbeckParams(
        reversion_speed_theta=theta,
        long_term_mean_mu=mu,
        volatility_sigma=sigma,
        half_life_days=half_life,
        r_squared=float(r_val**2),
    )


def kalman_filter_hedge_ratio(
    p1: pd.Series,
    p2: pd.Series,
    delta: float = 1e-4,
    vt: float = 1e-3,
) -> Tuple[pd.Series, pd.Series]:
    """Computes dynamic, real-time hedge ratio and intercept via Kalman Filter.

    State vector: x_t = [intercept_t, beta_t]^T
    State equation: x_t = x_{t-1} + w_t,  w_t ~ N(0, Q_t)
    Measurement equation: y_t = H_t x_t + v_t,  v_t ~ N(0, R)
    where H_t = [1, P_{2,t}].
    """
    df = pd.DataFrame({'P1': p1, 'P2': p2}).dropna()
    y = df['P1'].values
    x = df['P2'].values
    n = len(y)

    state_mean = np.array([0.0, 1.0])
    state_cov = np.eye(2) * 1.0

    Q = delta / (1.0 - delta) * np.eye(2)
    R = vt

    betas = np.zeros(n)
    intercepts = np.zeros(n)

    for t in range(n):
        H = np.array([[1.0, x[t]]])

        # Prediction Step
        state_cov_pred = state_cov + Q

        # Update Step
        y_hat = float(np.squeeze(H @ state_mean))
        error = y[t] - y_hat
        F = float(np.squeeze(H @ state_cov_pred @ H.T + R))
        K = state_cov_pred @ H.T / F

        state_mean = state_mean + (K * error).flatten()
        state_cov = state_cov_pred - K @ H @ state_cov_pred

        intercepts[t] = state_mean[0]
        betas[t] = state_mean[1]

    return (
        pd.Series(betas, index=df.index, name='Kalman_Beta'),
        pd.Series(intercepts, index=df.index, name='Kalman_Alpha'),
    )


class PairsTradingStrategy:
    """Project 13: Statistical Arbitrage & Cointegration Pairs Trading Strategy.

    Constructs a market-neutral synthetic spread between two cointegrated assets,
    calculates dynamic hedge ratios via rolling OLS or Kalman Filter, models
    Ornstein-Uhlenbeck mean reversion dynamics, and generates long/short spread orders.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        hedge_method: str = 'ols',
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        stop_loss_z: float = 3.5,
        adf_significance: float = 0.05,
    ) -> None:
        if z_entry <= z_exit:
            raise ValueError(f"z_entry ({z_entry}) must be strictly greater than z_exit ({z_exit})")
        if lookback_window < 10:
            raise ValueError(f"lookback_window must be >= 10, got {lookback_window}")

        self.lookback_window = lookback_window
        self.hedge_method = hedge_method.lower()
        self.z_entry = float(z_entry)
        self.z_exit = float(z_exit)
        self.stop_loss_z = float(stop_loss_z)
        self.adf_significance = float(adf_significance)

    def test_cointegration(self, p1: pd.Series, p2: pd.Series) -> CointegrationTestResult:
        """Executes Engle-Granger cointegration test."""
        return engle_granger_cointegration_test(p1, p2, significance_level=self.adf_significance)

    def compute_hedge_ratio(self, p1: pd.Series, p2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculates time-varying hedge ratio (beta) and intercept (alpha)."""
        df = pd.DataFrame({'P1': p1, 'P2': p2}).dropna()

        if self.hedge_method == 'kalman':
            return kalman_filter_hedge_ratio(df['P1'], df['P2'])

        # Rolling OLS regression
        cov_12 = df['P1'].rolling(window=self.lookback_window).cov(df['P2'])
        var_2 = df['P2'].rolling(window=self.lookback_window).var()
        rolling_beta = (cov_12 / var_2.replace(0, np.nan)).bfill().fillna(1.0)
        rolling_alpha = (df['P1'] - rolling_beta * df['P2']).rolling(window=self.lookback_window).mean().fillna(0.0)

        return rolling_beta, rolling_alpha

    def compute_spread_and_zscore(
        self,
        p1: pd.Series,
        p2: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculates the synthetic spread and its normalized rolling Z-score."""
        beta, alpha = self.compute_hedge_ratio(p1, p2)
        spread = p1 - (beta * p2)

        spread_mean = spread.rolling(window=self.lookback_window, min_periods=self.lookback_window).mean()
        spread_std = spread.rolling(window=self.lookback_window, min_periods=self.lookback_window).std(ddof=1)
        z_score = (spread - spread_mean) / spread_std.replace(0, np.nan)

        return spread, beta, spread_mean, spread_std, z_score

    def generate_signals(
        self,
        p1: Union[pd.Series, np.ndarray],
        p2: Union[pd.Series, np.ndarray],
    ) -> PairsTradingResult:
        """Generates pairs trading positions, dollar weights, and trade actions."""
        p1_s = pd.Series(p1) if isinstance(p1, np.ndarray) else p1.copy()
        p2_s = pd.Series(p2) if isinstance(p2, np.ndarray) else p2.copy()

        coint_res = self.test_cointegration(p1_s, p2_s)
        spread, beta, spread_mean, spread_std, z_score = self.compute_spread_and_zscore(p1_s, p2_s)
        ou_params = fit_ornstein_uhlenbeck(spread)

        n = len(p1_s)
        index = p1_s.index
        z_arr = z_score.values
        beta_arr = beta.values

        pair_positions = np.zeros(n, dtype=float)
        asset1_positions = np.zeros(n, dtype=float)
        asset2_positions = np.zeros(n, dtype=float)
        weight1 = np.zeros(n, dtype=float)
        weight2 = np.zeros(n, dtype=float)
        trade_actions = ['HOLD'] * n
        entries_long = np.zeros(n, dtype=bool)
        entries_short = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)

        current_pair_pos = 0.0

        for t in range(n):
            z = z_arr[t]
            b = beta_arr[t] if not np.isnan(beta_arr[t]) else 1.0

            if np.isnan(z):
                pair_positions[t] = 0.0
                continue

            if current_pair_pos == 0.0:
                if z <= -self.z_entry:
                    current_pair_pos = 1.0  # Long Spread
                    trade_actions[t] = 'BUY_SPREAD'
                    entries_long[t] = True
                elif z >= self.z_entry:
                    current_pair_pos = -1.0  # Short Spread
                    trade_actions[t] = 'SELL_SPREAD'
                    entries_short[t] = True

            elif current_pair_pos == 1.0:
                if z >= -self.z_exit or z <= -self.stop_loss_z:
                    current_pair_pos = 0.0
                    trade_actions[t] = 'EXIT_LONG_SPREAD'
                    exits[t] = True
                elif z >= self.z_entry:
                    current_pair_pos = -1.0
                    trade_actions[t] = 'REVERSE_SHORT_SPREAD'
                    entries_short[t] = True

            elif current_pair_pos == -1.0:
                if z <= self.z_exit or z >= self.stop_loss_z:
                    current_pair_pos = 0.0
                    trade_actions[t] = 'EXIT_SHORT_SPREAD'
                    exits[t] = True
                elif z <= -self.z_entry:
                    current_pair_pos = 1.0
                    trade_actions[t] = 'REVERSE_LONG_SPREAD'
                    entries_long[t] = True

            pair_positions[t] = current_pair_pos
            asset1_positions[t] = current_pair_pos
            asset2_positions[t] = -current_pair_pos * b

            denom = 1.0 + abs(b) if (1.0 + abs(b)) > 0 else 1.0
            if current_pair_pos == 1.0:
                weight1[t] = 1.0 / denom
                weight2[t] = -abs(b) / denom
            elif current_pair_pos == -1.0:
                weight1[t] = -1.0 / denom
                weight2[t] = abs(b) / denom
            else:
                weight1[t] = 0.0
                weight2[t] = 0.0

        return PairsTradingResult(
            price1=p1_s,
            price2=p2_s,
            hedge_ratio=beta,
            spread=spread,
            spread_mean=spread_mean,
            spread_std=spread_std,
            z_score=z_score,
            pair_position=pd.Series(pair_positions, index=index, name='Pair_Position'),
            asset1_position=pd.Series(asset1_positions, index=index, name='Asset1_Position'),
            asset2_position=pd.Series(asset2_positions, index=index, name='Asset2_Position'),
            dollar_weight1=pd.Series(weight1, index=index, name='Weight_1'),
            dollar_weight2=pd.Series(weight2, index=index, name='Weight_2'),
            trade_action=pd.Series(trade_actions, index=index, name='Trade_Action'),
            entries_long_spread=pd.Series(entries_long, index=index, name='Entry_Long_Spread'),
            entries_short_spread=pd.Series(entries_short, index=index, name='Entry_Short_Spread'),
            exits=pd.Series(exits, index=index, name='Exit'),
            ou_params=ou_params,
            coint_test=coint_res,
        )
