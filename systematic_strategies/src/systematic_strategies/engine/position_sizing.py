"""Position sizing and dynamic leverage engine for systematic strategies.

Implements:
- Fixed Fractional Sizing
- Dynamic Volatility Targeting (scaled to target annual vol)
- Kelly Criterion Sizing (Full & Fractional Kelly)
- Inverse Volatility & Equal Risk Contribution (Risk Parity) Sizing
"""

from typing import Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PositionSizer:
    """Position sizing and risk allocation framework."""

    @staticmethod
    def fixed_fractional(
        signals: Union[pd.Series, pd.DataFrame],
        fraction: float = 1.0,
        max_leverage: float = 1.0,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Scales raw directional signals (-1 to +1) by a constant fraction, capped at max_leverage."""
        weights = signals * fraction
        if isinstance(weights, pd.DataFrame):
            gross_exposure = weights.abs().sum(axis=1)
            scale = np.where(gross_exposure > max_leverage, max_leverage / np.maximum(gross_exposure, 1e-8), 1.0)
            return weights.mul(scale, axis=0)
        else:
            return np.clip(weights, -max_leverage, max_leverage)

    @staticmethod
    def volatility_targeting(
        signals: Union[pd.Series, pd.DataFrame],
        returns: Union[pd.Series, pd.DataFrame],
        target_vol: float = 0.15,
        lookback_window: int = 21,
        max_leverage: float = 2.5,
        periods_per_year: int = 252,
        min_vol_floor: float = 0.05,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Dynamically scales positions so strategy realized volatility targets target_vol.

        Formula:
            s_t = min( target_vol / max(sigma_t, min_vol_floor), max_leverage )
            w_t = signal_t * s_t

        Note: Realized volatility is calculated on lagged returns to prevent lookahead bias.
        """
        rolling_std = returns.shift(1).rolling(window=lookback_window, min_periods=max(5, lookback_window // 2)).std()
        ann_vol = rolling_std * np.sqrt(periods_per_year)
        ann_vol = ann_vol.clip(lower=min_vol_floor).fillna(target_vol)

        if isinstance(signals, pd.DataFrame) and isinstance(returns, pd.DataFrame):
            vol_scalar = target_vol / ann_vol
            vol_scalar = vol_scalar.clip(upper=max_leverage)
            raw_weights = signals * vol_scalar

            gross_lev = raw_weights.abs().sum(axis=1)
            scale = np.where(gross_lev > max_leverage, max_leverage / np.maximum(gross_lev, 1e-8), 1.0)
            return raw_weights.mul(scale, axis=0).fillna(0.0)
        else:
            vol_scalar = (target_vol / ann_vol).clip(upper=max_leverage)
            weights = (signals * vol_scalar).fillna(0.0)
            return weights.clip(-max_leverage, max_leverage)

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        win_loss_ratio: float,
        fraction: float = 0.5,
        max_leverage: float = 2.0,
    ) -> float:
        """Calculates optimal growth leverage using the Kelly formula."""
        if win_loss_ratio <= 0.0 or not (0.0 <= win_rate <= 1.0):
            return 0.0

        p = win_rate
        q = 1.0 - p
        b = win_loss_ratio

        f_star = (p * b - q) / b
        f_star = max(0.0, f_star)
        sized_f = f_star * fraction
        return float(min(sized_f, max_leverage))

    @staticmethod
    def inverse_volatility_weights(
        returns_df: pd.DataFrame,
        lookback_window: int = 63,
        target_vol: Optional[float] = None,
        max_leverage: float = 1.0,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """Computes Inverse Volatility (naive Risk Parity) weights across multiple assets."""
        rolling_vol = returns_df.shift(1).rolling(window=lookback_window, min_periods=21).std() * np.sqrt(periods_per_year)
        rolling_vol = rolling_vol.clip(lower=0.01).bfill().fillna(0.15)

        inv_vol = 1.0 / rolling_vol
        norm_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1.0 / returns_df.shape[1])

        if target_vol is not None:
            port_var = (norm_weights ** 2 * rolling_vol ** 2).sum(axis=1)
            port_vol = np.sqrt(port_var).clip(lower=0.01)
            scale = (target_vol / port_vol).clip(upper=max_leverage)
            norm_weights = norm_weights.mul(scale, axis=0)

        return norm_weights

    @staticmethod
    def equal_risk_contribution_weights(
        cov_matrix: np.ndarray,
        target_vol: Optional[float] = None,
        periods_per_year: int = 252,
    ) -> np.ndarray:
        """Calculates exact Equal Risk Contribution (Risk Parity) weights via numerical optimization."""
        n_assets = cov_matrix.shape[0]
        init_w = np.ones(n_assets) / n_assets

        def erc_objective(w: np.ndarray) -> float:
            marginal_contrib = cov_matrix @ w
            risk_contrib = w * marginal_contrib
            # Normalized risk contribution target: total_var / n
            target_rc = (w @ cov_matrix @ w) / n_assets
            return float(np.sum((risk_contrib - target_rc) ** 2)) * 1e6

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
        bounds = tuple((1e-4, 1.0) for _ in range(n_assets))

        res = minimize(erc_objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints, tol=1e-12)
        weights = res.x if res.success else init_w

        if target_vol is not None:
            ann_port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(periods_per_year)
            if ann_port_vol > 1e-4:
                weights = weights * (target_vol / ann_port_vol)

        return weights
