"""Historical Simulation Value-at-Risk (VaR) and Expected Shortfall (CVaR) Engine."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class HistoricalVaRCalculator:
    """Calculates empirical and age-weighted Historical Value-at-Risk (VaR) and CVaR.
    
    Includes:
    - Standard empirical quantile VaR / CVaR
    - Rolling window historical VaR
    - Age-weighted historical simulation (Boudoukh et al., 1998)
    - Bootstrap confidence intervals for VaR estimates
    """
    
    def __init__(self, returns: Union[pd.Series, pd.DataFrame, np.ndarray, List[float]]) -> None:
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] == 1:
                self.returns = returns.iloc[:, 0].dropna()
            else:
                raise ValueError("HistoricalVaRCalculator requires a 1D return series. Passed DataFrame with multiple columns.")
        elif isinstance(returns, pd.Series):
            self.returns = returns.dropna()
        else:
            arr = np.asarray(returns, dtype=float)
            arr = arr[~np.isnan(arr)]
            self.returns = pd.Series(arr)
            
        if len(self.returns) == 0:
            raise ValueError("Returns series cannot be empty.")
            
        self.values = self.returns.values.astype(float)
        self.n_obs = len(self.values)

    def compute_var(
        self,
        confidence_level: Union[float, List[float]] = 0.95,
        as_loss: bool = True,
    ) -> Union[float, Dict[float, float]]:
        """Computes historical empirical quantile Value-at-Risk.
        
        Args:
            confidence_level: Confidence level (e.g. 0.95, 0.99) or list of levels.
            as_loss: If True, returns positive loss magnitude. If False, returns negative return quantile.
            
        Returns:
            VaR value or dictionary mapping confidence level to VaR.
        """
        if isinstance(confidence_level, (list, tuple, np.ndarray)):
            return {cl: self._single_var(cl, as_loss=as_loss) for cl in confidence_level}
        return self._single_var(confidence_level, as_loss=as_loss)

    def _single_var(self, confidence_level: float, as_loss: bool = True) -> float:
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        alpha = 1.0 - confidence_level
        # Empirical quantile at alpha
        quantile_val = float(np.percentile(self.values, alpha * 100.0))
        return -quantile_val if as_loss else quantile_val

    def compute_cvar(
        self,
        confidence_level: Union[float, List[float]] = 0.95,
        as_loss: bool = True,
    ) -> Union[float, Dict[float, float]]:
        """Computes historical Expected Shortfall / Conditional VaR (CVaR).
        
        E[R | R <= VaR_cutoff]
        """
        if isinstance(confidence_level, (list, tuple, np.ndarray)):
            return {cl: self._single_cvar(cl, as_loss=as_loss) for cl in confidence_level}
        return self._single_cvar(confidence_level, as_loss=as_loss)

    def _single_cvar(self, confidence_level: float, as_loss: bool = True) -> float:
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        alpha = 1.0 - confidence_level
        cutoff = np.percentile(self.values, alpha * 100.0)
        tail_losses = self.values[self.values <= cutoff]
        if len(tail_losses) == 0:
            cvar_val = cutoff
        else:
            cvar_val = float(np.mean(tail_losses))
        return -cvar_val if as_loss else cvar_val

    def rolling_var(
        self,
        window: int = 252,
        confidence_level: float = 0.95,
        as_loss: bool = True,
    ) -> pd.Series:
        """Computes rolling historical VaR over a sliding window.
        
        Args:
            window: Number of trading days in rolling window (e.g. 252).
            confidence_level: Confidence level (e.g. 0.95).
            as_loss: If True, returns positive loss.
        """
        if window > self.n_obs:
            raise ValueError(f"Window size {window} exceeds total observations {self.n_obs}")
        alpha = 1.0 - confidence_level
        rolling_q = self.returns.rolling(window=window).quantile(alpha)
        return -rolling_q if as_loss else rolling_q

    def rolling_cvar(
        self,
        window: int = 252,
        confidence_level: float = 0.95,
        as_loss: bool = True,
    ) -> pd.Series:
        """Computes rolling historical CVaR over a sliding window."""
        if window > self.n_obs:
            raise ValueError(f"Window size {window} exceeds total observations {self.n_obs}")
        alpha = 1.0 - confidence_level
        
        def _window_cvar(sub_series):
            arr = sub_series.values
            cutoff = np.percentile(arr, alpha * 100.0)
            tail = arr[arr <= cutoff]
            val = np.mean(tail) if len(tail) > 0 else cutoff
            return -val if as_loss else val
            
        return self.returns.rolling(window=window).apply(_window_cvar, raw=False)

    def age_weighted_var(
        self,
        confidence_level: float = 0.95,
        decay_factor: float = 0.98,
        as_loss: bool = True,
    ) -> float:
        """Age-weighted Historical Simulation VaR (Boudoukh, Richardson, & Whitelaw, 1998).
        
        Applies exponentially decaying weights w_t = lambda^(T-t) * (1-lambda) / (1-lambda^T)
        to historical observations, placing higher probability mass on recent shocks.
        """
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"Decay factor must be in (0, 1), got {decay_factor}")
            
        T = self.n_obs
        t_indices = np.arange(1, T + 1) # t=1 (oldest) to t=T (most recent)
        
        # Exponential weights
        raw_weights = (decay_factor ** (T - t_indices)) * (1.0 - decay_factor) / (1.0 - decay_factor ** T)
        weights = raw_weights / np.sum(raw_weights)
        
        # Sort returns and corresponding weights
        sort_order = np.argsort(self.values)
        sorted_returns = self.values[sort_order]
        sorted_weights = weights[sort_order]
        
        cum_weights = np.cumsum(sorted_weights)
        alpha = 1.0 - confidence_level
        
        # Find interpolation point for cumulative weight == alpha
        if alpha <= cum_weights[0]:
            var_val = sorted_returns[0]
        else:
            idx = np.searchsorted(cum_weights, alpha)
            if idx >= len(sorted_returns):
                var_val = sorted_returns[-1]
            else:
                # Linear interpolation
                w0, w1 = cum_weights[idx - 1], cum_weights[idx]
                r0, r1 = sorted_returns[idx - 1], sorted_returns[idx]
                fraction = (alpha - w0) / (w1 - w0) if w1 > w0 else 0.0
                var_val = r0 + fraction * (r1 - r0)
                
        return float(-var_val if as_loss else var_val)

    def age_weighted_cvar(
        self,
        confidence_level: float = 0.95,
        decay_factor: float = 0.98,
        as_loss: bool = True,
    ) -> float:
        """Age-weighted Historical Simulation CVaR (Expected Shortfall)."""
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"Decay factor must be in (0, 1), got {decay_factor}")
            
        T = self.n_obs
        t_indices = np.arange(1, T + 1)
        raw_weights = (decay_factor ** (T - t_indices)) * (1.0 - decay_factor) / (1.0 - decay_factor ** T)
        weights = raw_weights / np.sum(raw_weights)
        
        sort_order = np.argsort(self.values)
        sorted_returns = self.values[sort_order]
        sorted_weights = weights[sort_order]
        
        cum_weights = np.cumsum(sorted_weights)
        alpha = 1.0 - confidence_level
        
        tail_mask = cum_weights <= alpha
        if not np.any(tail_mask):
            cvar_val = sorted_returns[0]
        else:
            tail_returns = sorted_returns[tail_mask]
            tail_w = sorted_weights[tail_mask]
            cvar_val = np.sum(tail_returns * tail_w) / np.sum(tail_w)
            
        return float(-cvar_val if as_loss else cvar_val)

    def bootstrap_confidence_interval(
        self,
        confidence_level: float = 0.95,
        ci_level: float = 0.95,
        n_bootstraps: int = 1000,
        random_state: int = 42,
        as_loss: bool = True,
    ) -> Tuple[float, float, float]:
        """Calculates bootstrap confidence intervals for historical VaR.
        
        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        point_estimate = self.compute_var(confidence_level, as_loss=as_loss)
        
        rng = np.random.default_rng(random_state)
        alpha = 1.0 - confidence_level
        
        boot_vars = np.empty(n_bootstraps)
        for b in range(n_bootstraps):
            sample = rng.choice(self.values, size=self.n_obs, replace=True)
            q = np.percentile(sample, alpha * 100.0)
            boot_vars[b] = -q if as_loss else q
            
        tail_pct = (1.0 - ci_level) / 2.0 * 100.0
        lower_bound = float(np.percentile(boot_vars, tail_pct))
        upper_bound = float(np.percentile(boot_vars, 100.0 - tail_pct))
        
        return point_estimate, lower_bound, upper_bound
