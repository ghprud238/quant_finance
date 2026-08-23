"""Parametric Value-at-Risk (VaR) and Analytical Tail Risk Models."""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class ParametricVaRModel:
    """Parametric Value-at-Risk (VaR) and Expected Shortfall (CVaR) Engine.
    
    Includes:
    - Gaussian / Delta-Normal VaR & CVaR
    - Student's t-distribution VaR & CVaR (fat-tail modeling)
    - Cornish-Fisher expansion VaR (skewness and excess kurtosis adjustment)
    - Multi-asset analytical portfolio VaR and component VaR decomposition
    - Horizon scaling (e.g. 1-day to 10-day via sqrt(T) rule)
    """

    def __init__(
        self,
        returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> None:
        if returns is not None:
            if isinstance(returns, pd.DataFrame):
                if returns.shape[1] == 1:
                    clean_ret = returns.iloc[:, 0].dropna().values.astype(float)
                else:
                    clean_ret = returns.dropna().values.astype(float)
            elif isinstance(returns, pd.Series):
                clean_ret = returns.dropna().values.astype(float)
            else:
                arr = np.asarray(returns, dtype=float)
                clean_ret = arr[~np.isnan(arr)]
                
            self.returns = clean_ret
            self.mean = float(np.mean(clean_ret)) if mean is None else mean
            self.std = float(np.std(clean_ret, ddof=1)) if std is None else std
            
            # Distribution moments
            if len(clean_ret.shape) == 1 or clean_ret.shape[1] == 1:
                flat = clean_ret.flatten()
                self.skew = float(stats.skew(flat))
                self.kurtosis = float(stats.kurtosis(flat)) # excess kurtosis (Fisher, normal=0)
            else:
                self.skew = 0.0
                self.kurtosis = 0.0
        else:
            self.returns = None
            self.mean = 0.0 if mean is None else mean
            self.std = 0.01 if std is None else std
            self.skew = 0.0
            self.kurtosis = 0.0

    def gaussian_var(
        self,
        confidence_level: float = 0.95,
        horizon: int = 1,
        as_loss: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> float:
        """Gaussian (Delta-Normal) Value-at-Risk.
        
        VaR_alpha = -(mu * T - z_alpha * sigma * sqrt(T))
        """
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        mu = self.mean if mean is None else mean
        sigma = self.std if std is None else std
        
        z = stats.norm.ppf(confidence_level)
        return_cutoff = mu * horizon - z * sigma * np.sqrt(horizon)
        return float(-return_cutoff if as_loss else return_cutoff)

    def gaussian_cvar(
        self,
        confidence_level: float = 0.95,
        horizon: int = 1,
        as_loss: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> float:
        """Gaussian Expected Shortfall / Conditional VaR (analytical).
        
        ES_alpha = -mu * T + sigma * sqrt(T) * pdf(z) / (1 - alpha)
        """
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        mu = self.mean if mean is None else mean
        sigma = self.std if std is None else std
        
        alpha = 1.0 - confidence_level
        z = stats.norm.ppf(confidence_level)
        pdf_z = stats.norm.pdf(z)
        
        cvar_cutoff = mu * horizon - sigma * np.sqrt(horizon) * (pdf_z / alpha)
        return float(-cvar_cutoff if as_loss else cvar_cutoff)

    def student_t_var(
        self,
        confidence_level: float = 0.95,
        df: Optional[float] = None,
        horizon: int = 1,
        as_loss: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> float:
        """Student's t-distribution Value-at-Risk.
        
        Accounts for leptokurtosis (fat tails).
        """
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        mu = self.mean if mean is None else mean
        sigma = self.std if std is None else std
        
        if df is None:
            if self.returns is not None and len(self.returns) > 10:
                params = stats.t.fit(self.returns)
                df = max(params[0], 2.1)
            else:
                df = 5.0
                
        if df <= 2.0:
            raise ValueError(f"Degrees of freedom must be > 2 for finite variance, got {df}")
            
        t_crit = stats.t.ppf(confidence_level, df=df)
        scaling = np.sqrt((df - 2.0) / df)
        return_cutoff = mu * horizon - t_crit * sigma * scaling * np.sqrt(horizon)
        return float(-return_cutoff if as_loss else return_cutoff)

    def student_t_cvar(
        self,
        confidence_level: float = 0.95,
        df: Optional[float] = None,
        horizon: int = 1,
        as_loss: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> float:
        """Student's t Expected Shortfall / Conditional VaR (analytical)."""
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        mu = self.mean if mean is None else mean
        sigma = self.std if std is None else std
        
        if df is None:
            if self.returns is not None and len(self.returns) > 10:
                params = stats.t.fit(self.returns)
                df = max(params[0], 2.1)
            else:
                df = 5.0
                
        if df <= 2.0:
            raise ValueError(f"Degrees of freedom must be > 2 for finite variance, got {df}")
            
        alpha = 1.0 - confidence_level
        t_crit = stats.t.ppf(confidence_level, df=df)
        pdf_t = stats.t.pdf(t_crit, df=df)
        scaling = np.sqrt((df - 2.0) / df)
        
        es_term = (pdf_t / alpha) * ((df + t_crit**2) / (df - 1.0)) * scaling
        cvar_cutoff = mu * horizon - sigma * np.sqrt(horizon) * es_term
        return float(-cvar_cutoff if as_loss else cvar_cutoff)

    def cornish_fisher_var(
        self,
        confidence_level: float = 0.95,
        horizon: int = 1,
        as_loss: bool = True,
        skew: Optional[float] = None,
        kurtosis: Optional[float] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> float:
        """Cornish-Fisher Modified Value-at-Risk.
        
        Expands the lower alpha-quantile z_alpha (alpha = 1 - confidence_level)
        adjusting for sample skewness S and excess kurtosis K:
        z_mod = z_alpha + (z_alpha^2 - 1)*S/6 + (z_alpha^3 - 3z_alpha)*K/24 - (2z_alpha^3 - 5z_alpha)*S^2/36
        return_cutoff = mu * horizon + z_mod * sigma * sqrt(horizon)
        """
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        mu = self.mean if mean is None else mean
        sigma = self.std if std is None else std
        s = self.skew if skew is None else skew
        k = self.kurtosis if kurtosis is None else kurtosis
        
        alpha = 1.0 - confidence_level
        z_alpha = stats.norm.ppf(alpha)
        z_mod = (
            z_alpha
            + (z_alpha**2 - 1.0) * s / 6.0
            + (z_alpha**3 - 3.0 * z_alpha) * k / 24.0
            - (2.0 * z_alpha**3 - 5.0 * z_alpha) * (s**2) / 36.0
        )
        
        return_cutoff = mu * horizon + z_mod * sigma * np.sqrt(horizon)
        return float(-return_cutoff if as_loss else return_cutoff)

    def portfolio_analytical_var(
        self,
        weights: Union[Sequence[float], np.ndarray, Dict[str, float]],
        cov_matrix: Union[np.ndarray, pd.DataFrame],
        mean_returns: Optional[Union[Sequence[float], np.ndarray, Dict[str, float]]] = None,
        confidence_level: float = 0.95,
        horizon: int = 1,
        as_loss: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """Multi-asset Portfolio Analytical VaR with Component VaR risk attribution.
        
        Returns:
            (portfolio_var, component_var_dict)
            where sum(component_var_dict.values()) == portfolio_var
        """
        if isinstance(weights, dict):
            asset_names = list(weights.keys())
            w = np.array([weights[k] for k in asset_names], dtype=float)
            if isinstance(cov_matrix, pd.DataFrame):
                cov = cov_matrix.loc[asset_names, asset_names].values
            else:
                cov = np.asarray(cov_matrix, dtype=float)
            if mean_returns is not None and isinstance(mean_returns, dict):
                mu = np.array([mean_returns[k] for k in asset_names], dtype=float)
            elif mean_returns is not None:
                mu = np.asarray(mean_returns, dtype=float)
            else:
                mu = np.zeros(len(w))
        else:
            w = np.asarray(weights, dtype=float)
            asset_names = [f"Asset_{i}" for i in range(len(w))]
            cov = np.asarray(cov_matrix, dtype=float)
            mu = np.zeros(len(w)) if mean_returns is None else np.asarray(mean_returns, dtype=float)
            
        port_mu = float(np.dot(w, mu))
        port_variance = float(w @ cov @ w)
        port_sigma = float(np.sqrt(max(port_variance, 1e-12)))
        
        z = stats.norm.ppf(confidence_level)
        port_var_val = -(port_mu * horizon - z * port_sigma * np.sqrt(horizon)) if as_loss else (port_mu * horizon - z * port_sigma * np.sqrt(horizon))
        
        # Component VaR decomposition
        # Marginal VaR_i = -(mu_i * T - z * (Cov @ w)_i / port_sigma * sqrt(T))
        cov_w = cov @ w
        marginal_var = -(mu * horizon - z * (cov_w / port_sigma) * np.sqrt(horizon)) if as_loss else (mu * horizon - z * (cov_w / port_sigma) * np.sqrt(horizon))
        component_var = w * marginal_var
        
        comp_dict = {asset_names[i]: float(component_var[i]) for i in range(len(w))}
        return float(port_var_val), comp_dict
