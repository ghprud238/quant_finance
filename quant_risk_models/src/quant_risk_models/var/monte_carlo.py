"""Monte Carlo Simulation Value-at-Risk (VaR) and Tail Risk Engine."""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


class MonteCarloVaREngine:
    """Monte Carlo Simulation Engine for Value-at-Risk and Expected Shortfall.
    
    Includes:
    - Geometric Brownian Motion (GBM) multi-path asset simulation
    - Merton Jump-Diffusion simulation with Poisson jump arrivals
    - Multi-asset correlated simulation via Cholesky decomposition
    - Horizon path simulation and fan chart quantile envelope extraction
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
        else:
            self.returns = None
            self.mean = 0.0 if mean is None else mean
            self.std = 0.01 if std is None else std

    def simulate_gbm(
        self,
        n_simulations: int = 100000,
        horizon: int = 1,
        n_steps: int = 1,
        initial_value: float = 1.0,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        random_state: int = 42,
    ) -> np.ndarray:
        """Simulates asset price paths under Geometric Brownian Motion.
        
        dS_t / S_t = mu * dt + sigma * dW_t
        S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma * sqrt(dt) * Z)
        
        Returns:
            Array of shape (n_simulations, n_steps + 1) with price paths.
        """
        mu = self.mean if mean is None else mean
        sigma = self.std if std is None else std
        
        rng = np.random.default_rng(random_state)
        dt = horizon / n_steps
        
        # Standard normal random matrix (n_simulations, n_steps)
        z = rng.standard_normal((n_simulations, n_steps))
        
        # Incremental log returns
        drift_term = (mu - 0.5 * (sigma**2)) * dt
        diffusion_term = sigma * np.sqrt(dt) * z
        log_returns_steps = drift_term + diffusion_term
        
        # Cumulative log returns
        cum_log_returns = np.zeros((n_simulations, n_steps + 1))
        cum_log_returns[:, 1:] = np.cumsum(log_returns_steps, axis=1)
        
        paths = initial_value * np.exp(cum_log_returns)
        return paths

    def simulate_merton_jump_diffusion(
        self,
        n_simulations: int = 100000,
        horizon: int = 1,
        n_steps: int = 1,
        initial_value: float = 1.0,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        jump_intensity: float = 0.05, # lambda: expected jumps per day
        jump_mean: float = -0.02,     # mu_J: mean jump size
        jump_std: float = 0.04,      # sigma_J: jump volatility
        random_state: int = 42,
    ) -> np.ndarray:
        """Simulates asset paths under Merton (1976) Jump-Diffusion.
        
        dS_t / S_t = (mu - lambda * k) * dt + sigma * dW_t + (Y - 1) * dN_t
        where k = E[Y - 1] = exp(mu_J + 0.5*sigma_J^2) - 1, N_t ~ Poisson(lambda * dt)
        """
        mu = self.mean if mean is None else mean
        sigma = self.std if std is None else std
        
        rng = np.random.default_rng(random_state)
        dt = horizon / n_steps
        
        # Compensator k
        k = np.exp(jump_mean + 0.5 * (jump_std**2)) - 1.0
        
        # Continuous diffusion part
        z = rng.standard_normal((n_simulations, n_steps))
        continuous_drift = (mu - jump_intensity * k - 0.5 * (sigma**2)) * dt
        continuous_term = continuous_drift + sigma * np.sqrt(dt) * z
        
        # Poisson jump arrivals
        n_jumps = rng.poisson(jump_intensity * dt, (n_simulations, n_steps))
        
        # Total jump magnitude per step
        jump_terms = np.zeros((n_simulations, n_steps))
        mask = n_jumps > 0
        if np.any(mask):
            total_events = np.sum(n_jumps)
            # Individual jump sizes
            individual_jumps = rng.normal(jump_mean, jump_std, total_events)
            
            # Map back to steps
            idx = 0
            for i, j in zip(*np.nonzero(mask)):
                cnt = n_jumps[i, j]
                jump_terms[i, j] = np.sum(individual_jumps[idx : idx + cnt])
                idx += cnt
                
        total_log_returns = continuous_term + jump_terms
        
        cum_log_returns = np.zeros((n_simulations, n_steps + 1))
        cum_log_returns[:, 1:] = np.cumsum(total_log_returns, axis=1)
        
        paths = initial_value * np.exp(cum_log_returns)
        return paths

    def simulate_correlated_portfolio(
        self,
        weights: Union[Sequence[float], np.ndarray, Dict[str, float]],
        cov_matrix: Union[np.ndarray, pd.DataFrame],
        mean_returns: Optional[Union[Sequence[float], np.ndarray, Dict[str, float]]] = None,
        n_simulations: int = 100000,
        horizon: int = 1,
        n_steps: int = 1,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates multi-asset correlated trajectories via Cholesky decomposition.
        
        Returns:
            (portfolio_terminal_returns, asset_terminal_returns)
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
            cov = np.asarray(cov_matrix, dtype=float)
            mu = np.zeros(len(w)) if mean_returns is None else np.asarray(mean_returns, dtype=float)
            
        n_assets = len(w)
        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        L = np.linalg.cholesky(cov_psd)
        
        rng = np.random.default_rng(random_state)
        dt = horizon / n_steps
        
        # Standard normal random vectors (n_simulations, n_steps, n_assets)
        z_raw = rng.standard_normal((n_simulations, n_steps, n_assets))
        
        # Correlated shocks: Z_corr = Z_raw @ L.T
        z_corr = z_raw @ L.T
        
        # Asset daily volatilities
        asset_vols = np.sqrt(np.diag(cov_psd))
        
        drift = (mu - 0.5 * (asset_vols**2)) * dt
        log_ret_steps = drift + np.sqrt(dt) * z_corr
        
        total_log_returns = np.sum(log_ret_steps, axis=1) # (n_simulations, n_assets)
        asset_terminal_returns = np.exp(total_log_returns) - 1.0
        
        # Portfolio simple return is weighted sum of asset simple returns
        portfolio_terminal_returns = asset_terminal_returns @ w
        return portfolio_terminal_returns, asset_terminal_returns

    def compute_var(
        self,
        simulated_returns: np.ndarray,
        confidence_level: float = 0.95,
        as_loss: bool = True,
    ) -> float:
        """Calculates Value-at-Risk from Monte Carlo simulated returns."""
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        alpha = 1.0 - confidence_level
        cutoff = float(np.percentile(simulated_returns, alpha * 100.0))
        return -cutoff if as_loss else cutoff

    def compute_cvar(
        self,
        simulated_returns: np.ndarray,
        confidence_level: float = 0.95,
        as_loss: bool = True,
    ) -> float:
        """Calculates Expected Shortfall (CVaR) from Monte Carlo simulated returns."""
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        alpha = 1.0 - confidence_level
        cutoff = np.percentile(simulated_returns, alpha * 100.0)
        tail = simulated_returns[simulated_returns <= cutoff]
        cvar_val = float(np.mean(tail)) if len(tail) > 0 else float(cutoff)
        return -cvar_val if as_loss else cvar_val

    def generate_fan_chart_data(
        self,
        n_simulations: int = 5000,
        horizon: int = 252,
        n_steps: int = 252,
        initial_value: float = 100.0,
        percentiles: Sequence[float] = (5, 25, 50, 75, 95),
        random_state: int = 42,
    ) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
        """Generates fan chart percentile envelopes across the simulation horizon.
        
        Returns:
            (time_steps, percentile_dict)
            where time_steps is array of days [0, 1, ..., n_steps]
            and percentile_dict maps percentile -> array of values along path.
        """
        paths = self.simulate_gbm(
            n_simulations=n_simulations,
            horizon=horizon,
            n_steps=n_steps,
            initial_value=initial_value,
            random_state=random_state,
        )
        
        time_steps = np.linspace(0, horizon, n_steps + 1)
        percentile_dict = {}
        for p in percentiles:
            percentile_dict[p] = np.percentile(paths, p, axis=0)
            
        return time_steps, percentile_dict
