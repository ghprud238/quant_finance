"""GARCH(1,1) and GJR-GARCH(1,1) Volatility Modeling and Multi-Step Forecasting."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize


@dataclass
class GARCHForecastResult:
    """Container for multi-step volatility forecasts."""
    horizon: int
    daily_variance_forecast: np.ndarray
    daily_volatility_forecast: np.ndarray
    annualized_volatility_forecast: np.ndarray
    cumulative_annualized_volatility: np.ndarray
    unconditional_volatility_ann: float
    persistence: float
    half_life_days: float

    def to_dataframe(self) -> pd.DataFrame:
        steps = np.arange(1, self.horizon + 1)
        return pd.DataFrame({
            "Step": steps,
            "Daily_Variance": self.daily_variance_forecast,
            "Daily_Vol": self.daily_volatility_forecast,
            "Daily_Vol_Ann": self.annualized_volatility_forecast,
            "Term_Vol_Ann": self.cumulative_annualized_volatility,
        }).set_index("Step")


@dataclass
class GARCHFitResult:
    """Container for GARCH model estimation results and diagnostics."""
    model_type: str  # 'GARCH' or 'GJR-GARCH'
    params: Dict[str, float]  # mu, omega, alpha, beta, gamma
    standard_errors: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    persistence: float
    unconditional_variance: float
    unconditional_volatility_ann: float
    half_life_days: float
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    residuals: pd.Series
    n_obs: int

    def summary(self) -> pd.DataFrame:
        rows = []
        for p_name, p_val in self.params.items():
            se = self.standard_errors.get(p_name, np.nan)
            t_val = self.t_stats.get(p_name, np.nan)
            pval = self.p_values.get(p_name, np.nan)
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            rows.append({
                "Parameter": p_name,
                "Estimate": p_val,
                "Std_Error": se,
                "t_Stat": t_val,
                "p_Value": pval,
                "Significance": sig,
            })
        return pd.DataFrame(rows).set_index("Parameter")


class GARCHModel:
    """GARCH(1,1) and GJR-GARCH(1,1) Volatility Modeling and Multi-Step Forecasting Engine.
    
    Supports:
    - Standard GARCH(1,1): sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    - GJR-GARCH(1,1): sigma_t^2 = omega + (alpha + gamma * I_{eps < 0}) * eps_{t-1}^2 + beta * sigma_{t-1}^2
    """

    def __init__(self, model_type: str = "GJR-GARCH", mean_model: str = "constant") -> None:
        model_type = model_type.upper()
        if model_type not in ["GARCH", "GJR-GARCH"]:
            raise ValueError("model_type must be either 'GARCH' or 'GJR-GARCH'")
        self.model_type = model_type
        self.mean_model = mean_model
        self.fit_result: Optional[GARCHFitResult] = None

    def fit(self, returns: Union[pd.Series, np.ndarray, List[float]]) -> GARCHFitResult:
        """Estimates model parameters via Maximum Likelihood Estimation (MLE)."""
        if isinstance(returns, pd.Series):
            ret_series = returns.dropna()
            dates_idx = ret_series.index
            r = ret_series.values
        else:
            r = np.asarray(returns, dtype=float)
            dates_idx = pd.RangeIndex(len(r))

        n = len(r)
        if n < 30:
            raise ValueError("At least 30 observations required for reliable GARCH estimation.")

        # Initial variance estimate
        sample_var = np.var(r, ddof=1)
        sample_mean = np.mean(r)

        # Initial parameter guess [mu, omega, alpha, beta, (gamma)]
        if self.model_type == "GARCH":
            # mu, omega, alpha, beta
            init_params = np.array([sample_mean, sample_var * 0.05, 0.08, 0.88])
            bounds = [
                (-0.1, 0.1),
                (1e-8, sample_var * 0.5),
                (1e-6, 0.4),
                (1e-6, 0.98),
            ]
        else:
            # GJR-GARCH: mu, omega, alpha, beta, gamma
            init_params = np.array([sample_mean, sample_var * 0.05, 0.04, 0.88, 0.06])
            bounds = [
                (-0.1, 0.1),
                (1e-8, sample_var * 0.5),
                (1e-6, 0.4),
                (1e-6, 0.98),
                (-0.3, 0.5),
            ]

        def neg_log_likelihood(params: np.ndarray) -> float:
            if self.model_type == "GARCH":
                mu, omega, alpha, beta = params
                gamma = 0.0
            else:
                mu, omega, alpha, beta, gamma = params

            # Stationarity & positivity constraints
            persistence = alpha + beta + 0.5 * gamma
            if persistence >= 0.9999 or alpha < 0 or beta < 0 or omega <= 0 or (alpha + gamma) < 0:
                return 1e8

            eps = r - mu
            sigma2 = np.zeros(n)
            # Unconditional variance for t=0
            uncond_var = omega / (1.0 - persistence)
            sigma2[0] = max(uncond_var, 1e-6)

            for t in range(1, n):
                prev_eps = eps[t-1]
                indicator = 1.0 if prev_eps < 0 else 0.0
                sigma2[t] = omega + (alpha + gamma * indicator) * (prev_eps ** 2) + beta * sigma2[t-1]
                if sigma2[t] <= 1e-10:
                    sigma2[t] = 1e-10

            # Gaussian log-likelihood
            ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2)
            if not np.isfinite(ll):
                return 1e8
            return -ll

        # Non-linear optimization
        opt_res = minimize(
            neg_log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not opt_res.success:
            # Fallback to SLSQP if L-BFGS-B struggled with bounds
            opt_res = minimize(
                neg_log_likelihood,
                init_params,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-9},
            )

        opt_params = opt_res.x
        if self.model_type == "GARCH":
            mu, omega, alpha, beta = opt_params
            gamma = 0.0
            param_names = ["mu", "omega", "alpha", "beta"]
            param_dict = {"mu": float(mu), "omega": float(omega), "alpha": float(alpha), "beta": float(beta)}
        else:
            mu, omega, alpha, beta, gamma = opt_params
            param_names = ["mu", "omega", "alpha", "beta", "gamma"]
            param_dict = {"mu": float(mu), "omega": float(omega), "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma)}

        # Numerical Hessian for standard errors
        k = len(opt_params)
        eps_step = 1e-5
        hessian = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                p_pp = opt_params.copy(); p_pp[i] += eps_step; p_pp[j] += eps_step
                p_pm = opt_params.copy(); p_pm[i] += eps_step; p_pm[j] -= eps_step
                p_mp = opt_params.copy(); p_mp[i] -= eps_step; p_mp[j] += eps_step
                p_mm = opt_params.copy(); p_mm[i] -= eps_step; p_mm[j] -= eps_step
                hessian[i, j] = (neg_log_likelihood(p_pp) - neg_log_likelihood(p_pm) - neg_log_likelihood(p_mp) + neg_log_likelihood(p_mm)) / (4 * eps_step * eps_step)

        try:
            cov_matrix = np.linalg.inv(hessian)
            diag = np.diag(cov_matrix)
            se_arr = np.sqrt(np.maximum(diag, 1e-10))
        except Exception:
            se_arr = np.ones(k) * 0.01

        se_dict = {p_name: float(se_arr[i]) for i, p_name in enumerate(param_names)}
        t_stat_dict = {p_name: float(param_dict[p_name] / max(se_dict[p_name], 1e-8)) for p_name in param_names}
        p_val_dict = {p_name: float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat_dict[p_name])))) for p_name in param_names}

        # Reconstruct conditional volatility series
        eps = r - mu
        sigma2 = np.zeros(n)
        persistence = alpha + beta + 0.5 * gamma
        uncond_var = omega / max(1.0 - persistence, 1e-6)
        sigma2[0] = uncond_var

        for t in range(1, n):
            prev_eps = eps[t-1]
            ind = 1.0 if prev_eps < 0 else 0.0
            sigma2[t] = omega + (alpha + gamma * ind) * (prev_eps ** 2) + beta * sigma2[t-1]

        cond_vol = pd.Series(np.sqrt(sigma2), index=dates_idx, name="Conditional_Vol")
        std_resid = pd.Series(eps / np.sqrt(sigma2), index=dates_idx, name="Std_Residuals")
        resid = pd.Series(eps, index=dates_idx, name="Residuals")

        log_lik = -float(opt_res.fun)
        aic = 2.0 * k - 2.0 * log_lik
        bic = np.log(n) * k - 2.0 * log_lik

        # Half life: ln(0.5) / ln(P)
        if persistence > 0 and persistence < 1.0:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.nan

        self.fit_result = GARCHFitResult(
            model_type=self.model_type,
            params=param_dict,
            standard_errors=se_dict,
            t_stats=t_stat_dict,
            p_values=p_val_dict,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            persistence=float(persistence),
            unconditional_variance=float(uncond_var),
            unconditional_volatility_ann=float(np.sqrt(uncond_var * 252)),
            half_life_days=float(half_life),
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            residuals=resid,
            n_obs=n,
        )
        return self.fit_result

    def forecast(self, horizon: int = 30) -> GARCHForecastResult:
        """Generates analytical multi-step forward volatility forecasts."""
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before forecasting. Call fit() first.")

        p = self.fit_result.params
        omega = p["omega"]
        alpha = p["alpha"]
        beta = p["beta"]
        gamma = p.get("gamma", 0.0)
        persistence = self.fit_result.persistence
        uncond_var = self.fit_result.unconditional_variance

        # Last observed conditional variance and residual
        last_var = self.fit_result.conditional_volatility.iloc[-1] ** 2
        last_eps = self.fit_result.residuals.iloc[-1]
        last_ind = 1.0 if last_eps < 0 else 0.0

        # Step 1 forecast
        var_h = np.zeros(horizon)
        var_h[0] = omega + (alpha + gamma * last_ind) * (last_eps ** 2) + beta * last_var

        # Step 2 to horizon: E_t[sigma_{t+h}^2] = sigma_L^2 + P^{h-1} * (sigma_{t+1}^2 - sigma_L^2)
        for h in range(1, horizon):
            var_h[h] = uncond_var + (persistence ** h) * (var_h[0] - uncond_var)

        daily_vol = np.sqrt(var_h)
        daily_vol_ann = daily_vol * np.sqrt(252)

        # Term / cumulative annualized volatility: sqrt( (1/H) * sum(var_h) * 252 )
        cum_var = np.cumsum(var_h)
        steps = np.arange(1, horizon + 1)
        term_vol_ann = np.sqrt((cum_var / steps) * 252)

        return GARCHForecastResult(
            horizon=horizon,
            daily_variance_forecast=var_h,
            daily_volatility_forecast=daily_vol,
            annualized_volatility_forecast=daily_vol_ann,
            cumulative_annualized_volatility=term_vol_ann,
            unconditional_volatility_ann=self.fit_result.unconditional_volatility_ann,
            persistence=persistence,
            half_life_days=self.fit_result.half_life_days,
        )
