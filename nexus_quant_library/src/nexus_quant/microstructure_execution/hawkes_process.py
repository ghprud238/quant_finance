"""
Multivariate Hawkes Point Processes, Information Diffusion & Branching Endogeneity.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kstest


@dataclass
class HawkesFitResult:
    """Output container for calibrated multivariate Hawkes process parameters."""
    dimension: int
    mu: np.ndarray             # Baseline background intensities (M,)
    alpha: np.ndarray          # Excitation matrix (M, M)
    beta: np.ndarray           # Exponential decay matrix (M, M)
    branching_matrix: np.ndarray # Kernel norm matrix A_mn = alpha_mn / beta_mn
    endogeneity_ratio: float   # Spectral radius max |eig(A)|
    log_likelihood: float
    aic: float
    bic: float


class MultivariateHawkesEngine:
    """
    Multivariate Mutually Exciting Point Process Engine with Exponential Kernels.
    Reference: Rambaldi, Pennesi, and Lillo (2015).
    """

    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.fit_result: Optional[HawkesFitResult] = None

    def log_likelihood(
        self,
        params_flat: np.ndarray,
        event_times_list: List[np.ndarray],
        horizon: float,
    ) -> float:
        """
        Compute exact negative log-likelihood for multivariate exponential Hawkes process.
        params_flat layout: [mu_0..mu_M-1, alpha_00..alpha_M-1,M-1, beta_00..beta_M-1,M-1].
        """
        m_dim = self.dimension
        mu = params_flat[:m_dim]
        alpha = params_flat[m_dim:m_dim + m_dim*m_dim].reshape((m_dim, m_dim))
        beta = params_flat[m_dim + m_dim*m_dim:].reshape((m_dim, m_dim))

        # Stationarity check: spectral radius of A = alpha / beta < 1.0
        branching = alpha / np.maximum(beta, 1e-6)
        eig_max = np.max(np.abs(np.linalg.eigvals(branching)))
        if eig_max >= 0.999:
            return 1e8

        log_lik = 0.0

        for m in range(m_dim):
            t_m = event_times_list[m]
            n_m = len(t_m)
            if n_m == 0:
                continue

            # 1. Integral of compensator: -mu_m * T - sum_n (alpha_mn / beta_mn) * sum_j (1 - exp(-beta_mn * (T - t_j^n)))
            comp_int = mu[m] * horizon
            for n in range(m_dim):
                t_n = event_times_list[n]
                if len(t_n) > 0:
                    exp_terms = 1.0 - np.exp(-beta[m, n] * (horizon - t_n))
                    comp_int += (alpha[m, n] / beta[m, n]) * np.sum(exp_terms)

            # 2. Sum of log-intensities at event arrival times
            log_lambda_sum = 0.0
            # Pre-compute inter-event recursive decays
            for k in range(n_m):
                t_k = t_m[k]
                lambda_k = mu[m]
                for n in range(m_dim):
                    t_n = event_times_list[n]
                    t_past = t_n[t_n < t_k]
                    if len(t_past) > 0:
                        decay_sum = np.sum(np.exp(-beta[m, n] * (t_k - t_past)))
                        lambda_k += alpha[m, n] * decay_sum
                log_lambda_sum += np.log(max(lambda_k, 1e-8))

            log_lik += (log_lambda_sum - comp_int)

        return -float(log_lik)

    def fit(
        self,
        event_times_list: List[np.ndarray],
        horizon: Optional[float] = None,
        max_iter: int = 150,
    ) -> HawkesFitResult:
        """
        Calibrate multivariate Hawkes parameters (mu, alpha, beta) via Maximum Likelihood Estimation.
        """
        m_dim = self.dimension
        if horizon is None:
            horizon = max(np.max(t) for t in event_times_list if len(t) > 0) + 1.0

        # Heuristic initial guess
        mu_init = np.array([len(t) / (horizon * 2.0) for t in event_times_list])
        alpha_init = np.full((m_dim, m_dim), 0.15)
        beta_init = np.full((m_dim, m_dim), 1.20)
        p0 = np.concatenate([mu_init, alpha_init.ravel(), beta_init.ravel()])

        bounds = []
        # Bounds for mu
        for _ in range(m_dim):
            bounds.append((1e-4, 50.0))
        # Bounds for alpha
        for _ in range(m_dim * m_dim):
            bounds.append((1e-4, 20.0))
        # Bounds for beta
        for _ in range(m_dim * m_dim):
            bounds.append((0.05, 50.0))

        res = minimize(
            self.log_likelihood,
            p0,
            args=(event_times_list, horizon),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-6},
        )

        p_opt = res.x
        mu_opt = p_opt[:m_dim]
        alpha_opt = p_opt[m_dim:m_dim + m_dim*m_dim].reshape((m_dim, m_dim))
        beta_opt = p_opt[m_dim + m_dim*m_dim:].reshape((m_dim, m_dim))

        branching = alpha_opt / np.maximum(beta_opt, 1e-6)
        endogeneity = float(np.max(np.abs(np.linalg.eigvals(branching))))

        n_params = len(p_opt)
        n_events = sum(len(t) for t in event_times_list)
        neg_ll = float(res.fun)
        aic = 2.0 * n_params + 2.0 * neg_ll
        bic = n_params * np.log(max(n_events, 2)) + 2.0 * neg_ll

        fit_res = HawkesFitResult(
            dimension=m_dim,
            mu=mu_opt,
            alpha=alpha_opt,
            beta=beta_opt,
            branching_matrix=branching,
            endogeneity_ratio=endogeneity,
            log_likelihood=-neg_ll,
            aic=float(aic),
            bic=float(bic),
        )
        self.fit_result = fit_res
        return fit_res

    def impulse_response_half_life(self, source_node: int, target_node: int) -> float:
        """Half-life of information absorption: ln(2) / beta_{target, source}."""
        if self.fit_result is None:
            raise ValueError("Model not fitted.")
        b = self.fit_result.beta[target_node, source_node]
        return float(np.log(2.0) / max(b, 1e-6))

    def evaluate_time_change_diagnostics(
        self,
        event_times_list: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate goodness-of-fit via Papangelou's Random Time-Change Theorem:
        Integrated intensity compensators between consecutive events must be i.i.d. Exp(1).
        """
        if self.fit_result is None:
            raise ValueError("Model not fitted.")

        mu = self.fit_result.mu
        alpha = self.fit_result.alpha
        beta = self.fit_result.beta
        m_dim = self.dimension

        p_values = []
        for m in range(m_dim):
            t_m = event_times_list[m]
            if len(t_m) < 5:
                p_values.append(1.0)
                continue

            # Compute compensator gaps Delta Lambda_k
            compensator_gaps = []
            for k in range(1, len(t_m)):
                t_prev = t_m[k-1]
                t_curr = t_m[k]

                # Integral from t_prev to t_curr of lambda_m(s)
                gap_val = mu[m] * (t_curr - t_prev)
                for n in range(m_dim):
                    t_n = event_times_list[n]
                    past_ev = t_n[t_n < t_curr]
                    if len(past_ev) > 0:
                        # Int_{t_prev}^{t_curr} exp(-beta*(s - t_j)) ds
                        t_upper = np.maximum(0.0, t_curr - past_ev)
                        t_lower = np.maximum(0.0, t_prev - past_ev)
                        decay_int = (np.exp(-beta[m, n] * t_lower) - np.exp(-beta[m, n] * t_upper)) / beta[m, n]
                        gap_val += alpha[m, n] * np.sum(decay_int)

                compensator_gaps.append(max(gap_val, 1e-8))

            # KS test against standard exponential distribution Exp(1) (scale = 1.0)
            ks_res = kstest(compensator_gaps, "expon", args=(0, 1.0))
            p_values.append(float(ks_res.pvalue))

        return {
            "KS_P_Value_News": p_values[0] if len(p_values) > 0 else 1.0,
            "KS_P_Value_Jumps": p_values[1] if len(p_values) > 1 else 1.0,
            "Goodness_Of_Fit_Passed": bool(min(p_values) > 0.01),
        }
