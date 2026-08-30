"""Multivariate Hawkes Process Engine for Financial Information Diffusion & Endogeneity Modeling."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
from scipy.stats import kstest, expon


@dataclass
class HawkesFitResult:
    """Fitted parameters and econometric diagnostics for a Multivariate Hawkes Process."""
    dimension: int
    mu: np.ndarray             # (M,) Baseline intensities
    alpha: np.ndarray          # (M, M) Excitation magnitude matrix
    beta: np.ndarray           # (M, M) Exponential decay matrix
    branching_matrix: np.ndarray # (M, M) A_ij = alpha_ij / beta_ij
    spectral_radius: float     # Maximum absolute eigenvalue of branching matrix
    endogeneity_ratio: float   # Endogenous feedback fraction
    log_likelihood: float
    aic: float
    bic: float
    n_events: List[int]
    horizon: float
    half_lives: np.ndarray     # (M, M) ln(2) / beta_ij


@dataclass
class ResidualDiagnosticResult:
    """Time-transformed residual goodness-of-fit diagnostics."""
    compensators: List[np.ndarray]
    ks_statistics: List[float]
    ks_p_values: List[float]
    is_valid_exp1: List[bool]
    mean_inter_arrival: List[float]


class MultivariateHawkesEngine:
    """2-Variate & M-Variate Hawkes Point Process Model with Exponential Kernels."""

    def __init__(self, dimension: int = 2):
        self.dim = dimension
        self.fitted_params: Optional[HawkesFitResult] = None

    @staticmethod
    def _compute_recursive_log_likelihood(
        params_flat: np.ndarray,
        event_times: List[np.ndarray],
        horizon: float,
        dim: int,
    ) -> float:
        """Computes exact negative log-likelihood for M-variate Hawkes with exponential decay."""
        # Unpack parameters: mu (M,), alpha (M, M), beta (M, M)
        mu = params_flat[0:dim]
        alpha = params_flat[dim : dim + dim * dim].reshape((dim, dim))
        beta = params_flat[dim + dim * dim : dim + 2 * dim * dim].reshape((dim, dim))

        # Stationarity / Positivity penalty
        if np.any(mu <= 1e-6) or np.any(alpha < 0) or np.any(beta <= 1e-6):
            return 1e12

        # Branching ratio check
        branching = alpha / beta
        try:
            spectral_rad = float(np.max(np.abs(np.linalg.eigvals(branching))))
            if spectral_rad >= 0.999:
                return 1e12 + (spectral_rad - 0.99) * 1e6
        except Exception:
            return 1e12

        total_ll = 0.0

        # Term 1: Integral of intensity from 0 to T: int_0^T lambda_m(t) dt
        for m in range(dim):
            integral_m = mu[m] * horizon
            for n in range(dim):
                t_n = event_times[n]
                if len(t_n) > 0:
                    sum_term = np.sum(1.0 - np.exp(-beta[m, n] * (horizon - t_n)))
                    integral_m += (alpha[m, n] / beta[m, n]) * sum_term
            total_ll -= integral_m

        # Term 2: Sum of log(lambda_m(t_k^m)) across all events
        for m in range(dim):
            t_m = event_times[m]
            if len(t_m) == 0:
                continue

            for k, tk in enumerate(t_m):
                lam_k = mu[m]
                for n in range(dim):
                    t_n = event_times[n]
                    # History of n prior to tk
                    prior_t = t_n[t_n < tk]
                    if len(prior_t) > 0:
                        decay = np.exp(-beta[m, n] * (tk - prior_t))
                        lam_k += alpha[m, n] * np.sum(decay)

                if lam_k <= 1e-12:
                    return 1e12
                total_ll += np.log(lam_k)

        # Return negative log-likelihood for minimization
        return -total_ll

    def fit(
        self,
        event_times: List[np.ndarray],
        horizon: Optional[float] = None,
        max_iter: int = 250,
    ) -> HawkesFitResult:
        """Fits multivariate Hawkes process parameters using Maximum Likelihood Estimation (MLE)."""
        dim = len(event_times)
        self.dim = dim
        
        all_events = np.concatenate(event_times) if any(len(e) > 0 for e in event_times) else np.array([1.0])
        T = horizon if horizon is not None else float(np.max(all_events)) + 1.0

        # Initial parameter estimates
        n_events = [len(e) for e in event_times]
        mu_init = np.array([max(1e-4, n / T * 0.5) for n in n_events])
        alpha_init = np.full((dim, dim), 0.20)
        beta_init = np.full((dim, dim), 1.00)

        # For cross-excitations
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    alpha_init[i, j] = 0.15
                    beta_init[i, j] = 0.80

        x0 = np.concatenate([mu_init, alpha_init.flatten(), beta_init.flatten()])

        # Bounds: mu > 0, alpha >= 0, beta > 0
        bounds = []
        for _ in range(dim):
            bounds.append((1e-5, 50.0))  # mu
        for _ in range(dim * dim):
            bounds.append((0.0, 50.0))   # alpha
        for _ in range(dim * dim):
            bounds.append((1e-4, 50.0))  # beta

        res = opt.minimize(
            fun=self._compute_recursive_log_likelihood,
            x0=x0,
            args=(event_times, T, dim),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False, "ftol": 1e-7},
        )

        params = res.x
        mu_fit = params[0:dim]
        alpha_fit = params[dim : dim + dim * dim].reshape((dim, dim))
        beta_fit = params[dim + dim * dim : dim + 2 * dim * dim].reshape((dim, dim))

        branching = alpha_fit / np.maximum(beta_fit, 1e-6)
        spectral_rad = float(np.max(np.abs(np.linalg.eigvals(branching))))
        
        # Endogeneity ratio is the fraction of total events generated by self/cross excitement
        # Endogeneity ratio = spectral_radius in linear Hawkes
        endogeneity_ratio = min(0.999, spectral_rad)

        log_lik = -float(res.fun)
        k_params = len(x0)
        total_n = sum(n_events)
        aic = 2.0 * k_params - 2.0 * log_lik
        bic = k_params * np.log(max(1, total_n)) - 2.0 * log_lik
        half_lives = np.log(2.0) / np.maximum(beta_fit, 1e-6)

        fit_res = HawkesFitResult(
            dimension=dim,
            mu=mu_fit,
            alpha=alpha_fit,
            beta=beta_fit,
            branching_matrix=branching,
            spectral_radius=spectral_rad,
            endogeneity_ratio=endogeneity_ratio,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            n_events=n_events,
            horizon=T,
            half_lives=half_lives,
        )
        self.fitted_params = fit_res
        return fit_res

    def simulate(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        horizon: float,
        seed: Optional[int] = 42,
    ) -> List[np.ndarray]:
        """Simulates M-variate Hawkes process using Ogata's Modified Thinning Algorithm."""
        if seed is not None:
            np.random.seed(seed)

        dim = len(mu)
        events = [[] for _ in range(dim)]
        t = 0.0

        # Memory matrix for recursive decay: R[m, n]
        R = np.zeros((dim, dim))
        last_event_t = 0.0

        while t < horizon:
            # Current intensities
            # Decay R from last_event_t to t
            dt = t - last_event_t
            R_curr = R * np.exp(-beta * dt)
            lam_vec = mu + np.sum(alpha * R_curr, axis=1)
            lam_total = np.sum(lam_vec)

            if lam_total <= 0:
                break

            # Step 1: Draw next candidate time from homogeneous Poisson process with rate lam_total
            u = np.random.uniform(0.0, 1.0)
            t += -np.log(max(u, 1e-12)) / lam_total

            if t >= horizon:
                break

            # Step 2: Compute true intensities at candidate time t
            dt_cand = t - last_event_t
            R_cand = R * np.exp(-beta * dt_cand)
            lam_vec_cand = mu + np.sum(alpha * R_cand, axis=1)
            lam_total_cand = np.sum(lam_vec_cand)

            # Step 3: Acceptance / Rejection
            d = np.random.uniform(0.0, 1.0)
            if d * lam_total <= lam_total_cand:
                # Accept! Determine which dimension m generated the event
                cum_lam = np.cumsum(lam_vec_cand)
                m_chosen = int(np.searchsorted(cum_lam, d * lam_total))
                m_chosen = min(m_chosen, dim - 1)

                events[m_chosen].append(t)
                
                # Update memory matrix R
                R = R_cand
                R[:, m_chosen] += 1.0
                last_event_t = t

        return [np.array(e) for e in events]

    def impulse_response_kernel(
        self,
        t_grid: np.ndarray,
        from_node: int = 0,
        to_node: int = 1,
    ) -> np.ndarray:
        """Computes the cross-excitation impulse response kernel phi_ij(t) = alpha_ij * exp(-beta_ij * t)."""
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before evaluating impulse response.")
        
        a = self.fitted_params.alpha[to_node, from_node]
        b = self.fitted_params.beta[to_node, from_node]
        return a * np.exp(-b * t_grid)

    def information_absorption_half_life(
        self,
        from_node: int = 0,
        to_node: int = 1,
    ) -> float:
        """Computes information absorption half-life: t_1/2 = ln(2) / beta_ij."""
        if self.fitted_params is None:
            raise ValueError("Model must be fitted.")
        b = self.fitted_params.beta[to_node, from_node]
        return float(np.log(2.0) / max(b, 1e-6))

    def evaluate_residual_diagnostics(
        self,
        event_times: List[np.ndarray],
    ) -> ResidualDiagnosticResult:
        """Performs Papangelou Time-Change compensator integral and Kolmogorov-Smirnov Exp(1) test."""
        if self.fitted_params is None:
            raise ValueError("Model must be fitted.")

        dim = self.dim
        mu = self.fitted_params.mu
        alpha = self.fitted_params.alpha
        beta = self.fitted_params.beta

        compensators = []
        ks_stats = []
        ks_p_vals = []
        is_valid = []
        mean_intervals = []

        for m in range(dim):
            t_m = event_times[m]
            if len(t_m) < 2:
                compensators.append(np.array([]))
                ks_stats.append(1.0)
                ks_p_vals.append(0.0)
                is_valid.append(False)
                mean_intervals.append(0.0)
                continue

            # Compute compensator increments Lambda(t_{k-1}, t_k)
            tau_m = []
            for k in range(1, len(t_m)):
                t_prev = t_m[k - 1]
                t_curr = t_m[k]
                
                # Base integral
                integral_k = mu[m] * (t_curr - t_prev)
                
                # Cross excitation integrals
                for n in range(dim):
                    t_n = event_times[n]
                    # Events of type n occurring before t_curr
                    prior = t_n[t_n < t_curr]
                    if len(prior) > 0:
                        # Split: prior <= t_prev vs prior > t_prev
                        p1 = prior[prior <= t_prev]
                        if len(p1) > 0:
                            integral_k += (alpha[m, n] / beta[m, n]) * np.sum(
                                np.exp(-beta[m, n] * (t_prev - p1)) - np.exp(-beta[m, n] * (t_curr - p1))
                            )
                        p2 = prior[prior > t_prev]
                        if len(p2) > 0:
                            integral_k += (alpha[m, n] / beta[m, n]) * np.sum(
                                1.0 - np.exp(-beta[m, n] * (t_curr - p2))
                            )
                            
                tau_m.append(integral_k)

            tau_arr = np.array(tau_m)
            compensators.append(tau_arr)
            
            # KS test against standard Exponential(scale=1.0)
            ks_res = kstest(tau_arr, "expon")
            ks_stats.append(float(ks_res.statistic))
            ks_p_vals.append(float(ks_res.pvalue))
            # If p-value > 0.01, cannot reject hypothesis that residuals follow Exp(1)
            is_valid.append(bool(ks_res.pvalue > 0.01))
            mean_intervals.append(float(np.mean(tau_arr)) if len(tau_arr) > 0 else 0.0)

        return ResidualDiagnosticResult(
            compensators=compensators,
            ks_statistics=ks_stats,
            ks_p_values=ks_p_vals,
            is_valid_exp1=is_valid,
            mean_inter_arrival=mean_intervals,
        )
