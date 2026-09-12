"""
Core Mathematical Utilities for Nexus Quant Platform.
Includes matrix operations, Higham nearest correlation projection, Cholesky decomposition,
statistical moments, and optimization helpers.
"""

from typing import Tuple, Optional, Callable, Dict, Any, List
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, OptimizeResult


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a square matrix is symmetric and strictly positive definite."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues > tol))
    except np.linalg.LinAlgError:
        return False


def nearest_correlation_matrix(
    corr_matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-7, min_eigenval: float = 1e-5
) -> np.ndarray:
    """
    Project an approximate correlation matrix to the nearest positive definite correlation matrix
    using Higham's (2002) alternating projection algorithm with eigenvalue floor.
    """
    mat = np.array(corr_matrix, dtype=float, copy=True)
    n = mat.shape[0]
    y = np.copy(mat)
    delta_s = np.zeros_like(mat)

    for _ in range(max_iter):
        r = y - delta_s
        # Project onto positive semi-definite cone with eigenvalue floor
        vals, vecs = np.linalg.eigh(r)
        vals = np.maximum(vals, min_eigenval)
        x = vecs @ np.diag(vals) @ vecs.T
        delta_s = x - r
        # Project onto unit diagonal and symmetry
        y = np.copy(x)
        np.fill_diagonal(y, 1.0)
        y = 0.5 * (y + y.T)
        y = np.clip(y, -1.0 + min_eigenval, 1.0 - min_eigenval)
        np.fill_diagonal(y, 1.0)
        if np.max(np.abs(y - x)) < tol:
            break

    # Ensure strictly positive definite
    vals, vecs = np.linalg.eigh(y)
    if np.any(vals <= 1e-8):
        vals = np.maximum(vals, min_eigenval)
        y = vecs @ np.diag(vals) @ vecs.T
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(y))
        y = y / np.outer(d, d)
        np.fill_diagonal(y, 1.0)

    return y


def cholesky_factor(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the lower-triangular Cholesky factor L such that Sigma = L @ L.T.
    If matrix is not strictly positive definite, project to nearest PSD first.
    """
    if not is_positive_definite(matrix):
        matrix = nearest_correlation_matrix(matrix)
    return np.linalg.cholesky(matrix)


def skewness(data: np.ndarray, bias: bool = False) -> float:
    """Calculate sample skewness."""
    arr = np.asarray(data, dtype=float)
    return float(stats.skew(arr, bias=bias))


def kurtosis(data: np.ndarray, excess: bool = True, bias: bool = False) -> float:
    """Calculate sample kurtosis (excess by default)."""
    arr = np.asarray(data, dtype=float)
    k = stats.kurtosis(arr, fisher=excess, bias=bias)
    return float(k)


def jarque_bera_stat(data: np.ndarray) -> Tuple[float, float, bool]:
    """
    Jarque-Bera goodness-of-fit test for normality.
    Returns: (statistic, p_value, is_normal_at_5pct).
    """
    arr = np.asarray(data, dtype=float).flatten()
    arr = arr[~np.isnan(arr)]
    stat_val, p_val = stats.jarque_bera(arr)
    return float(stat_val), float(p_val), bool(p_val > 0.05)


def solve_constrained_qp(
    objective_fn: Callable[[np.ndarray], float],
    gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    x0: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    equality_constraints: Optional[List[Dict[str, Any]]] = None,
    inequality_constraints: Optional[List[Dict[str, Any]]] = None,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> OptimizeResult:
    """
    Robust wrapper for Sequential Least Squares Programming (SLSQP).
    """
    constraints = []
    if equality_constraints:
        constraints.extend(equality_constraints)
    if inequality_constraints:
        constraints.extend(inequality_constraints)

    res = minimize(
        fun=objective_fn,
        x0=x0,
        jac=gradient_fn,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": tol, "maxiter": max_iter, "disp": False},
    )
    return res
