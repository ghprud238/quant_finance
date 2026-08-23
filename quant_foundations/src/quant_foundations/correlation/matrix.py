"""Correlation and Covariance Matrix Estimation and Regularization.

Provides robust tools for calculating correlation and covariance matrices,
Ledoit-Wolf shrinkage, RiskMetrics EWMA covariance, rolling correlation,
positive definiteness checks, and Higham (2002) nearest correlation projection.
"""

from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd


def compute_correlation_matrix(
    returns_df: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """Compute the correlation matrix of asset returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame with observations as rows and assets as columns.
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        Correlation method to use:
        - 'pearson': standard linear correlation coefficient.
        - 'spearman': Spearman rank correlation.
        - 'kendall': Kendall Tau rank correlation.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix with assets as index and columns.

    Raises
    ------
    TypeError
        If `returns_df` is not a pandas DataFrame.
    ValueError
        If `returns_df` is empty, has fewer than 2 rows, or if `method` is invalid.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    if returns_df.empty or len(returns_df) < 2:
        raise ValueError("returns_df must contain at least 2 rows of observations.")
    
    valid_methods = {'pearson', 'spearman', 'kendall'}
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")

    corr = returns_df.corr(method=method)
    return corr


def compute_covariance_matrix(
    returns_df: pd.DataFrame,
    annualized: bool = True,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """Compute the sample covariance matrix of asset returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame with observations as rows and assets as columns.
    annualized : bool, default True
        If True, annualize the covariance matrix by scaling with `periods_per_year`.
    periods_per_year : int, default 252
        Number of trading periods in a year (e.g., 252 for daily, 52 for weekly, 12 for monthly).

    Returns
    -------
    pd.DataFrame
        Covariance matrix with assets as index and columns.

    Raises
    ------
    TypeError
        If `returns_df` is not a pandas DataFrame.
    ValueError
        If `returns_df` is empty or has fewer than 2 rows.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    if returns_df.empty or len(returns_df) < 2:
        raise ValueError("returns_df must contain at least 2 rows of observations.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    cov = returns_df.cov()
    if annualized:
        cov = cov * periods_per_year
    return cov


def ledoit_wolf_shrinkage(
    returns_df: pd.DataFrame,
    shrinkage_target: str = 'constant_correlation',
    annualized: bool = True,
    periods_per_year: int = 252
) -> Tuple[pd.DataFrame, float]:
    """Pure NumPy implementation of Ledoit & Wolf (2004) covariance shrinkage.

    Implements:
    1. 'constant_correlation': Ledoit & Wolf (2004) "Honey, I Shrunk the Sample
       Covariance Matrix", shrinking towards the constant correlation target.
    2. 'identity': Ledoit & Wolf (2004) "A well-conditioned estimator for
       large-dimensional covariance matrices", shrinking towards a scaled identity matrix.

    Formula:
        Sigma_shrunk = delta * Target + (1 - delta) * Sample_Cov

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame with shape (T, N).
    shrinkage_target : {'constant_correlation', 'identity'}, default 'constant_correlation'
        The structured target matrix to shrink towards.
    annualized : bool, default True
        If True, annualizes the resulting covariance matrix by multiplying by `periods_per_year`.
    periods_per_year : int, default 252
        Number of periods per year for annualization.

    Returns
    -------
    shrunk_covariance : pd.DataFrame
        Shrunk covariance matrix with asset names as index and columns.
    optimal_shrinkage_intensity : float
        Optimal shrinkage intensity delta in the interval [0.0, 1.0].

    Raises
    ------
    TypeError
        If `returns_df` is not a pandas DataFrame.
    ValueError
        If `returns_df` has fewer than 3 observations or fewer than 2 assets,
        or if `shrinkage_target` is invalid.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    if returns_df.empty or len(returns_df) < 3:
        raise ValueError("returns_df must contain at least 3 rows of observations.")
    if returns_df.shape[1] < 2:
        raise ValueError("returns_df must contain at least 2 assets (columns).")
    
    valid_targets = {'constant_correlation', 'identity'}
    if shrinkage_target not in valid_targets:
        raise ValueError(f"Invalid shrinkage_target '{shrinkage_target}'. Must be one of {valid_targets}.")

    T, N = returns_df.shape
    X = returns_df.values.astype(float)
    
    # Demean returns
    mean = np.mean(X, axis=0)
    Y = X - mean # (T, N)
    
    # Sample covariance S (1/T formulation consistent with asymptotic derivations)
    S = np.dot(Y.T, Y) / T
    var = np.diag(S)
    
    # Asymptotic variance of sample covariance elements: pi_mat
    # pi_hat_ij = 1/T * sum_t (y_{i,t} * y_{j,t} - s_{ij})^2
    Y2 = Y[:, :, None] * Y[:, None, :] # (T, N, N)
    dev = Y2 - S # (T, N, N)
    pi_mat = np.mean(dev**2, axis=0) # (N, N)
    pi_hat = np.sum(pi_mat)

    if shrinkage_target == 'constant_correlation':
        # Sample correlation matrix
        sqrt_var = np.sqrt(np.maximum(var, 1e-16))
        outer_sqrt = np.outer(sqrt_var, sqrt_var)
        R = S / outer_sqrt
        
        # Mean correlation across off-diagonal elements
        r_bar = (np.sum(R) - N) / (N * (N - 1))
        
        # Target matrix F
        F = r_bar * outer_sqrt
        np.fill_diagonal(F, var)
        
        # Cross-terms: rho_hat
        dev_diag = Y**2 - var # (T, N)
        theta_i_ij = np.mean(dev_diag[:, :, None] * dev, axis=0) # (N, N)
        theta_j_ij = np.mean(dev_diag[:, None, :] * dev, axis=0) # (N, N)
        
        sqrt_ratio_j_i = np.outer(1.0 / sqrt_var, sqrt_var)
        sqrt_ratio_i_j = np.outer(sqrt_var, 1.0 / sqrt_var)
        
        rho_mat = 0.5 * r_bar * (sqrt_ratio_j_i * theta_i_ij + sqrt_ratio_i_j * theta_j_ij)
        np.fill_diagonal(rho_mat, np.diag(pi_mat))
        rho_hat = np.sum(rho_mat)
        
        # Misspecification parameter gamma_hat = ||S - F||_F^2
        gamma_hat = np.sum((S - F)**2)
        
        if gamma_hat <= 1e-16:
            delta = 0.0
        else:
            kappa = (pi_hat - rho_hat) / gamma_hat
            delta = max(0.0, min(1.0, float(kappa / T)))
            
    elif shrinkage_target == 'identity':
        # Scaled identity target: F = mu * I where mu = trace(S) / N
        mu = np.trace(S) / N
        F = mu * np.eye(N)
        
        gamma_hat = np.sum((S - F)**2)
        if gamma_hat <= 1e-16:
            delta = 0.0
        else:
            delta = max(0.0, min(1.0, float((pi_hat / T) / gamma_hat)))

    # Shrunk covariance matrix
    shrunk_cov = delta * F + (1.0 - delta) * S

    # Scale if annualized
    if annualized:
        shrunk_cov = shrunk_cov * periods_per_year

    shrunk_df = pd.DataFrame(
        shrunk_cov,
        index=returns_df.columns,
        columns=returns_df.columns
    )
    return shrunk_df, delta


def ewma_covariance(
    returns_df: pd.DataFrame,
    decay_factor: float = 0.94,
    annualized: bool = True,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """Compute Exponentially Weighted Moving Average (EWMA) covariance matrix.

    Follows the J.P. Morgan RiskMetrics methodology where more recent observations
    receive exponentially decaying weights:
        w_t = (1 - lambda) * lambda^{T - 1 - t}

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame with shape (T, N).
    decay_factor : float, default 0.94
        Smoothing parameter lambda in (0, 1). Typically 0.94 for daily returns
        and 0.97 for monthly returns.
    annualized : bool, default True
        If True, annualize the resulting covariance matrix by scaling with `periods_per_year`.
    periods_per_year : int, default 252
        Number of trading periods in a year.

    Returns
    -------
    pd.DataFrame
        EWMA covariance matrix with asset names as index and columns.

    Raises
    ------
    TypeError
        If `returns_df` is not a pandas DataFrame.
    ValueError
        If `returns_df` is empty, or `decay_factor` is not in (0, 1).
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    if returns_df.empty or len(returns_df) < 1:
        raise ValueError("returns_df must contain at least 1 observation.")
    if not (0.0 < decay_factor < 1.0):
        raise ValueError("decay_factor must be strictly between 0 and 1.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    T, N = returns_df.shape
    X = returns_df.values.astype(float)
    
    # Observation index: 0 is oldest, T-1 is newest
    # Weights decay backwards from T-1
    powers = np.arange(T - 1, -1, -1)
    weights = decay_factor ** powers
    weights /= np.sum(weights)
    
    # Weighted mean
    weighted_mean = np.sum(X * weights[:, None], axis=0)
    X_centered = X - weighted_mean
    
    # Weighted covariance: sum_t w_t * (X_centered[t] outer X_centered[t])
    cov = (X_centered * weights[:, None]).T @ X_centered
    
    # Enforce symmetry
    cov = 0.5 * (cov + cov.T)
    
    if annualized:
        cov = cov * periods_per_year
        
    return pd.DataFrame(
        cov,
        index=returns_df.columns,
        columns=returns_df.columns
    )


def rolling_correlation(
    returns_df: pd.DataFrame,
    asset1: str,
    asset2: str,
    window: int = 63
) -> pd.Series:
    """Compute the rolling Pearson correlation between two assets over a rolling window.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame.
    asset1 : str
        Name of the first asset column in `returns_df`.
    asset2 : str
        Name of the second asset column in `returns_df`.
    window : int, default 63
        Rolling window size in observations (e.g. 63 days ~ 1 quarter).

    Returns
    -------
    pd.Series
        Time series of rolling correlation values with identical index to `returns_df`.

    Raises
    ------
    TypeError
        If `returns_df` is not a pandas DataFrame.
    KeyError
        If `asset1` or `asset2` is not present in `returns_df.columns`.
    ValueError
        If `window` is less than 2.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    if asset1 not in returns_df.columns:
        raise KeyError(f"Asset '{asset1}' not found in returns_df columns.")
    if asset2 not in returns_df.columns:
        raise KeyError(f"Asset '{asset2}' not found in returns_df columns.")
    if window < 2:
        raise ValueError("window must be an integer >= 2.")

    rolling_corr = (
        returns_df[asset1]
        .rolling(window=window, min_periods=window)
        .corr(returns_df[asset2])
    )
    rolling_corr.name = f"rolling_corr_{asset1}_{asset2}"
    return rolling_corr


def is_positive_definite(
    matrix: Union[np.ndarray, pd.DataFrame],
    tol: float = 1e-8
) -> bool:
    """Check if a matrix is symmetric and strictly positive definite.

    Parameters
    ----------
    matrix : Union[np.ndarray, pd.DataFrame]
        Square matrix to check.
    tol : float, default 1e-8
        Tolerance for eigenvalue positivity and symmetry.

    Returns
    -------
    bool
        True if matrix is symmetric and all eigenvalues > tol, False otherwise.
    """
    if isinstance(matrix, pd.DataFrame):
        mat = matrix.values
    else:
        mat = np.asarray(matrix, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False

    # Check symmetry
    if not np.allclose(mat, mat.T, atol=1e-6):
        return False

    # Check positive definiteness via eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(mat)
        return bool(np.all(eigvals > tol))
    except (np.linalg.LinAlgError, ValueError):
        return False


def nearest_correlation_matrix(
    corr_matrix: Union[np.ndarray, pd.DataFrame],
    max_iter: int = 100,
    tol: float = 1e-7
) -> Union[np.ndarray, pd.DataFrame]:
    """Compute the nearest positive semi-definite correlation matrix (Higham 2002).

    Uses Dykstra's alternating projection algorithm to find the closest correlation
    matrix in Frobenius norm to an invalid or indefinite matrix.

    Reference:
        Higham, N. J. (2002). "Computing the nearest correlation matrix—a problem
        from finance". IMA Journal of Numerical Analysis, 22(3), 329-343.

    Parameters
    ----------
    corr_matrix : Union[np.ndarray, pd.DataFrame]
        Square matrix (possibly non-positive definite or non-symmetric).
    max_iter : int, default 100
        Maximum number of alternating projection iterations.
    tol : float, default 1e-7
        Convergence tolerance on Frobenius norm change.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        The nearest valid correlation matrix with unit diagonal and positive semi-definiteness.
    """
    is_df = isinstance(corr_matrix, pd.DataFrame)
    if is_df:
        mat = corr_matrix.values.astype(float)
        idx = corr_matrix.index
        cols = corr_matrix.columns
    else:
        mat = np.asarray(corr_matrix, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("corr_matrix must be a 2D square matrix.")

    n = mat.shape[0]
    
    # Symmetrize and initialize
    Y = 0.5 * (mat + mat.T)
    np.fill_diagonal(Y, 1.0)
    
    delta_S = np.zeros((n, n), dtype=float)

    for _ in range(max_iter):
        R = Y - delta_S
        
        # Projection 1: Project onto positive semi-definite cone (P_S)
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0.0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        X = 0.5 * (X + X.T)
        
        # Update Dykstra correction
        delta_S = X - R
        
        # Projection 2: Project onto unit diagonal subspace (P_U)
        Y_next = X.copy()
        np.fill_diagonal(Y_next, 1.0)
        Y_next = 0.5 * (Y_next + Y_next.T)
        
        # Check convergence
        diff = np.linalg.norm(Y_next - Y, ord='fro')
        Y = Y_next
        if diff < tol:
            break

    # Final guarantee of positive definiteness and unit diagonal
    eigvals, eigvecs = np.linalg.eigh(Y)
    if np.min(eigvals) < 1e-7:
        eigvals = np.maximum(eigvals, 1e-6)
        # Rescale so diagonal is exactly 1.0
        diag_scaling = np.sqrt(np.sum((eigvecs**2) * eigvals, axis=1))
        D_inv = np.diag(1.0 / np.maximum(diag_scaling, 1e-16))
        Y = D_inv @ (eigvecs @ np.diag(eigvals) @ eigvecs.T) @ D_inv
        np.fill_diagonal(Y, 1.0)
        Y = 0.5 * (Y + Y.T)

    if is_df:
        return pd.DataFrame(Y, index=idx, columns=cols)
    return Y
