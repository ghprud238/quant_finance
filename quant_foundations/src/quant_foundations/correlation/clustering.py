"""Hierarchical Correlation Clustering and Tree Seriation (Quasi-Diagonalization).

Implements correlation distance metrics, agglomerative hierarchical linkage,
and Hierarchical Risk Parity (HRP) quasi-diagonalization seriation to cluster
and order assets along the correlation matrix diagonal.
"""

from typing import Optional, Union, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage


def correlation_distance(
    corr_matrix: Union[np.ndarray, pd.DataFrame]
) -> Union[np.ndarray, pd.DataFrame]:
    """Compute the angular correlation distance matrix from a correlation matrix.

    Formula:
        d_{i, j} = \\sqrt{0.5 \\cdot (1 - \\rho_{i, j})}

    Properties:
        - d_{i, i} = 0 (identity)
        - d_{i, j} \\in [0, 1] for \\rho_{i, j} \\in [-1, 1]
        - d_{i, j} = d_{j, i} (symmetry)
        - Satisfies the triangle inequality (true metric distance).

    Parameters
    ----------
    corr_matrix : Union[np.ndarray, pd.DataFrame]
        Square correlation matrix of shape (N, N).

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Distance matrix of the same type and structure as `corr_matrix`.

    Raises
    ------
    TypeError
        If `corr_matrix` is not a numpy ndarray or pandas DataFrame.
    ValueError
        If `corr_matrix` is not a 2D square matrix.
    """
    is_df = isinstance(corr_matrix, pd.DataFrame)
    if is_df:
        mat = corr_matrix.values.astype(float)
        idx = corr_matrix.index
        cols = corr_matrix.columns
    elif isinstance(corr_matrix, np.ndarray):
        mat = corr_matrix.astype(float)
    else:
        raise TypeError("corr_matrix must be a numpy ndarray or pandas DataFrame.")

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("corr_matrix must be a 2D square matrix.")

    # Clip correlations to [-1.0, 1.0] for numerical safety
    clipped_mat = np.clip(mat, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - clipped_mat), 0.0))
    np.fill_diagonal(dist, 0.0)

    if is_df:
        return pd.DataFrame(dist, index=idx, columns=cols)
    return dist


def hierarchical_correlation_clustering(
    corr_matrix: Union[np.ndarray, pd.DataFrame],
    method: str = 'ward'
) -> np.ndarray:
    """Perform hierarchical agglomerative clustering on a correlation matrix.

    Transforms the correlation matrix into a distance metric space using
    `correlation_distance` and computes the hierarchical linkage tree.

    Parameters
    ----------
    corr_matrix : Union[np.ndarray, pd.DataFrame]
        Square correlation matrix of shape (N, N).
    method : {'ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'}, default 'ward'
        Linkage algorithm to use.

    Returns
    -------
    np.ndarray
        Linkage matrix Z of shape (N-1, 4) containing the hierarchical cluster tree.

    Raises
    ------
    ValueError
        If `corr_matrix` has fewer than 2 assets or if `method` is invalid.
    """
    valid_methods = {'ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'}
    if method not in valid_methods:
        raise ValueError(f"Invalid linkage method '{method}'. Must be one of {valid_methods}.")

    dist = correlation_distance(corr_matrix)
    if isinstance(dist, pd.DataFrame):
        dist_mat = dist.values
    else:
        dist_mat = dist

    N = dist_mat.shape[0]
    if N < 2:
        raise ValueError("Hierarchical clustering requires at least 2 assets.")

    # Convert square distance matrix to condensed form for scipy linkage
    # Ensure exact zero diagonal and symmetry
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    condensed_dist = squareform(dist_mat, checks=False)

    linkage_mat = linkage(condensed_dist, method=method)
    return linkage_mat


def quasi_diagonalize(
    linkage_matrix: np.ndarray,
    labels: Optional[Union[List[str], pd.Index]] = None
) -> Union[List[int], List[str]]:
    """Reorder assets using Hierarchical Tree Seriation (Quasi-Diagonalization).

    Sorts the dendrogram leaves so that similar/highly correlated assets
    are placed adjacently along the diagonal of the reordered matrix.
    Follows Marcos López de Prado (2016) Hierarchical Risk Parity (HRP) methodology.

    Reference:
        López de Prado, M. (2016). "Building Diversified Portfolios that Outperform
        Out of Sample". The Journal of Portfolio Management, 42(4), 59-69.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix of shape (N-1, 4) returned by `hierarchical_correlation_clustering`.
    labels : list of str or pd.Index, optional
        Optional list of asset labels corresponding to original matrix indices.
        If provided, the returned list contains asset labels; otherwise integer indices.

    Returns
    -------
    Union[List[int], List[str]]
        Ordered list of asset indices or labels in quasi-diagonal sequence.

    Raises
    ------
    TypeError
        If `linkage_matrix` is not a numpy ndarray.
    ValueError
        If `linkage_matrix` has invalid shape or contents.
    """
    linkage_mat = np.asarray(linkage_matrix)
    if linkage_mat.ndim != 2 or linkage_mat.shape[1] != 4:
        raise ValueError("linkage_matrix must have shape (N-1, 4).")

    num_items = linkage_mat.shape[0] + 1
    sort_ix = [int(linkage_mat[-1, 0]), int(linkage_mat[-1, 1])]

    while max(sort_ix) >= num_items:
        new_sort_ix = []
        for item in sort_ix:
            if item >= num_items:
                row = item - num_items
                new_sort_ix.append(int(linkage_mat[row, 0]))
                new_sort_ix.append(int(linkage_mat[row, 1]))
            else:
                new_sort_ix.append(item)
        sort_ix = new_sort_ix

    if labels is not None:
        label_list = list(labels)
        if len(label_list) != num_items:
            raise ValueError(
                f"Length of labels ({len(label_list)}) must match number of items ({num_items})."
            )
        return [label_list[i] for i in sort_ix]

    return sort_ix
