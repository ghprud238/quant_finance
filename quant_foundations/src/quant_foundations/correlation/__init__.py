"""Module 3: Correlation & Covariance Engine, Shrinkage, PCA, Clustering.

Provides high-performance estimation, regularization, spectral decomposition,
and hierarchical clustering of asset correlation and covariance matrices.
"""

from quant_foundations.correlation.matrix import (
    compute_correlation_matrix,
    compute_covariance_matrix,
    ledoit_wolf_shrinkage,
    ewma_covariance,
    rolling_correlation,
    is_positive_definite,
    nearest_correlation_matrix,
)

from quant_foundations.correlation.pca import (
    PCAFactorEngine,
)

from quant_foundations.correlation.clustering import (
    correlation_distance,
    hierarchical_correlation_clustering,
    quasi_diagonalize,
)

__all__ = [
    "compute_correlation_matrix",
    "compute_covariance_matrix",
    "ledoit_wolf_shrinkage",
    "ewma_covariance",
    "rolling_correlation",
    "is_positive_definite",
    "nearest_correlation_matrix",
    "PCAFactorEngine",
    "correlation_distance",
    "hierarchical_correlation_clustering",
    "quasi_diagonalize",
]
