"""Principal Component Analysis (PCA) Factor Engine for Quantitative Portfolios.

Provides spectral decomposition of correlation and covariance matrices, factor loadings,
explained variance ratios, principal portfolio weights, factor return generation,
scree tests, and return reconstruction for risk modeling and dimension reduction.
"""

from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd


class PCAFactorEngine:
    """Principal Component Analysis (PCA) Factor Engine for asset returns.

    Decomposes the asset covariance or correlation matrix into orthogonal
    eigenmodes:
        Sigma = V Lambda V^T

    Where:
        Lambda = diag(lambda_1, ..., lambda_N) sorted descending.
        V = [v_1, ..., v_N] are the orthonormal factor loading eigenvectors.

    Attributes
    ----------
    use_correlation : bool
        Whether to perform PCA on the correlation matrix (True) or covariance matrix (False).
    periods_per_year : int
        Number of periods per year for annualization.
    is_fitted : bool
        True if the model has been fitted to return data.
    asset_names : list of str
        List of asset names from the fitted returns.
    n_features : int
        Number of assets/features.
    eigenvalues : np.ndarray
        Sorted eigenvalues in descending order.
    eigenvectors : np.ndarray
        Corresponding orthonormal eigenvectors (columns).
    explained_variance_ratio : np.ndarray
        Proportion of total variance explained by each principal component.
    cumulative_explained_variance : np.ndarray
        Cumulative proportion of total variance explained.
    covariance_matrix : pd.DataFrame
        Sample covariance matrix of the fitted data.
    correlation_matrix : pd.DataFrame
        Sample correlation matrix of the fitted data.
    """

    def __init__(
        self,
        use_correlation: bool = True,
        periods_per_year: int = 252
    ) -> None:
        """Initialize PCAFactorEngine.

        Parameters
        ----------
        use_correlation : bool, default True
            If True, standardize variables and decompose the correlation matrix.
            If False, decompose the unstandardized covariance matrix.
        periods_per_year : int, default 252
            Trading periods per year for annualizing stats.
        """
        self.use_correlation = use_correlation
        self.periods_per_year = periods_per_year
        self.is_fitted = False
        
        self.asset_names: List[str] = []
        self.n_features: int = 0
        self.eigenvalues: np.ndarray = np.array([])
        self.eigenvectors: np.ndarray = np.array([])
        self.explained_variance_ratio: np.ndarray = np.array([])
        self.cumulative_explained_variance: np.ndarray = np.array([])
        self.mean_returns: Optional[pd.Series] = None
        self.std_returns: Optional[pd.Series] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self._fitted_df: Optional[pd.DataFrame] = None

    def fit(self, returns_df: pd.DataFrame) -> 'PCAFactorEngine':
        """Fit the PCA model on historical asset returns.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Asset returns with observations as rows and assets as columns.

        Returns
        -------
        PCAFactorEngine
            Fitted instance of self.

        Raises
        ------
        TypeError
            If `returns_df` is not a pandas DataFrame.
        ValueError
            If `returns_df` is empty or has fewer than 2 rows or assets.
        """
        if not isinstance(returns_df, pd.DataFrame):
            raise TypeError("returns_df must be a pandas DataFrame.")
        if returns_df.empty or len(returns_df) < 2:
            raise ValueError("returns_df must have at least 2 observations.")
        if returns_df.shape[1] < 1:
            raise ValueError("returns_df must have at least 1 asset column.")

        self._fitted_df = returns_df.copy()
        self.asset_names = list(returns_df.columns)
        self.n_features = len(self.asset_names)
        self.mean_returns = returns_df.mean()
        self.std_returns = returns_df.std(ddof=1).replace(0.0, 1e-8)

        self.covariance_matrix = returns_df.cov()
        self.correlation_matrix = returns_df.corr()

        matrix_to_decompose = (
            self.correlation_matrix.values
            if self.use_correlation
            else self.covariance_matrix.values
        )

        # Eigen-decomposition for real symmetric matrix
        eigvals, eigvecs = np.linalg.eigh(matrix_to_decompose)

        # Sort descending by eigenvalue
        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues = np.maximum(eigvals[idx], 0.0)
        self.eigenvectors = eigvecs[:, idx]

        # Enforce deterministic sign convention: make the component with largest absolute loading positive
        for i in range(self.eigenvectors.shape[1]):
            max_idx = np.argmax(np.abs(self.eigenvectors[:, i]))
            if self.eigenvectors[max_idx, i] < 0:
                self.eigenvectors[:, i] *= -1.0

        total_variance = np.sum(self.eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio = self.eigenvalues / total_variance
        else:
            self.explained_variance_ratio = np.zeros_like(self.eigenvalues)

        self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)
        self.is_fitted = True
        return self

    def _check_is_fitted(self) -> None:
        """Verify that the engine has been fitted."""
        if not self.is_fitted:
            raise RuntimeError("PCAFactorEngine is not fitted. Call fit() first.")

    def get_eigenvalues(self) -> pd.Series:
        """Get the sorted eigenvalues corresponding to principal components.

        Returns
        -------
        pd.Series
            Eigenvalues labeled by PC index ('PC1', 'PC2', ...).
        """
        self._check_is_fitted()
        pc_labels = [f"PC{i+1}" for i in range(self.n_features)]
        return pd.Series(self.eigenvalues, index=pc_labels, name="eigenvalues")

    def get_explained_variance_ratio(self) -> pd.Series:
        """Get the explained variance ratio of each principal component.

        Returns
        -------
        pd.Series
            Variance ratio explained by each PC.
        """
        self._check_is_fitted()
        pc_labels = [f"PC{i+1}" for i in range(self.n_features)]
        return pd.Series(
            self.explained_variance_ratio,
            index=pc_labels,
            name="explained_variance_ratio"
        )

    def get_cumulative_explained_variance(self) -> pd.Series:
        """Get the cumulative explained variance ratio across principal components.

        Returns
        -------
        pd.Series
            Cumulative variance ratio up to each PC.
        """
        self._check_is_fitted()
        pc_labels = [f"PC{i+1}" for i in range(self.n_features)]
        return pd.Series(
            self.cumulative_explained_variance,
            index=pc_labels,
            name="cumulative_explained_variance"
        )

    def get_loadings(self, n_components: Optional[int] = None) -> pd.DataFrame:
        """Get the principal component factor loadings (eigenvector matrix V).

        Parameters
        ----------
        n_components : int, optional
            Number of top principal components to return. If None, returns all components.

        Returns
        -------
        pd.DataFrame
            Matrix of loadings with assets as rows and 'PC1', 'PC2', ... as columns.
        """
        self._check_is_fitted()
        k = self.n_features if n_components is None else min(max(1, n_components), self.n_features)
        cols = [f"PC{i+1}" for i in range(k)]
        return pd.DataFrame(
            self.eigenvectors[:, :k],
            index=self.asset_names,
            columns=cols
        )

    def get_principal_portfolios(
        self,
        n_components: Optional[int] = None,
        standardize_weights: bool = True,
        normalization: str = 'unit_sum',
        normalize: Optional[str] = None
    ) -> pd.DataFrame:
        """Compute the principal portfolio asset allocation weights.

        Each principal component defines an eigen-portfolio that isolates
        orthogonal risk factors.

        Parameters
        ----------
        n_components : int, optional
            Number of top principal portfolios to return.
        standardize_weights : bool, default True
            If True, normalize weights according to `normalization`.
        normalization : {'unit_sum', 'unit_l1', 'unit_l2'}, default 'unit_sum'
            Normalization scheme:
            - 'unit_sum': weights sum to 1 (w / sum(w)).
            - 'unit_l1': gross exposure sums to 1 (w / sum(|w|)).
            - 'unit_l2': Euclidean norm equals 1 (w / ||w||_2).
        normalize : str, optional
            Alias for `normalization`.

        Returns
        -------
        pd.DataFrame
            Principal portfolio weights with assets as rows and PC labels as columns.
        """
        self._check_is_fitted()
        norm_method = normalize if normalize is not None else normalization
        k = self.n_features if n_components is None else min(max(1, n_components), self.n_features)
        weights = self.eigenvectors[:, :k].copy()

        if standardize_weights:
            if norm_method == 'unit_sum':
                sums = np.sum(weights, axis=0)
                sums[np.abs(sums) < 1e-12] = 1.0
                weights = weights / sums
            elif norm_method == 'unit_l1':
                sums = np.sum(np.abs(weights), axis=0)
                sums[sums < 1e-12] = 1.0
                weights = weights / sums
            elif norm_method == 'unit_l2':
                norms = np.linalg.norm(weights, axis=0)
                norms[norms < 1e-12] = 1.0
                weights = weights / norms
            else:
                raise ValueError(f"Unknown normalization '{norm_method}'. Must be 'unit_sum', 'unit_l1', or 'unit_l2'.")

        cols = [f"PC{i+1}" for i in range(k)]
        return pd.DataFrame(weights, index=self.asset_names, columns=cols)

    def transform(
        self,
        returns_df: Optional[pd.DataFrame] = None,
        n_components: Optional[int] = None
    ) -> pd.DataFrame:
        """Project returns onto the principal component factor space (factor returns).

        Parameters
        ----------
        returns_df : pd.DataFrame, optional
            Returns DataFrame to project. If None, projects the fitted returns.
        n_components : int, optional
            Number of principal component factors to retain.

        Returns
        -------
        pd.DataFrame
            Principal component factor scores / returns with columns 'PC1', 'PC2', ...
        """
        self._check_is_fitted()
        df = self._fitted_df if returns_df is None else returns_df
        if not isinstance(df, pd.DataFrame):
            raise TypeError("returns_df must be a pandas DataFrame.")

        k = self.n_features if n_components is None else min(max(1, n_components), self.n_features)

        # Center / standardize
        if self.use_correlation:
            X = (df[self.asset_names] - self.mean_returns) / self.std_returns
        else:
            X = df[self.asset_names] - self.mean_returns

        factors = X.values @ self.eigenvectors[:, :k]
        cols = [f"PC{i+1}" for i in range(k)]
        return pd.DataFrame(factors, index=df.index, columns=cols)

    def principal_portfolio_returns(
        self,
        returns_df: Optional[pd.DataFrame] = None,
        n_components: Optional[int] = None,
        standardize_weights: bool = True
    ) -> pd.DataFrame:
        """Compute the realized returns of the principal portfolios.

        Multiplies asset returns by the principal portfolio weights:
            R_port = R_assets @ W_principal

        Parameters
        ----------
        returns_df : pd.DataFrame, optional
            Asset returns to evaluate. If None, uses fitted returns.
        n_components : int, optional
            Number of principal portfolios to generate returns for.
        standardize_weights : bool, default True
            Whether to normalize portfolio weights (unit L1 gross exposure).

        Returns
        -------
        pd.DataFrame
            Time series of principal portfolio returns.
        """
        self._check_is_fitted()
        df = self._fitted_df if returns_df is None else returns_df
        weights_df = self.get_principal_portfolios(
            n_components=n_components,
            standardize_weights=standardize_weights,
            normalization='unit_l1' if standardize_weights else 'unit_l2'
        )
        port_returns = df[self.asset_names].values @ weights_df.values
        return pd.DataFrame(
            port_returns,
            index=df.index,
            columns=weights_df.columns
        )

    def reconstruct_returns(
        self,
        returns_df: Optional[pd.DataFrame] = None,
        n_components: Optional[int] = None
    ) -> pd.DataFrame:
        """Reconstruct original asset returns from a subset of principal components.

        R_reconstructed = F_{1:k} V_{1:k}^T + mu

        Parameters
        ----------
        returns_df : pd.DataFrame, optional
            Returns DataFrame to reconstruct. If None, reconstructs fitted returns.
        n_components : int, optional
            Number of top principal components used for reconstruction.
            If None, uses all components (lossless reconstruction).

        Returns
        -------
        pd.DataFrame
            Reconstructed returns with original asset columns and index.
        """
        self._check_is_fitted()
        df = self._fitted_df if returns_df is None else returns_df
        k = self.n_features if n_components is None else min(max(1, n_components), self.n_features)

        # Factor scores
        factors = self.transform(df, n_components=k).values # (T, k)
        vecs = self.eigenvectors[:, :k] # (N, k)

        # Reconstructed normalized values
        reconstructed = factors @ vecs.T # (T, N)

        # Invert standardization / centering
        if self.use_correlation:
            reconstructed = reconstructed * self.std_returns.values + self.mean_returns.values
        else:
            reconstructed = reconstructed + self.mean_returns.values

        return pd.DataFrame(
            reconstructed,
            index=df.index,
            columns=self.asset_names
        )

    def scree_test(self, threshold: float = 0.80) -> int:
        """Determine number of principal components needed to explain a variance threshold.

        Parameters
        ----------
        threshold : float, default 0.80
            Target cumulative variance explained (e.g. 0.80 for 80%).

        Returns
        -------
        int
            Minimum number of principal components required.
        """
        self._check_is_fitted()
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be strictly between 0 and 1.")

        idx = np.where(self.cumulative_explained_variance >= threshold)[0]
        if len(idx) > 0:
            return int(idx[0] + 1)
        return self.n_features

    def dimension_reduction(
        self,
        returns_df: Optional[pd.DataFrame] = None,
        threshold: float = 0.80
    ) -> Tuple[pd.DataFrame, int]:
        """Perform statistical dimension reduction retaining components up to variance threshold.

        Parameters
        ----------
        returns_df : pd.DataFrame, optional
            Returns DataFrame to transform. If None, transforms fitted returns.
        threshold : float, default 0.80
            Cumulative variance threshold to retain.

        Returns
        -------
        factors_df : pd.DataFrame
            Reduced factor matrix with retained principal components.
        k_retained : int
            Number of components retained.
        """
        k = self.scree_test(threshold=threshold)
        factors = self.transform(returns_df=returns_df, n_components=k)
        return factors, k
