"""
This module contains util functions to 
compute statistcal metrics on Pandas DataFrames
"""

#------------------------------
# Imports
#------------------------------
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union

logger = logging.getLogger(__name__)

def compute_correlation(
    df: pd.DataFrame, abs_val: bool = True, tri_sup: bool = True) -> Tuple[np.ndarray, Optional[List[int]]]:
    """
    Computes the correlation matrix for a DataFrame with options for absolute values
    and upper triangular filtering. 

    Parameters
    ----------
    df : pd.DataFrame
        Input Pandas DataFrame containing numerical features.
    abs_val : bool, default=True
        If True, takes the absolute value of the correlation matrix.
    tri_sup : bool, default=True
        If True, retains only the upper triangular part of the matrix (where j > i),
        setting all other entries to zero.

    Returns
    -------
    Tuple[np.ndarray, Optional[dict]]: 
        - np.ndarray: Processed correlation matrix.
    """
    logger.info(f"Computing correlation matrix for DataFrame of shape {df.shape} (abs_val={abs_val}, tri_sup={tri_sup})")
    corr_matrix = (df.corr()).to_numpy()
    if abs_val:
        corr_matrix = np.abs(corr_matrix)
    
    if tri_sup:
        corr_sup = np.zeros(corr_matrix.shape)
        for i in range(0, corr_sup.shape[0]):
            for j in range(0, corr_sup.shape[0]):
                if j > i:
                    corr_sup[i, j] = corr_matrix[i, j]
        corr_matrix = corr_sup

    return corr_matrix

#----------------------------------------

def remove_correlated_features(
    X: Union[pd.DataFrame, np.ndarray],
    corr_matrix: np.ndarray,
    threshold: float = 0.8,
) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
    """
    Removes correlated features from the feature matrix based on a correlation matrix (ndarray).

    A feature j is dropped only if it is correlated with a feature i that is 
    currently maintained in the dataset. If feature i was previously dropped, 
    feature j is not dropped due to feature i, ensuring features are not unnecessarily 
    removed if their correlated counterparts are no longer present.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Matrix of features (samples x features).
    corr_matrix : np.ndarray
        Correlation matrix of shape (n_features, n_features).
    threshold : float, default=0.8
        Correlation threshold. Feature pairs with absolute correlation >= threshold
        are identified as correlated pairs.

    Returns
    -------
    Tuple[pd.DataFrame or np.ndarray, np.ndarray]
        - X_filtered : Matrix containing only the maintained features.
        - support : Boolean 1D NumPy array indicating which features were maintained (True) or dropped (False).
    """
    n_features = X.shape[1]
    logger.info(f"Removing correlated features: initial n_features={n_features}, threshold={threshold}")

    high_corr = {}
    n_cols = corr_matrix.shape[0]
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            val = max(abs(corr_matrix[i, j]), abs(corr_matrix[j, i]))
            if val >= threshold:
                high_corr[(i, j)] = float(
                    corr_matrix[i, j]
                    if abs(corr_matrix[i, j]) >= abs(corr_matrix[j, i])
                    else corr_matrix[j, i]
                )

    # Normalize pair ordering so (i, j) always has i < j
    corr_pairs = set()
    for (a, b) in high_corr.keys():
        if a != b:
            corr_pairs.add((min(a, b), max(a, b)))

    logger.info(f"Found {len(corr_pairs)} correlated feature pairs with correlation >= {threshold}")

    support = np.ones(n_features, dtype=bool)

    for i in range(n_features):
        if support[i]:
            for j in range(i + 1, n_features):
                if (i, j) in corr_pairs:
                    support[j] = False

    if isinstance(X, pd.DataFrame):
        X_filtered = X.iloc[:, support]
    else:
        X_filtered = X[:, support]

    dropped_count = n_features - np.sum(support)
    logger.info(f"Removed correlated features: dropped {dropped_count}/{n_features} features, remaining={X_filtered.shape[1]}")

    return X_filtered, support

#--------------------------------------------------------------------------------

if __name__ == '__main__':

    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    
    X = np.random.rand(4, 4)
    corr = np.array([[1, 0.78, 0.5, 0.3], [0, 1, 0.8, 0.1], [0, 0, 1, 0.5], [0, 0, 0, 1]])
    print(X)
    X_filtered, support = remove_correlated_features(X, corr, threshold=0.7)
    print(X_filtered)
    print(support)

    
    