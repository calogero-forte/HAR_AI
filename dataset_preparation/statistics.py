"""
This module contains util functions to 
compute statistcal metrics on Pandas DataFrames
"""

#------------------------------
# Imports
#------------------------------
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union

def study_correlation(
    df: pd.DataFrame, abs_val: bool = True, tri_sup: bool = True, threshold: Optional[float] = None
) -> Tuple[np.ndarray, Optional[List[int]]]:
    """
    Computes the correlation matrix for a DataFrame with options for absolute values
    and upper triangular filtering. Optionally identifies column indices exceeding a correlation threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input Pandas DataFrame containing numerical features.
    abs_val : bool, default=True
        If True, takes the absolute value of the correlation matrix.
    tri_sup : bool, default=True
        If True, retains only the upper triangular part of the matrix (where j > i),
        setting all other entries to zero.
    threshold : float, optional
        Correlation threshold. If provided, returns a list of column indices where correlation
        exceeds this threshold.

    Returns
    -------
    Tuple[np.ndarray, Optional[dict]]: 
        - np.ndarray: Processed correlation matrix.
        - Optional[dict]: Dictionary with keys as (j, i) = feature j correlated with feature i (where j > i), 
              and values as the correlation value, or None if threshold is not provided.
    """
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

    high_corr: Optional[dict] = None
    if threshold is not None:
        high_corr = {}
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if corr_matrix[i, j] >= threshold:
                    high_corr[(i, j)] = corr_matrix[i, j]

    return corr_matrix, high_corr

#----------------------------------------

def remove_correlated_features(
    X: Union[pd.DataFrame, np.ndarray], high_corr: Dict[Tuple[int, int], float]
) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
    """
    Removes correlated features from the feature matrix based on the high_corr dictionary.
    For each correlated pair (i, j) with i < j, if feature i is maintained, feature j is dropped.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Matrix of features (samples x features).
    high_corr : Dict[Tuple[int, int], float], optional
        Dictionary mapping feature index pairs (i, j) to their correlation values.
        If None, no features are removed.

    Returns
    -------
    Tuple[pd.DataFrame or np.ndarray, np.ndarray]
        - X_filtered : Matrix containing only the maintained features.
        - support : Boolean 1D NumPy array indicating which features were maintained (True) or dropped (False).
    """
    n_features = X.shape[1]
    support = np.ones(n_features, dtype=bool)

    for i in range(n_features):
        if support[i]:
            for j in range(i + 1, n_features):
                if (i, j) in high_corr or (j, i) in high_corr:
                    support[j] = False

    if isinstance(X, pd.DataFrame):
        X_filtered = X.iloc[:, support]
    else:
        X_filtered = X[:, support]

    return X_filtered, support