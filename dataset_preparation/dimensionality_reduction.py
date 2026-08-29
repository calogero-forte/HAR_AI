"""
This module provides functions to reduce the 
dataset dimensionality 
"""
#------------------------------
# Import
#------------------------------
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
# pyrefly: ignore [missing-import]
import global_variables

def reduce_by_variance(df_i: pd.DataFrame, threshold_i: float = 0.1) -> pd.DataFrame:
    """
    Reduce the dataset dimensionality by removing features with low variance.

    Parameters
    ----------
    df_i : pd.DataFrame
        Input Pandas DataFrame containing numerical features.
    threshold_i : float, default=0.1
        Variance threshold cut-off i.e. the fraction of variance to remove

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the features with variance above the threshold.
    """
    vt = VarianceThreshold(threshold=threshold_i)
    X_var = vt.fit_transform(df_i)

    return pd.DataFrame(data=X_var, columns=df_i.columns[vt.get_support()])

#----------------------------------------

def reduce_by_pca(X_train_i: pd.DataFrame, X_test_i: pd.DataFrame, variance_i: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reduce the dataset dimensionality by applying PCA.

    Parameters
    ----------
    X_train _i : pd.DataFrame
        Input Pandas DataFrame containing train dataset.
    X_test _i : pd.DataFrame
        Input Pandas DataFrame containing test dataset.
    variance_i : float, default=0.9
        Target variance ratio to preserve.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the PCA-transformed features.
    """
 
    pca = PCA(n_components=variance_i)

    # Train dataset
    X_train_pca = pca.fit_transform(X_train_i)
    X_train_preserved = X_train_pca[ :, pca.explained_variance_ratio_ > variance_i ].copy()
    X_train_pca = pd.DataFrame( data=X_train_preserved, columns=[ f'feat_{i}' for i in range(X_train_preserved.shape[1]) ] )

    # Test dataset
    X_test_pca = pca.transform(X_test_i)
    X_test_preserved = X_test_pca[ :, pca.explained_variance_ratio_ > variance_i ].copy()
    X_test_pca = pd.DataFrame( X_test_preserved, columns=[f'feat_{i}' for i in range(X_test_preserved.shape[1])] )
    
    # Return both train and test datasets
    return X_train_pca, X_test_pca

#----------------------------------------
 
def reduce_by_tsne(df_i: pd.DataFrame, n_components_i: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reduce the dataset dimensionality by applying t-SNE.

    Parameters
    ----------
    df _i : pd.DataFrame
        Input Pandas DataFrame containing the dataset to reduce.
    n_components_i : int, default=2
        Number of components to reduce the dataset to.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the t-SNE-transformed features.
    """
 

    t_sne = TSNE(n_components=n_components_i, random_state=global_variables.SEED, perplexity=30, learning_rate=200)
    return t_sne.fit_transform(df_i)
    