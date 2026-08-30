"""
This module provides functions to reduce the 
dataset dimensionality 
"""
#------------------------------
# Import
#------------------------------
import logging
from matplotlib.artist import np
from typing import Optional
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
# pyrefly: ignore [missing-import]
import global_variables

logger = logging.getLogger(__name__)

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
    logger.info(f"Reducing features by variance threshold: threshold={threshold_i}, initial features={df_i.shape[1]}")
    vt = VarianceThreshold(threshold=threshold_i)
    X_var = vt.fit_transform(df_i)
    res_df = pd.DataFrame(data=X_var, columns=df_i.columns[vt.get_support()])
    
    # TODO: Handle error if variance threshold eliminates all features or invalid threshold provided
    logger.info(f"Variance reduction complete: remaining features={res_df.shape[1]}")
    return res_df

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
        DataFrame containing the PCA-transformed features (X_train_pca, X_test_pca)
    """
 
    logger.info(f"Applying PCA reduction with variance ratio={variance_i} on train shape={X_train_i.shape}, test shape={X_test_i.shape}")

    # Instance
    pca = PCA(n_components=variance_i)

    # Train dataset
    X_train_pca = pca.fit_transform(X_train_i)
    X_train_preserved = X_train_pca[ :, pca.explained_variance_ratio_ > variance_i ].copy()
    X_train_pca = pd.DataFrame( data=X_train_preserved, columns=[ f'feat_{i}' for i in range(X_train_preserved.shape[1]) ] )

    # Test dataset
    X_test_pca = pca.transform(X_test_i)
    X_test_preserved = X_test_pca[ :, pca.explained_variance_ratio_ > variance_i ].copy()
    X_test_pca = pd.DataFrame( X_test_preserved, columns=[f'feat_{i}' for i in range(X_test_preserved.shape[1])] )
    
    logger.info(f"PCA reduction complete: reduced train shape={X_train_pca.shape}, reduced test shape={X_test_pca.shape}")
    # Return both train and test datasets
    return X_train_pca, X_test_pca

#----------------------------------------

def reduce_by_lda(X_train_i: pd.DataFrame, X_test_i: pd.DataFrame, y_train_i: np.ndarray, 
    n_components_i: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reduce the dataset dimensionality by applying
    Linear Discriminant Analysis.

    Parameters
    ----------
    X_train _i : pd.DataFrame
        Input Pandas DataFrame containing train dataset.
    y_train_i : np.ndarray
        Input target labels for train dataset.
    X_test _i : pd.DataFrame
        Input Pandas DataFrame containing test dataset.
    n_components_i : int, default=None
        Number of components to reduce the dataset to.
        If None, it will be the number of classes - 1.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the LDA-transformed features (X_train_lda, X_test_lda)
    """

    logger.info(f"Applying LDA reduction with n_components={n_components_i} on train shape={X_train_i.shape}, test shape={X_test_i.shape}")

    # TODO: Handle error if n_components exceeds max allowed components min(n_classes - 1, n_features)

    # Instance
    lda = LDA(n_components=n_components_i)

    # Train dataset
    X_train_lda = lda.fit_transform(X_train_i, y_train_i)
    X_train_lda = pd.DataFrame(X_train_lda, columns=lda.get_feature_names_out())

    # Test dataset
    X_test_lda = lda.transform(X_test_i)
    X_test_lda = pd.DataFrame(X_test_lda, columns=lda.get_feature_names_out())
    
    logger.info(f"LDA reduction complete: reduced train shape={X_train_lda.shape}, reduced test shape={X_test_lda.shape}")
    return X_train_lda, X_test_lda

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
 
    logger.info(f"Applying t-SNE reduction with n_components={n_components_i} on dataset shape={df_i.shape}")
    t_sne = TSNE(n_components=n_components_i, random_state=global_variables.SEED, perplexity=30, learning_rate=200)
    res = t_sne.fit_transform(df_i)
    logger.info(f"t-SNE reduction complete: output shape={res.shape}")
    return res
    