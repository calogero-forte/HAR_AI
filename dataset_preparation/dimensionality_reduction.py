"""
Module: 			dimensionality_reduction.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.6
Last modify date: 	09/08/2026
"""
#------------------------------
# Import
#------------------------------
import logging
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
# pyrefly: ignore [missing-import]
import global_variables

logger = logging.getLogger(__name__)


def reduce_by_variance(
    df_i: pd.DataFrame,
    threshold_i: float = 0.1,
    X_test_i: Optional[pd.DataFrame] = None,
    X_val_i: Optional[pd.DataFrame] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    """
    Reduce the dataset dimensionality by removing features with low variance.

    Parameters
    ----------
    df_i : pd.DataFrame
        Input Pandas DataFrame containing numerical features for training.
    threshold_i : float, default=0.1
        Variance threshold cut-off i.e. the fraction of variance to remove.
    X_test_i : pd.DataFrame, optional
        Optional test dataset to transform.
    X_val_i : pd.DataFrame, optional
        Optional validation dataset to transform.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]
        DataFrame containing only features above the variance threshold.
        If test/val sets are provided, returns tuple of transformed DataFrames.
    """
    logger.info(f"Reducing features by variance threshold: threshold={threshold_i}, initial features={df_i.shape[1]}")
    vt = VarianceThreshold(threshold=threshold_i)
    X_var = vt.fit_transform(df_i)
    cols = df_i.columns[vt.get_support()]
    res_df = pd.DataFrame(data=X_var, columns=cols, index=df_i.index)

    if X_test_i is None and X_val_i is None:
        logger.info(f"Variance reduction complete: remaining features={res_df.shape[1]}")
        return res_df

    res_test_df = None
    if X_test_i is not None:
        X_test_var = vt.transform(X_test_i)
        res_test_df = pd.DataFrame(data=X_test_var, columns=cols, index=X_test_i.index)

    res_val_df = None
    if X_val_i is not None:
        X_val_var = vt.transform(X_val_i)
        res_val_df = pd.DataFrame(data=X_val_var, columns=cols, index=X_val_i.index)

    logger.info(f"Variance reduction complete: remaining features={res_df.shape[1]}")

    if res_test_df is not None and res_val_df is not None:
        return res_df, res_val_df, res_test_df
    elif res_test_df is not None:
        return res_df, res_test_df
    else:
        return res_df, res_val_df

#----------------------------------------

def reduce_by_pca(
    X_train_i: pd.DataFrame,
    X_test_i: pd.DataFrame,
    X_val_i: Optional[pd.DataFrame] = None,
    variance_i: float = 0.9
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Reduce the dataset dimensionality by applying PCA.

    Parameters
    ----------
    X_train_i : pd.DataFrame
        Input Pandas DataFrame containing train dataset.
    X_test_i : pd.DataFrame
        Input Pandas DataFrame containing test dataset.
    X_val_i : pd.DataFrame, optional
        Optional Pandas DataFrame containing validation dataset.
    variance_i : float, default=0.9
        Target variance ratio to preserve.

    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        Tuple containing the PCA-transformed features (X_train_pca, X_test_pca)
        or (X_train_pca, X_val_pca, X_test_pca) if X_val_i is provided.
    """
    val_log = f", val shape={X_val_i.shape}" if X_val_i is not None else ""
    logger.info(f"Applying PCA reduction with variance ratio={variance_i} on train shape={X_train_i.shape}, test shape={X_test_i.shape}{val_log}")

    pca = PCA(n_components=variance_i)

    # Train dataset
    X_train_pca = pca.fit_transform(X_train_i)
    cols = [f'feat_{i}' for i in range(X_train_pca.shape[1])]
    X_train_df = pd.DataFrame(data=X_train_pca, columns=cols, index=X_train_i.index)

    # Test dataset
    X_test_pca = pca.transform(X_test_i)
    X_test_df = pd.DataFrame(data=X_test_pca, columns=cols, index=X_test_i.index)

    if X_val_i is not None:
        X_val_pca = pca.transform(X_val_i)
        X_val_df = pd.DataFrame(data=X_val_pca, columns=cols, index=X_val_i.index)
        logger.info(f"PCA reduction complete: reduced train shape={X_train_df.shape}, reduced val shape={X_val_df.shape}, reduced test shape={X_test_df.shape}")
        return X_train_df, X_val_df, X_test_df

    logger.info(f"PCA reduction complete: reduced train shape={X_train_df.shape}, reduced test shape={X_test_df.shape}")
    return X_train_df, X_test_df

#----------------------------------------

def reduce_by_lda(
    X_train_i: pd.DataFrame,
    y_train_i: np.ndarray,
    X_test_i: pd.DataFrame,
    X_val_i: Optional[pd.DataFrame] = None,
    n_components_i: Optional[int] = None
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Reduce the dataset dimensionality by applying Linear Discriminant Analysis.

    Parameters
    ----------
    X_train_i : pd.DataFrame
        Input Pandas DataFrame containing train dataset.
    y_train_i : np.ndarray
        Input target labels for train dataset.
    X_test_i : pd.DataFrame
        Input Pandas DataFrame containing test dataset.
    X_val_i : pd.DataFrame, optional
        Optional Pandas DataFrame containing validation dataset.
    n_components_i : int, default=None
        Number of components to reduce the dataset to.

    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        Tuple containing the LDA-transformed features (X_train_lda, X_test_lda)
        or (X_train_lda, X_val_lda, X_test_lda) if X_val_i is provided.
    """
    val_log = f", val shape={X_val_i.shape}" if X_val_i is not None else ""
    logger.info(f"Applying LDA reduction with n_components={n_components_i} on train shape={X_train_i.shape}, test shape={X_test_i.shape}{val_log}")

    lda = LDA(n_components=n_components_i)

    # Train dataset
    X_train_lda = lda.fit_transform(X_train_i, y_train_i)
    cols = lda.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_lda, columns=cols, index=X_train_i.index)

    # Test dataset
    X_test_lda = lda.transform(X_test_i)
    X_test_df = pd.DataFrame(X_test_lda, columns=cols, index=X_test_i.index)

    if X_val_i is not None:
        X_val_lda = lda.transform(X_val_i)
        X_val_df = pd.DataFrame(X_val_lda, columns=cols, index=X_val_i.index)
        logger.info(f"LDA reduction complete: reduced train shape={X_train_df.shape}, reduced val shape={X_val_df.shape}, reduced test shape={X_test_df.shape}")
        return X_train_df, X_val_df, X_test_df

    logger.info(f"LDA reduction complete: reduced train shape={X_train_df.shape}, reduced test shape={X_test_df.shape}")
    return X_train_df, X_test_df

#----------------------------------------

def reduce_by_tsne(
    df_i: pd.DataFrame,
    n_components_i: int = 2,
    X_test_i: Optional[pd.DataFrame] = None,
    X_val_i: Optional[pd.DataFrame] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    """
    Reduce the dataset dimensionality by applying t-SNE.

    Parameters
    ----------
    df_i : pd.DataFrame
        Input Pandas DataFrame containing dataset to reduce.
    n_components_i : int, default=2
        Number of components to reduce the dataset to.
    X_test_i : pd.DataFrame, optional
        Optional test DataFrame.
    X_val_i : pd.DataFrame, optional
        Optional validation DataFrame.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]
        t-SNE transformed DataFrame or tuple of transformed DataFrames.
    """
    logger.info(f"Applying t-SNE reduction with n_components={n_components_i} on dataset shape={df_i.shape}")

    def _transform_tsne(data: pd.DataFrame) -> pd.DataFrame:
        perp = min(30, max(1, data.shape[0] - 1))
        t_sne = TSNE(n_components=n_components_i, random_state=global_variables.SEED, perplexity=perp, learning_rate=200)
        return pd.DataFrame(t_sne.fit_transform(data), columns=[f'tsne_{i}' for i in range(n_components_i)], index=data.index)

    res_train = _transform_tsne(df_i)

    if X_test_i is None and X_val_i is None:
        logger.info(f"t-SNE reduction complete: output shape={res_train.shape}")
        return res_train

    res_test_df = _transform_tsne(X_test_i) if X_test_i is not None else None
    res_val_df = _transform_tsne(X_val_i) if X_val_i is not None else None

    logger.info(f"t-SNE reduction complete: output shape={res_train.shape}")

    if res_test_df is not None and res_val_df is not None:
        return res_train, res_val_df, res_test_df
    elif res_test_df is not None:
        return res_train, res_test_df
    else:
        return res_train, res_val_df
    