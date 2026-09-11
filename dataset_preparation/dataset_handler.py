"""
Module: 			dataset_handler.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.12
Last modify date: 	09/11/2026
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
# pyrefly: ignore [missing-import]
import global_variables

logger = logging.getLogger(__name__)


class DatasetHandler:
    """
    This class maintains the UCI HAR Dataset 
    in DataFrames for Train and Test and 
    prodides method to get the data, to manage them and
    to save them again in CSV.
    """

    def __init__(self, train_dataset_path_i: str, test_dataset_path_i: str, 
        val_split_i: float = 0.0, split_by_subject_i: bool = True) -> None:

        #----------------------------------------
        # Load training set in a Pandas Dataframe
        logger.info(f"Loading training dataset from: {train_dataset_path_i}")
        df = pd.read_csv(train_dataset_path_i)

        # Store target numbers and labels
        self._targets_names = list( df['target_name'].unique() )
        feat_name_orig = df.columns.to_numpy(dtype=str)
        self._features_names = feat_name_orig[1 : -2].copy()

        # Placeholder for Train and validation datasets
        self._Xtrain = None
        self._ytrain = None
        self._Xval = None
        self._yval = None

        # Splitting dataset
        if(val_split_i <= 0.0):
            df_train = df.copy()
            logger.info("No val set created")
        else:
            df_train, df_val = DatasetHandler.__split_train_val(df, val_split_i, split_by_subject_i)
            
        
        # Build and store X_train and features_names
        # Remove subject id, target name (which is string so gives problems) and target
        self._X_train = df_train.drop(labels=['subject_id', 'target_name', 'target'], axis=1)
        logger.info(f"Training dataset loaded: {self._X_train.shape[0]} samples, {self._X_train.shape[1]} features")
        # Store train targets
        self._y_train = df_train['target'].to_numpy(dtype=int)
        
        if(val_split_i > 0.0):
            self._X_val = df_val.drop(labels=['subject_id', 'target_name', 'target'], axis=1)
            self._y_val = df_val['target'].to_numpy(dtype=int)
            logger.info(f"Validation dataset loaded: {self._X_val.shape[0]} samples, {self._X_val.shape[1]} features")

        #----------------------------------------
        # Load test set in a Pandas Dataframe
        logger.info(f"Loading test dataset from: {test_dataset_path_i}")
        df_test = pd.read_csv(test_dataset_path_i)

        # Store test targets
        self._y_test = df_test['target'].to_numpy(dtype=int)

        # Build and store X_test (feature names are already stored from train_set)
        # Remove subject id, target name (which is string so gives problems) and target
        self._X_test = df_test.drop(labels=['subject_id', 'target_name', 'target'], axis=1)
        logger.info(f"Test dataset loaded: {self._X_test.shape[0]} samples, {self._X_test.shape[1]} features")

        # Maintain a list of the modified train and test dataframes
        self.__train_modified = list[pd.DataFrame]()
        self.__val_modified = list[pd.DataFrame]()
        self.__test_modified = list[pd.DataFrame]()

    #----------------------------------------

    def get_train_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X_train, self._y_train

    #----------------------------------------

    def get_val_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X_val, self._y_val

    #----------------------------------------

    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X_test, self._y_test

    #----------------------------------------

    def get_features_names(self) -> list[str]:
        return self._features_names
    
    #----------------------------------------

    def get_targets_names(self) -> list[str]:
        return self._targets_names

    #----------------------------------------
    
    def update_train_dataset(self, df: pd.DataFrame) -> None:
        self.__train_modified.append(df.copy())
        logger.info(f"Updated modified train dataset list (buffer size: {len(self.__train_modified)}, shape: {df.shape})")

    #----------------------------------------

    def update_val_dataset(self, df: pd.DataFrame) -> None:
        self.__val_modified.append(df.copy())
        logger.info(f"Updated modified val dataset list (buffer size: {len(self.__val_modified)}, shape: {df.shape})")

    #----------------------------------------

    def update_test_dataset(self, df: pd.DataFrame) -> None:
        self.__test_modified.append(df.copy())
        logger.info(f"Updated modified test dataset list (buffer size: {len(self.__test_modified)}, shape: {df.shape})")

    #----------------------------------------

    def get_train_dataset_by_index(self, idx_i: int) -> pd.DataFrame:
        if(idx_i == -1 and len(self.__train_modified) > 0):
            return self.__train_modified[-1]
        if( idx_i >= 0 and idx_i < len(self.__train_modified) ):
            return self.__train_modified[idx_i]
        else:
            logger.error(f"Index {idx_i} is out of bounds for train dataset with size {len(self.__train_modified)}")
            
    #----------------------------------------

    def get_val_dataset_by_index(self, idx_i: int) -> pd.DataFrame:
        if(idx_i == -1 and len(self.__val_modified) > 0):
            return self.__val_modified[-1]
        if( idx_i >= 0 and idx_i < len(self.__val_modified) ):
            return self.__val_modified[idx_i]
        else:
            logger.error(f"Index {idx_i} is out of bounds for val dataset with size {len(self.__val_modified)}")

    #----------------------------------------

    def get_test_dataset_by_index(self, idx_i: int) -> pd.DataFrame:
        if(idx_i == -1 and len(self.__test_modified) > 0):
            return self.__test_modified[-1]
        if( idx_i >= 0 and idx_i < len(self.__test_modified) ):
            return self.__test_modified[idx_i]
        else:
            logger.error(f"Index {idx_i} is out of bounds for test dataset with size {len(self.__test_modified)}")

    #----------------------------------------

    @staticmethod
    def __split_train_val(df_train_i: pd.DataFrame, val_split_i: float = 0.2, split_by_subject_i: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the input training dataframe into a training set and a validation set.

        Parameters
        ----------
        df_train_i : pd.DataFrame
            The input training dataframe to split
        val_split_i : float, default=0.2
            The proportion of dataset to allocate to validation set
        split_by_subject_i : bool, default=True
            If True, split by subject_id so no subject_id in train remains in validation set.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple (df_train, df_val)
        """
        logger.info(f"Splitting dataset (shape: {df_train_i.shape}) into train/val (val_split={val_split_i}, split_by_subject={split_by_subject_i})")

        if split_by_subject_i:
            if 'subject_id' in df_train_i.columns:
                gss = GroupShuffleSplit(n_splits=1, test_size=val_split_i, random_state=global_variables.SEED)
                train_idx, val_idx = next(gss.split(df_train_i, groups=df_train_i['subject_id']))
                df_train = df_train_i.iloc[train_idx].copy()
                df_val = df_train_i.iloc[val_idx].copy()
            else:
                logger.warning("'subject_id' column not found in dataframe. Falling back to standard stratified split.")
                stratify_col = df_train_i['target'] if 'target' in df_train_i.columns else None
                df_train, df_val = train_test_split(
                    df_train_i,
                    test_size=val_split_i,
                    random_state=global_variables.SEED,
                    stratify=stratify_col
                )
        else:
            stratify_col = df_train_i['target'] if 'target' in df_train_i.columns else None
            df_train, df_val = train_test_split(
                df_train_i,
                test_size=val_split_i,
                random_state=global_variables.SEED,
                stratify=stratify_col
            )

        logger.info(f"Split complete: train shape={df_train.shape}, val shape={df_val.shape}")
        return df_train, df_val