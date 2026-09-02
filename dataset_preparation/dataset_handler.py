from pandas._libs import indexing
from pandas._libs import indexing
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DatasetHandler:
    """
    This class maintains the UCI HAR Dataset 
    in DataFrames for Train and Test and 
    prodides method to get the data, to manage them and
    to save them again in CSV.
    """

    def __init__(self, train_dataset_path_i: str, test_dataset_path_i: str):

        #----------------------------------------
        # Load training set in a Pandas Dataframe
        logger.info(f"Loading training dataset from: {train_dataset_path_i}")
        df_train = pd.read_csv(train_dataset_path_i)

        # Store target numbers and labels
        self._y_train = df_train['target'].to_numpy(dtype=int)
        self._targets_names = list( df_train['target_name'].unique() )
        
        # Build and store X_train and features_names
        # Remove subject id, target name (which is string so gives problems) and target
        self._X_train = df_train.drop(labels=['subject_id', 'target_name', 'target'], axis=1)
        feat_name_orig = df_train.columns.to_numpy(dtype=str)
        self._features_names = feat_name_orig[1 : -2].copy()
        logger.info(f"Training dataset loaded: {self._X_train.shape[0]} samples, {self._X_train.shape[1]} features")

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
        self.__test_modified = list[pd.DataFrame]()

    #----------------------------------------

    def get_train_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X_train, self._y_train

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
            
        

    def get_test_dataset_by_index(self, idx_i: int) -> pd.DataFrame:
        if(idx_i == -1 and len(self.__test_modified) > 0):
            return self.__test_modified[-1]
        if( idx_i >= 0 and idx_i < len(self.__test_modified) ):
            return self.__test_modified[idx_i]
        else:
            logger.error(f"Index {idx_i} is out of bounds for test dataset with size {len(self.__test_modified)}")
            