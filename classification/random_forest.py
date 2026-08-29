#------------------------------
# Import
#------------------------------

from sklearn.ensemble import RandomForestClassifier

from .base_classifier import BaseClassifier
from typing import Tuple
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#------------------------------
# Classes
#------------------------------
class RandomForest(BaseClassifier):

    def __init__(self, n_estimators_i: int = 400, max_features_i: str = 'sqrt', 
        min_samples_split_i: int = 10, bootstrap_i: bool = True, oob_score_i: bool = True):
        """
        Constructor of the Random Forest class.
        
        Parameters
        ----------
        n_estimators_i : int, default=400
            The number of trees in the forest.
        max_features_i : str, default='sqrt'
            The number of features to consider when looking for the best split.
        min_samples_split_i : int, default=10
            The minimum number of samples required to split an internal node.
        bootstrap_i : bool, default=True
            Whether bootstrap samples are used when building trees.
        oob_score_i : bool, default=True
            Whether to use out-of-bag samples to estimate the generalization error.
        """
        self._n_estimators = n_estimators_i
        self._max_features = max_features_i
        self._min_samples_split = min_samples_split_i
        self._bootstrap = bootstrap_i
        self._oob_score = oob_score_i

        # Classifier instance
        self.rf = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_features=self._max_features,
            min_samples_split=self._min_samples_split,
            bootstrap=self._bootstrap,
            oob_score=self._oob_score
        )

        # Store predictions, metrics and confusion matrix
        self._y_pred: np.ndarray = None
        self._accuracy : float = None
        self._classification_report : str = None
        self._confusion_matrix: np.ndarray = None

    #----------------------------------------

    def train(self, X_train_i: np.ndarray | pd.DataFrame, y_train_i: np.ndarray) -> None:
        """
        Train the Random Forest classifier.
        
        Parameters
        ----------
        X_train_i : np.ndarray | pd.DataFrame
            The input features for training.
        y_train_i : np.ndarray
            The target labels for training.
        """
        self.rf.fit(X=X_train_i, y=y_train_i)

    #----------------------------------------

    def predict(self, X_test_i: np.ndarray | pd.DataFrame, y_test_i: np.ndarray) -> np.ndarray:
        """
        Predict and return target labels for the input features.
        Store internally the predictions, the metrics and the confuction matrix.
        
        Parameters
        ----------
        X_test_i : np.ndarray | pd.DataFrame
            The input features for prediction.
        y_test_i : np.ndarray
            The target labels for prediction.
        
        Returns
        -------
        np.ndarray
            The predicted target labels.
        """
        self._y_pred = self.rf.predict(X=X_test_i)
        self._classification_report = classification_report(y_true=y_test_i, y_pred=self._y_pred)
        self._accuracy = accuracy_score(y_true=y_test_i, y_pred=self._y_pred)
        self._confusion_matrix = confusion_matrix(y_true=y_test_i, y_pred=self._y_pred)

        return self._y_pred
        