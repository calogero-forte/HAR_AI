"""
Module: 			random_forest.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.4
Last modify date: 	09/06/2026
"""

#------------------------------
# Import
#------------------------------

import logging
from typing import Tuple, Optional
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# pyrefly: ignore [missing-import]
from .base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

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
        super().__init__()

        # Classifier parameters
        self._n_estimators = n_estimators_i
        self._max_features = max_features_i
        self._min_samples_split = min_samples_split_i
        self._bootstrap = bootstrap_i
        self._oob_score = oob_score_i

        logger.info(
            f"Initializing RandomForest classifier (n_estimators={self._n_estimators}, "
            f"max_features={self._max_features}, min_samples_split={self._min_samples_split}, "
            f"bootstrap={self._bootstrap}, oob_score={self._oob_score})"
        )

        # Classifier instance
        self._classifier = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_features=self._max_features,
            min_samples_split=self._min_samples_split,
            bootstrap=self._bootstrap,
            oob_score=self._oob_score
        )

    #----------------------------------------

    def predict(self, X_test_i: np.ndarray | pd.DataFrame, y_true_i: np.ndarray | pd.Series, **kwargs) -> np.ndarray:
        """
        Override of the base predict method
        """
        self._y_pred = super().predict(X_test_i, y_true_i, **kwargs)
        return self._y_pred

    #----------------------------------------

    def cross_evaluate(self, X_train_i: np.ndarray | pd.DataFrame, y_train_i: np.ndarray | pd.Series, cv_i: int = 5, **kwargs) -> None:
        """
        Override of the base cross_evaluate method
        """
        dataset_shape = getattr(X_train_i, "shape", len(X_train_i))
        logger.info(f"Starting cross-validation ({cv_i} folds) on dataset shape {dataset_shape}...")
        grid = GridSearchCV(
            self._classifier,
            self._param_grid,
            cv=cv_i,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_train_i, y_train_i)
        self._best_params = grid.best_params_
        self._best_score = float(grid.best_score_)
        self._best_estimator = grid.best_estimator_

        logger.info(f"Cross-evaluation completed. Best score: {self._best_score:.4f}")
        logger.info(f"Best parameters: {self._best_params}")

    