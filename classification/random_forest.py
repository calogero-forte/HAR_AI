"""
Module: 			random_forest.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.11
Last modify date: 	09/11/2026
"""

#------------------------------
# Import
#------------------------------

import logging
from typing import Tuple, Optional
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# pyrefly: ignore [missing-import]
import global_variables
# pyrefly: ignore [missing-import]
from .base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

#------------------------------
# Classes
#------------------------------
class RandomForest(BaseClassifier):

    def __init__(self, n_estimators_i: int = 400, max_features_i: str = 'sqrt', 
        min_samples_split_i: int = 10, bootstrap_i: bool = True, oob_score_i: bool = True,
        random_state_i: int = global_variables.SEED):
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
        random_state_i : int, default=global_variables.SEED
            Controls the randomness of the bootstrapping and feature sampling.
        """
        super().__init__()

        # Classifier parameters
        self._n_estimators = n_estimators_i
        self._max_features = max_features_i
        self._min_samples_split = min_samples_split_i
        self._bootstrap = bootstrap_i
        self._oob_score = oob_score_i
        self._random_state = random_state_i

        logger.info(
            f"Initializing RandomForest classifier (n_estimators={self._n_estimators}, "
            f"max_features={self._max_features}, min_samples_split={self._min_samples_split}, "
            f"bootstrap={self._bootstrap}, oob_score={self._oob_score}, "
            f"random_state={self._random_state})"
        )

        # Classifier instance
        self._classifier = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_features=self._max_features,
            min_samples_split=self._min_samples_split,
            bootstrap=self._bootstrap,
            oob_score=self._oob_score,
            random_state=self._random_state
        )

    #----------------------------------------

    def predict(self, X_test_i: np.ndarray | pd.DataFrame, y_true_i: np.ndarray | pd.Series, **kwargs) -> np.ndarray:
        """
        Override of the base predict method
        """
        self._y_pred = super().predict(X_test_i, y_true_i, **kwargs)
        return self._y_pred

    #----------------------------------------

    def cross_evaluate(
        self,
        X_train_i: np.ndarray | pd.DataFrame,
        y_train_i: np.ndarray | pd.Series,
        X_val_i: np.ndarray | pd.DataFrame,
        y_val_i: np.ndarray | pd.Series,
        **kwargs
    ) -> None:
        """
        Override of the base cross_evaluate method for RandomForest.

        Parameters
        ----------
        X_train_i : np.ndarray | pd.DataFrame
            The input features for training
        y_train_i : np.ndarray | pd.Series
            The target labels for training
        X_val_i : np.ndarray | pd.DataFrame
            The input features for validation
        y_val_i : np.ndarray | pd.Series
            The target labels for validation
        **kwargs :
            Additional keyword arguments
        """
        if not self._param_grid:
            logger.error("No parameter grid specified for cross-evaluation.")
            return

        dataset_shape = getattr(X_train_i, "shape", len(X_train_i))
        val_shape = getattr(X_val_i, "shape", len(X_val_i))
        logger.info(f"Starting RandomForest cross-evaluation on dataset shape {dataset_shape} with validation shape {val_shape}...")
        logger.info(f"Parameter grid configured with keys: {list(self._param_grid.keys())}")

        # Get the hyperparameters
        p_n_estimators = self._param_grid.get("n_estimators", [self._n_estimators])
        p_max_depth = self._param_grid.get("max_depth", [None])
        p_min_samples_split = self._param_grid.get("min_samples_split", [self._min_samples_split])
        p_min_samples_leaf = self._param_grid.get("min_samples_leaf", [1])
        p_max_features = self._param_grid.get("max_features", [self._max_features])
        p_bootstrap = self._param_grid.get("bootstrap", [self._bootstrap])

        results = []
        best_val_acc = -1.0

        trial_idx = 0
        for bootstrap in p_bootstrap:
            for max_features in p_max_features:
                for min_samples_leaf in p_min_samples_leaf:
                    for min_samples_split in p_min_samples_split:
                        for max_depth in p_max_depth:
                            for n_estimators in p_n_estimators:

                                trial_idx += 1
                                logger.info(
                                    f"[Trial {trial_idx}] Configuration: n_estimators={n_estimators}, "
                                    f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
                                    f"min_samples_leaf={min_samples_leaf}, max_features={max_features}, "
                                    f"bootstrap={bootstrap}"
                                )

                                rf = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    bootstrap=bootstrap,
                                    oob_score=self._oob_score if bootstrap else False,
                                    random_state=self._random_state
                                )

                                logger.info(f"[Trial {trial_idx}] Fitting model...")
                                rf.fit(X_train_i, y_train_i)

                                train_pred = rf.predict(X_train_i)
                                train_acc = float(accuracy_score(y_train_i, train_pred))

                                val_pred = rf.predict(X_val_i)
                                val_acc = float(accuracy_score(y_val_i, val_pred))

                                logger.info(
                                    f"[Trial {trial_idx}] Completed - Train Accuracy: {train_acc:.4f}, "
                                    f"Val Accuracy: {val_acc:.4f}"
                                )

                                trial_result = {
                                    "n_estimators": n_estimators,
                                    "max_depth": max_depth,
                                    "min_samples_split": min_samples_split,
                                    "min_samples_leaf": min_samples_leaf,
                                    "max_features": max_features,
                                    "bootstrap": bootstrap,
                                    "accuracy": train_acc,
                                    "val_accuracy": val_acc,
                                }
                                results.append(trial_result)

                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    self._best_score = float(val_acc)
                                    self._best_params = trial_result
                                    self._best_estimator = rf

                                del rf

        logger.info(f"RandomForest cross-evaluation completed. Evaluated {len(results)} configurations.")
        logger.info(f"Best validation score: {self._best_score:.4f}" if self._best_score is not None else "Best validation score: N/A")
        logger.info(f"Best parameters: {self._best_params}")

    