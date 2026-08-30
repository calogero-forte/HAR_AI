"""
Base class for all classifiers.
"""

#------------------------------
# Import
#------------------------------

import logging
from typing import Optional, Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)

#------------------------------
# Classes
#------------------------------
class BaseClassifier:
    
    def __init__(self) -> None:
        """
        Initialize the Base Classifier

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Classifier instance
        self._classifier: Any = None

        # Parameter grid fro cross validation
        self._param_grid: Dict[str, List] = {}
        
        # Best parameters, score and estimator
        self._best_params: Dict[str, Any] = {}
        self._best_score: float = None
        self._best_estimator: Any = None

        # Store predictions, metrics and confusion matrix
        self._y_pred: np.ndarray = None
        self._accuracy : float = None
        self._classification_report : str = None
        self._confusion_matrix: np.ndarray = None

    #----------------------------------------

    def get_predictions(self) -> Optional[np.ndarray | None]:
        """
        Get the predictions

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray | None
        The predictions if the classifier has been ran, None otherwise
        """
        return self._y_pred

    #----------------------------------------

    def get_accuracy(self) -> Optional[float | None]:
        """
        Get the accuracy

        Parameters
        ----------
        None

        Returns
        -------
        float | None
        The accuracy if the classifier has been ran, None otherwise
        """
        return self._accuracy
    
    #----------------------------------------

    def get_classification_report(self) -> Optional[str | None]:
        """
        Get the classification report

        Parameters
        ----------
        None

        Returns
        -------
        str | None
        The classification report if the classifier has been ran, None otherwise
        """
        return self._classification_report

    #----------------------------------------

    def get_confusion_matrix(self) -> Optional[np.ndarray | None]:
        """
        Get the confusion matrix

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray | None
        The confusion matrix if the classifier has been ran, None otherwise
        """
        return self._confusion_matrix
    
    #----------------------------------------

    def train(self, X_train_i: np.ndarray | pd.DataFrame, y_train_i: np.ndarray) -> None:
        """
        Train the base classifier

        Parameters
        ----------
        X_train_i : np.ndarray | pd.DataFrame
            The input features for training
        y_train_i : np.ndarray
            The target labels for training
        
        Returns
        -------
        None
        """
        logger.info(f"Training classifier on dataset shape: {X_train_i.shape}")
        self._classifier.fit(X=X_train_i, y=y_train_i)
        logger.info("Classifier training completed.")
    
    #----------------------------------------

    def predict(self, X_test_i: np.ndarray | pd.DataFrame, y_test_i: np.ndarray) -> np.ndarray:
        """
        Predict and return target labels for the input features.
        The method use the base classifier of this instance or, the best one if 
        the grid search has been performed.
        Store internally the predictions, the metrics and the confuction matrix
        
        Parameters
        ----------
        X_test_i : np.ndarray | pd.DataFrame
            The input features for prediction
        y_test_i : np.ndarray
            The target labels for prediction
        
        Returns
        -------
        np.ndarray
            The predicted target labels
        """

        if(self._best_estimator == None):
            self._y_pred = self.rf.predict(X=X_test_i)
        else:
            self._y_pred = self._best_estimator.predict(X=X_test_i)
        
        self._classification_report = classification_report(y_true=y_test_i, y_pred=self._y_pred)
        self._accuracy = accuracy_score(y_true=y_test_i, y_pred=self._y_pred)
        self._confusion_matrix = confusion_matrix(y_true=y_test_i, y_pred=self._y_pred)

        logger.info(f"Prediction completed on {len(self._y_pred)} test samples with accuracy: {self._accuracy:.4f}")
        return self._y_pred

    #----------------------------------------

    def set_param_grid(self, param_grid_i: Dict[str, List]) -> None:
        """
        Set the parameter grid

        Parameters
        ----------
        param_grid_i : Dict[str, List]
            The parameter grid

        Returns
        -------
        None
        """
        self._param_grid = param_grid_i
        logger.info(f"Parameter grid configured with keys: {list(param_grid_i.keys())}")

    #----------------------------------------

    def cross_evaluate(self, X_train_i: np.ndarray | pd.DataFrame, y_train_i: np.ndarray, cv_i: int = 5) -> None:
        """
        Cross-evaluate the classifier
        
        Parameters
        ----------
        X_i : np.ndarray | pd.DataFrame
            The input features for cross-evaluation
        y_i : np.ndarray
            The target labels for cross-evaluation
        cv_i : int, default=5
            The number of folds for cross-evaluation
        
        Returns
        -------
        None
        """

        logger.info(f"Starting cross-validation ({cv_i} folds) on dataset shape {X_train_i.shape}...")
        grid = GridSearchCV(
            self._classifier,
            self._param_grid,
            cv=cv_i,
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(X_train_i, y_train_i)

        self._best_params = grid.best_params_
        self._best_score = grid.best_score_
        self._best_estimator = grid.best_estimator_
        
        logger.info(f"Cross-evaluation completed. Best score: {self._best_score:.4f}")
        logger.info(f"Best parameters: {self._best_params}")
        
    
    