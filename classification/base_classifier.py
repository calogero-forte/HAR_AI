"""
Module: 			base_classifier.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.4
Last modify date: 	09/06/2026

Base class for all classifiers.
"""

#------------------------------
# Import
#------------------------------

from abc import abstractmethod, ABC
import logging
from typing import Optional, Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)


#------------------------------
# Classes
#------------------------------
class BaseClassifier(ABC):
    
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

        # Parameter grid for cross validation
        self._param_grid: Dict[str, List] = {}
        
        # Best parameters, score and estimator
        self._best_params: Dict[str, Any] = {}
        self._best_score: Optional[float] = None
        self._best_estimator: Any = None

        # Store true labels when calling predict
        self._y_true: Optional[np.ndarray] = None

        # Store predictions, metrics and confusion matrix
        self._y_pred: Optional[np.ndarray] = None
        self._accuracy: Optional[float] = None
        self._classification_report: Optional[str] = None
        self._confusion_matrix: Optional[np.ndarray] = None

    #----------------------------------------

    def get_predictions(self) -> Optional[np.ndarray]:
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

    @property
    def accuracy(self) -> Optional[float]:
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
        if(self._y_pred is None):
            logger.error("The classifier has not been ran yet.")
            return
        self._accuracy = accuracy_score(self._y_true, self._y_pred)
        return self._accuracy

    #----------------------------------------

    @property
    def confusion_matrix(self) -> Optional[np.ndarray]:
        """
        Get the confusion matrix

        Returns
        -------
        Optional[np.ndarray]
            The confusion matrix if the classifier has been ran, None otherwise
        """
        if(self._y_pred is None):
            logger.error("The classifier has not been ran yet.")
            return
        self._confusion_matrix = confusion_matrix(self._y_true, self._y_pred)
        return self._confusion_matrix

    #----------------------------------------

    @property
    def classification_report(self) -> Optional[str]:
        """
        Get the classification report

        Returns
        -------
        Optional[str]
            The classification report if the classifier has been ran, None otherwise
        """
        if(self._y_pred is None):
            logger.error("The classifier has not been ran yet.")
            return
        self._classification_report = classification_report(self._y_true, self._y_pred)
        return self._classification_report

    #----------------------------------------

    def train(self, X_train_i: np.ndarray | pd.DataFrame, y_train_i: np.ndarray | pd.Series, **kwargs) -> None:
        """
        Train the base classifier. Works for both scikit-learn estimators and Keras models.

        Parameters
        ----------
        X_train_i : np.ndarray | pd.DataFrame
            The input features for training
        y_train_i : np.ndarray | pd.Series
            The target labels for training
        **kwargs :
            Additional arguments passed to fit (e.g. epochs, batch_size for Keras)
        """
        dataset_shape = getattr(X_train_i, "shape", len(X_train_i))
        logger.info(f"Training classifier on dataset shape: {dataset_shape}")
        
        if isinstance(self._classifier, Model):
            # Ensure model is compiled if it wasn't already
            if not getattr(self._classifier, "compiled", True):
                logger.error("Keras model not compiled. Compiling with default parameters.")
                return
            else:
                self._classifier.fit(X_train_i, y_train_i, **kwargs)
        else:
            self._classifier.fit(X_train_i, y_train_i, **kwargs)

        logger.info("Classifier training completed.")
    
    #----------------------------------------

    def predict(self, X_test_i: np.ndarray, y_true_i: np.ndarray | pd.Series, **kwargs: Any) -> np.ndarray:
        """
        Predict and return target labels for input features.
        Uses the base classifier or the best estimator if cross-evaluation has been performed.
        Works for both scikit-learn estimators and Keras models.
        Stores predictions, classification report, accuracy, and confusion matrix internally.

        Parameters
        ----------
        X_test_i : np.ndarray | pd.DataFrame
            The input features for prediction
        y_true_i : np.ndarray | pd.Series
            The target labels for prediction
        **kwargs :
            Additional arguments passed to predict (e.g. batch_size for Keras)

        Returns
        -------
        np.ndarray
            The predicted target labels
        """
        self._y_true = y_true_i
        estimator = self._best_estimator if self._best_estimator is not None else self._classifier

        return estimator.predict(X_test_i, **kwargs)

    #----------------------------------------

    def set_param_grid(self, param_grid_i: Dict[str, List]) -> None:
        """
        Set the parameter grid
        """
        self._param_grid = param_grid_i
        logger.info(f"Parameter grid configured with keys: {list(param_grid_i.keys())}")

    #----------------------------------------

    @abstractmethod
    def cross_evaluate(self, X_train_i: np.ndarray | pd.DataFrame, y_train_i: np.ndarray | pd.Series, cv_i: int = 5, **kwargs) -> None:
        """
        Cross-evaluate the classifier using scikit-learn GridSearchCV or K-Fold for Keras models.

        Parameters
        ----------
        X_train_i : np.ndarray | pd.DataFrame
            The input features for cross-evaluation
        y_train_i : np.ndarray | pd.Series
            The target labels for cross-evaluation
        cv_i : int, default=5
            The number of folds for cross-evaluation

        Returns
        -------
        None
        """
        pass

        