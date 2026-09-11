"""
Module: 			base_classifier.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.6
Last modify date: 	09/08/2026

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
from keras.models import Model

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

    def train(
        self,
        X_train_i: np.ndarray | pd.DataFrame,
        y_train_i: np.ndarray | pd.Series,
        X_val_i: Optional[np.ndarray | pd.DataFrame] = None,
        y_val_i: Optional[np.ndarray | pd.Series] = None,
        **kwargs: Any
    ) -> None:
        """
        Train the base classifier. Works for both scikit-learn estimators and Keras models.

        Parameters
        ----------
        X_train_i : np.ndarray | pd.DataFrame
            The input features for training
        y_train_i : np.ndarray | pd.Series
            The target labels for training
        X_val_i : np.ndarray | pd.DataFrame, optional
            The input features for validation (used if model is a Keras model), default=None
        y_val_i : np.ndarray | pd.Series, optional
            The target labels for validation (used if model is a Keras model), default=None
        **kwargs :
            Additional arguments passed to fit (e.g. epochs, batch_size for Keras)
        """
        dataset_shape = getattr(X_train_i, "shape", len(X_train_i))
        logger.info(f"Training classifier on dataset shape: {dataset_shape}")
        
        fit_kwargs = dict(kwargs)

        if isinstance(self._classifier, Model):
            if X_val_i is not None and y_val_i is not None:
                val_shape = getattr(X_val_i, "shape", len(X_val_i))
                logger.info(f"Training Keras model with validation set shape: {val_shape}")
                fit_kwargs["validation_data"] = (X_val_i, y_val_i)
            else:
                logger.info("Training Keras model without validation set.")

            self._classifier.fit(X_train_i, y_train_i, **fit_kwargs)
        else:
            if X_val_i is not None or y_val_i is not None:
                logger.info("Validation set provided but model is not a Keras model; ignoring validation set.")
            self._classifier.fit(X_train_i, y_train_i, **fit_kwargs)

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
    def cross_evaluate(
        self,
        X_train_i: np.ndarray | pd.DataFrame,
        y_train_i: np.ndarray | pd.Series,
        X_val_i: Optional[np.ndarray | pd.DataFrame] = None,
        y_val_i: Optional[np.ndarray | pd.Series] = None,
        cv_i: int = 5,
        **kwargs
    ) -> None:
        """
        Cross-evaluate the classifier using scikit-learn GridSearchCV or K-Fold for Keras models.

        Parameters
        ----------
        X_train_i : np.ndarray | pd.DataFrame
            The input features for cross-evaluation
        y_train_i : np.ndarray | pd.Series
            The target labels for cross-evaluation
        X_val_i : np.ndarray | pd.DataFrame, optional
            The input features for validation (if applicable), default=None
        y_val_i : np.ndarray | pd.Series, optional
            The target labels for validation (if applicable), default=None
        cv_i : int, default=5
            The number of folds for cross-evaluation

        Returns
        -------
        None
        """
        pass

    #----------------------------------------

    @abstractmethod
    def save_best_estimator(self, path_i: str) -> None:
        """
        Save the best estimator to a file

        Parameters
        ----------
        path_i : str
            The path to the file

        Return
        ------
        None
        """
        pass


        