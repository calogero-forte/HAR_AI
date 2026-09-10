"""
Module: 			mlp.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.10
Last modify date: 	09/10/2026
"""

import logging
import json
from typing import Optional, List, Dict
from tensorflow.config.experimental import enable_op_determinism
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization
from keras.regularizers import L1, L2
from keras.optimizers import Adam
from keras.utils import set_random_seed
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import keras_tuner as kt

# pyrefly: ignore [missing-import]
from classification.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

#------------------------------
# KFold GridSearch Tuner
#------------------------------
class KFoldGridSearch(kt.GridSearch):
    """
    Subclass of keras_tuner.GridSearch performing K-Fold cross-validation for each hyperparameter trial.
    """
    def __init__(self, cv: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.cv = cv

    def run_trial(self, trial, x, y, **kwargs):
        hp = trial.hyperparameters
        epochs = hp.get("epochs") if "epochs" in hp else kwargs.get("epochs", 10)
        batch_size = hp.get("batch_size") if "batch_size" in hp else kwargs.get("batch_size", 32)

        X_arr = np.asarray(x)
        y_arr = np.asarray(y)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        val_accs = []

        for train_idx, val_idx in kf.split(X_arr, y_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            model = self.hypermodel.build(hp)
            model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

            eval_res = model.evaluate(X_val, y_val, verbose=0)
            acc = eval_res[1] if isinstance(eval_res, (list, tuple)) else eval_res
            val_accs.append(acc)

        mean_score = float(np.mean(val_accs))
        self.oracle.update_trial(trial.trial_id, {"val_accuracy": mean_score})


#------------------------------
# Class
#------------------------------
class MLP(BaseClassifier):

    def __init__(self, input_shape_i: Optional[tuple] = None, hidden_layers_i: Optional[List[int]] = None, output_units_i: Optional[int] = None,
        hidden_activation_i: Optional[List[str]] = None, output_activation_i: Optional[str] = None) -> None:
        
        super().__init__()
        logger.info("Initializing MLP classifier...")
        self._classifier = Sequential(name='MLP_Network')
        if hidden_layers_i is not None:
            self.add_hidden_layers(hidden_layers_i, hidden_activation_i)
        if output_units_i is not None:
            self.add_output_layer(output_units_i, output_activation_i)

        # Store the history of the best trial
        self._best_train_history: Optional[Dict[str, List[float]]] = None

        # Experiment reproducibility
        set_random_seed(42)
        enable_op_determinism()


    #----------------------------------------
    
    def add_hidden_layers(self, hidden_layers_i: List[int], hidden_activation_i: Optional[List[str]] = None) -> None:
        """
        Add one or multiple hidden layers with the specified number of units and activation function

        Parameters
        ----------
        hidden_layers_i : List[int]
            number of units for each hidden layer
        hidden_activation_i : List[str]
            activation function for each hidden layer.
            If None, ReLU is used for all hidden layers.

        Return
        ------
        None
        """
        if hidden_activation_i is None:
            hidden_activation_i = ['relu'] * len(hidden_layers_i)
        
        logger.info(f"Adding {len(hidden_layers_i)} hidden layer(s) with units {hidden_layers_i} and activations {hidden_activation_i}")
        for i in range(len(hidden_layers_i)):
            self._classifier.add(
                Dense(
                    units=hidden_layers_i[i],
                    activation=hidden_activation_i[i],
                    name=f"hidden_layer_{i}"
                )
            )    
    
    #----------------------------------------

    def add_output_layer(self, output_units_i: int, output_activation_i: Optional[str] = None) -> None:
        """
        Add the output layer with the specified number of units and activation function

        Parameters
        ----------
        output_units_i : int
            number of units for the output layer
        output_activation_i : str
            activation function for the output layer.
            If None, softmax is used for all output layers.

        Return
        ------
        None
        """
        if output_activation_i is None:
            output_activation_i = 'softmax'
        
        logger.info(f"Adding output layer with {output_units_i} units and activation '{output_activation_i}'")
        self._classifier.add(
            Dense(
                units=output_units_i,
                activation=output_activation_i,
                name="output_layer"
            )
        ) 

    #----------------------------------------

    def compile(self, optimizer_i='adam', loss_i='sparse_categorical_crossentropy', metrics_i=['accuracy']) -> None:
        """
        Compile the model with the specified optimizer, loss function and metrics

        Parameters
        ----------
        optimizer_i : str
            The optimizer to use
        loss_i : str
            The loss function to use
        metrics_i : List[str]
            The metrics to use
        
        Return
        ------
        None
        """
        logger.info(f"Compiling MLP model (optimizer='{optimizer_i}', loss='{loss_i}', metrics={metrics_i})")
        self._classifier.compile(
            optimizer=optimizer_i,
            loss=loss_i,
            metrics=metrics_i
        )

    #----------------------------------------

    def print_summary(self) -> None:
        """
        Print the summary of the model

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        logger.info("Displaying MLP model summary:")
        self._classifier.summary()

    #----------------------------------------

    def predict(self, X_test_i: np.ndarray | pd.DataFrame, y_true_i: np.ndarray | pd.Series, **kwargs) -> np.ndarray:
        """
        Override of the base predict method
        """
        dataset_shape = getattr(X_test_i, "shape", len(X_test_i))
        logger.info(f"Running MLP prediction on test dataset shape: {dataset_shape}")
        y_pred_tmp = super().predict(X_test_i, y_true_i, **kwargs)
        self._y_pred = MLP._probailities_to_target(y_pred_tmp)
        logger.info(f"Prediction completed. Generated predictions for {len(self._y_pred)} samples.")
        return self._y_pred

    #----------------------------------------

    def best_history_to_json(self, json_path_i: str) -> None:
        """
        Save the history of the best trial to a JSON file

        Parameters
        ----------
        json_path_i : str
            The path to the JSON file

        Return
        ------
        None
        """
        logger.info(f"Saving the history of the best trial to {json_path_i}...")
        if self._best_train_history is None:
            logger.error("No best trial history found.")
            return
        with open(json_path_i, "w") as f:
            json.dump(self._best_train_history, f)
        logger.info("Best trial history saved successfully.")

    #----------------------------------------

    @staticmethod
    def best_history_from_json(json_path_i: str) -> Dict[str, List[float]]:
        """
        Load the history of the best trial from a JSON file

        Parameters
        ----------
        json_path_i : str
            The path to the JSON file

        Return
        ------
        Dict[str, List[float]]
            The history of the best trial
        """
        logger.info(f"Loading the history of the best trial from {json_path_i}...")
        with open(json_path_i, "r") as f:
            best_train_history_i = json.load(f)
        logger.info("Best trial history loaded successfully.")
        return best_train_history_i

    #----------------------------------------

    @staticmethod
    def _probailities_to_target(probabilities_i: np.ndarray) -> np.ndarray:
        """
        Convert probabilities to target labels

        Parameters
        ----------
        probailities_i : np.ndarray
            The probabilities to convert

        Return
        ------
        np.ndarray
            The target labels
        """
        if probabilities_i.ndim > 1 and probabilities_i.shape[1] > 1:
            return np.argmax(probabilities_i, axis=1)
        else:
            return (probabilities_i.ravel() > 0.5).astype(int)

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
        Override of the base cross_evaluate method for MLP.

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
        logger.info(f"Starting MLP cross-evaluation on dataset shape {dataset_shape} with validation shape {val_shape}...")
        logger.info(f"Parameter grid configured with keys: {list(self._param_grid.keys())}")

        # Get the hyperparameters
        p_layers = self._param_grid.get("hidden_layers", [1])
        p_units = self._param_grid.get("units", [8])
        if "Units_per_layer" in self._param_grid:
            p_units = self._param_grid.get("Units_per_layer")
        p_learning_rate = self._param_grid.get("learning_rate", [0.001])
        p_batch_size = self._param_grid.get("batch_size", [32])
        p_epochs = self._param_grid.get("epochs", [10])
        p_regularizer = self._param_grid.get("regularizer", [None])

        results = []
        best_val_acc = -1.0

        input_dim = X_train_i.shape[1] if hasattr(X_train_i, "shape") else len(X_train_i[0])
        output_dim = len(np.unique(y_train_i))

        trial_idx = 0
        for regularizer in p_regularizer:
            for learning_rate in p_learning_rate:    
                for epochs in p_epochs:
                    for batch_size in p_batch_size:
                        for layers in p_layers:
                            for units in p_units:

                                trial_idx += 1
                                regularizer_name = type(regularizer).__name__
                                regularizer_rate = regularizer.l1 if isinstance(regularizer, L1) else \
                                    regularizer.l2 if isinstance(regularizer, L2) else None
                                logger.info(
                                    f"[Trial {trial_idx}] Configuration: hidden_layers={layers}, units={units}, "
                                    f"learning_rate={learning_rate}, regularizer=({regularizer_name}, {regularizer_rate}), "
                                    f"batch_size={batch_size}, epochs={epochs}"
                                )

                                mlp = Sequential(name=f"MLP_Trial_{trial_idx}")
                                mlp.add(Input(shape=(input_dim,)))
                                for _ in range(layers):
                                    mlp.add(Dense(
                                        units=units,
                                        activation="relu",
                                        kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer
                                    ))
                                mlp.add(Dense(units=output_dim, activation="softmax"))

                                mlp.compile(
                                    optimizer=Adam(learning_rate=learning_rate),
                                    loss="sparse_categorical_crossentropy",
                                    metrics=["accuracy"]
                                )

                                logger.info(f"[Trial {trial_idx}] Fitting model (batch_size={batch_size}, epochs={epochs})...")
                                res = mlp.fit(X_train_i, y_train_i, batch_size=batch_size, epochs=epochs, validation_data=(X_val_i, y_val_i))

                                val_acc = res.history['val_accuracy'][-1]
                                train_acc = res.history['accuracy'][-1]
                                val_loss = res.history['val_loss'][-1]
                                train_loss = res.history['loss'][-1]

                                logger.info(
                                    f"[Trial {trial_idx}] Completed - Train Accuracy: {train_acc:.4f}, "
                                    f"Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}"
                                )

                                trial_result = {
                                    "hidden_layers": layers,
                                    "units": units,
                                    "batch_size": batch_size,
                                    "epochs": epochs,
                                    "learning_rate": learning_rate,
                                    "regularizer": f"({regularizer_name}, {regularizer_rate}) ", 
                                    "accuracy": train_acc,
                                    "val_accuracy": val_acc,
                                    "loss": train_loss,
                                    "val_loss": val_loss,
                                }
                                results.append(trial_result)

                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    self._best_score = float(val_acc)
                                    self._best_params = trial_result
                                    self._best_estimator = mlp
                                    self._best_train_history = res.history

                                # Deleting the model
                                del mlp
                                    
        logger.info(f"MLP cross-evaluation completed. Evaluated {len(results)} configurations.")
        logger.info(f"Best validation score: {self._best_score:.4f}" if self._best_score is not None else "Best validation score: N/A")
        logger.info(f"Best parameters: {self._best_params}")

                
                        

        

