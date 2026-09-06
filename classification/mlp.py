"""
Module: 			mlp.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.4
Last modify date: 	09/06/2026
"""

#------------------------------
# Import
#------------------------------
import tempfile
from typing import Optional, List
from keras import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Model, clone_model
from keras.losses import Loss
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
        self._classifier = Sequential(name='MLP_Network')
        if hidden_layers_i is not None:
            self.add_hidden_layers(hidden_layers_i, hidden_activation_i)
        if output_units_i is not None:
            self.add_output_layer(output_units_i, output_activation_i)

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
        self._classifier.summary()

    #----------------------------------------

    def predict(self, X_test_i: np.ndarray | pd.DataFrame, y_true_i: np.ndarray | pd.Series, **kwargs) -> np.ndarray:
        """
        Override of the base predict method
        """
        y_pred_tmp = super().predict(X_test_i, y_true_i, **kwargs)
        self._y_pred = MLP._probailities_to_target(y_pred_tmp)
        return self._y_pred

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

    def cross_evaluate(self, X_train_i: np.ndarray | pd.DataFrame, y_train_i: np.ndarray | pd.Series, cv_i: int = 5, **kwargs) -> None:
        """
        Override of the base cross_evaluate method
        """
        X = np.asarray(X_train_i)
        y = np.asarray(y_train_i)

        if not self._param_grid:
            logger.error("No parameter grid specified")
            return

        hp = kt.HyperParameters()
        param_mapping = {}

        for param_name, param_values in self._param_grid.items():
            if all(isinstance(v, (int, float, str, bool)) for v in param_values):
                hp.Choice(param_name, values=param_values)
            else:
                str_vals = [str(v) for v in param_values]
                param_mapping[param_name] = dict(zip(str_vals, param_values))
                hp.Choice(param_name, values=str_vals)

        opt_name = getattr(getattr(self._classifier, "optimizer", None), "name", "adam")
        loss_fn = getattr(self._classifier, "loss", "sparse_categorical_crossentropy")

        def build_model(hp_trial):
            if "optimizer" in hp_trial:
                opt = hp_trial.get("optimizer")
            else:
                opt = opt_name if isinstance(opt_name, str) else "adam"

            if "loss" in hp_trial:
                loss = hp_trial.get("loss")
            else:
                loss = loss_fn if isinstance(loss_fn, (str, Loss)) else "sparse_categorical_crossentropy"

            try:
                model = clone_model(self._classifier)
                model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
            except Exception:
                model = self._classifier
                model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
            return model

        with tempfile.TemporaryDirectory() as tmp_dir:
            tuner = KFoldGridSearch(
                cv=cv_i,
                hypermodel=build_model,
                hyperparameters=hp,
                objective=kt.Objective("val_accuracy", direction="max"),
                directory=tmp_dir,
                project_name="mlp_grid_search",
                overwrite=True,
            )

            tuner.search(X, y, verbose=kwargs.get("verbose", 0))

            best_trials = tuner.oracle.get_best_trials(1)
            if best_trials:
                best_trial = best_trials[0]
                self._best_score = float(best_trial.score)
                best_hps = best_trial.hyperparameters.values
                resolved_best_params = {}
                for k, v in best_hps.items():
                    if k in param_mapping and str(v) in param_mapping[k]:
                        resolved_best_params[k] = param_mapping[k][str(v)]
                    else:
                        resolved_best_params[k] = v
                self._best_params = resolved_best_params

                best_hp_obj = tuner.get_best_hyperparameters(1)[0]
                self._best_estimator = tuner.hypermodel.build(best_hp_obj)
            else:
                self._best_score = None
                self._best_params = {}
                self._best_estimator = self._classifier

            final_epochs = self._best_params.get("epochs", kwargs.get("epochs", 10))
            final_batch_size = self._best_params.get("batch_size", kwargs.get("batch_size", 32))
            self._best_estimator.fit(X, y, epochs=final_epochs, batch_size=final_batch_size, verbose=kwargs.get("verbose", 0))

