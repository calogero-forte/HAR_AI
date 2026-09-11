"""
Module: 			main.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.4
Last modify date: 	09/06/2026
"""

import logging
from keras.regularizers import L1, L2
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-import]
import global_variables
# pyrefly: ignore [missing-import]
from dataset_preparation.dataset_handler import DatasetHandler
# pyrefly: ignore [missing-import]
from classification.random_forest import RandomForest
# pyrefly: ignore [missing-import]
from classification.mlp import MLP
# pyrefly: ignore [missing-import]
from evaluation.eval_utilities import print_classification_report, print_accuracy, plot_confution_matrix
# pyrefly: ignore [missing-import]
from dataset_preparation.dimensionality_reduction import reduce_by_lda

# Configure logging without timestamp
logging.basicConfig(level=logging.INFO, format="%(name)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

#------------------------------
# Global Variables
#------------------------------
SEED = global_variables.SEED
TRAIN_ATASET_PATH = "./Dataset/train_dataset.csv"
TEST_DATASET_PATH = "./Dataset/test_dataset.csv"
BEST_TRAIN_HISTORY_PATH = "../best_train_history.json"

#------------------------------------------------------------------------------------------

#------------------------------
# Load Dataset
#------------------------------

logger.info("--- Starting Execution ---")
# Load the dataset and split the training set by subjects
handler = DatasetHandler(TRAIN_ATASET_PATH, TEST_DATASET_PATH, val_split_i=0.2) 

# Linear Discriminant Analysis
logger.info("Performing Linear Discriminant Analysis (LDA) for dimensionality reduction...")
X_train_df, X_val_df, X_test_df = reduce_by_lda(X_train_i=handler.get_train_set()[0], y_train_i=handler.get_train_set()[1],
    X_val_i= handler.get_val_set()[0], X_test_i=handler.get_test_set()[0])

# Store the new dataframes
handler.update_train_dataset(X_train_df)
handler.update_val_dataset(X_val_df)
handler.update_test_dataset(X_test_df)

# Scaler
logger.info("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_std = scaler.fit_transform( handler.get_train_dataset_by_index(-1) )
X_val_std = scaler.transform( handler.get_val_dataset_by_index(-1) )
X_test_std = scaler.transform( handler.get_test_dataset_by_index(-1) )

# Store the new dataframes
handler.update_train_dataset(X_train_std)
handler.update_val_dataset(X_val_std)
handler.update_test_dataset(X_test_std)

#------------------------------
# Machine Learning model
#------------------------------

# Classifier instance
rf = RandomForest()
# Parameters grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 6, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True],
}
rf.set_param_grid(param_grid)
# Find the best parameters
logger.info("Starting hyperparameter tuning via cross-evaluation...")
rf.cross_evaluate( X_train_i=X_train_std, y_train_i=handler.get_train_set()[1], X_val_i=X_val_std, y_val_i=handler.get_val_set()[1] )
# Predict
logger.info("Running prediction on test set...")
rf.predict(X_test_i=X_test_std, y_true_i=handler.get_test_set()[1])
# Evaluation
logger.info("Evaluating Random Forest performance...")
print_classification_report(rf)
print_accuracy(rf)
plot_confution_matrix(rf)
logger.info("--- Machine Learning Pipeline Execution Complete ---")

#------------------------------
# Deep Learning model
#------------------------------

mlp = MLP()
# Parameters grid
mlp_param_grid = {
    "hidden_layers": [1, 2, 3],
    "units": [6, 7, 8, 16, 32],
    "epochs": [10, 15, 20],
    "learning_rate": [0.01, 0.001],
    "batch_size": [16, 32, 64],
    "regularizer": [None, L1(0.01), L1(0.1), L2(0.01), L2(0.1)]
}
mlp.set_param_grid(mlp_param_grid)
# Find the best parameters
logger.info("Starting hyperparameter tuning via cross-evaluation...")
mlp.cross_evaluate(X_train_i=X_train_std, y_train_i=handler.get_train_set()[1], X_val_i=X_val_std, y_val_i=handler.get_val_set()[1])
# Predict
logger.info("Running prediction on test set...")
mlp.predict(X_test_i=X_test_std, y_true_i=handler.get_test_set()[1])
# Evaluation
logger.info("Evaluating MLP performance...")
print_classification_report(mlp)
print_accuracy(mlp)
plot_confution_matrix(mlp)
mlp.best_history_to_json(BEST_TRAIN_HISTORY_PATH)
logger.info("--- Deep Learning Pipeline Execution Complete ---")
logger.info("--- End Of Execution ---")

