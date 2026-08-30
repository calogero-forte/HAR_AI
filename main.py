
import logging
# pyrefly: ignore [missing-import]
from dataset_preparation.dataset_handler import DatasetHandler
# pyrefly: ignore [missing-import]
from classification.random_forest import RandomForest
# pyrefly: ignore [missing-import]
from evaluation.eval_utilities import print_classification_report, print_accuracy, plot_confution_matrix
# pyrefly: ignore [missing-import]
from dataset_preparation.dimensionality_reduction import reduce_by_lda
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure logging without timestamp
logging.basicConfig(level=logging.INFO, format="%(name)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

#------------------------------
# Global Variables
#------------------------------
SEED = 42
TRAIN_ATASET_PATH = "./Dataset/train_dataset.csv"
TEST_DATASET_PATH = "./Dataset/test_dataset.csv"

#------------------------------------------------------------------------------------------

#------------------------------
# Load Dataset
#------------------------------

logger.info("--- Starting UCI HAR Dataset Pipeline ---")
handler = DatasetHandler(TRAIN_ATASET_PATH, TEST_DATASET_PATH)

# Linear Discriminant Analysis
logger.info("Performing Linear Discriminant Analysis (LDA) for dimensionality reduction...")
X_train_df, X_test_df = reduce_by_lda(X_train_i=handler.get_train_set()[0], X_test_i=handler.get_test_set()[0], y_train_i=handler.get_train_set()[1])

# Store the new dataframes
handler.update_train_dataset(X_train_df)
handler.update_test_dataset(X_test_df)

# Scaler
logger.info("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_std = scaler.fit_transform( handler.get_train_dataset_list()[-1] )
X_test_std = scaler.transform( handler.get_test_dataset_list()[-1] )

#------------------------------
# Find best parameters
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
rf.cross_evaluate( X_train_i=X_train_std, y_train_i=handler.get_train_set()[1] )

#------------------------------
# Classification & Evaluation
#------------------------------

# Predict
logger.info("Running prediction on test set...")
rf.predict(X_test_std, handler.get_test_set()[1])

# Evaluation
logger.info("Evaluating Random Forest performance...")
print_classification_report(rf)
print_accuracy(rf)
plot_confution_matrix(rf)
logger.info("--- Pipeline Execution Complete ---")