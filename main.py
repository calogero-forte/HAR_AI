
#------------------------------
# Imports
#------------------------------

from dataset_preparation.dataset_handler import DatasetHandler

from classification.random_forest import RandomForest

from sklearn.preprocessing import StandardScaler

from evaluation.eval_utilities import print_classification_report, print_accuracy, plot_confution_matrix


#------------------------------
# Global Variables
#------------------------------
SEED = 42
TRAIN_ATASET_PATH = "./Dataset/train_dataset.csv"
TEST_DATASET_PATH = "./Dataset/test_dataset.csv"

#------------------------------
# Main Code
#------------------------------

# Load Dataset
handler = DatasetHandler(TRAIN_ATASET_PATH, TEST_DATASET_PATH)

# Standardization
scaler = StandardScaler()
X_train_std = scaler.fit_transform(handler.get_train_set()[0])
X_test_std = scaler.transform(handler.get_test_set()[0])

# Classification
rf = RandomForest()
rf.train(X_train_std, handler.get_train_set()[1])
rf.predict(X_test_std, handler.get_test_set()[1])

# Evaluation
print_classification_report(rf)
print_accuracy(rf)
plot_confution_matrix(rf)