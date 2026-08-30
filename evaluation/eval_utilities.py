"""
This module contains functions to evaluate the performance of a classification model.
"""

#------------------------------
# Imports
#------------------------------

import logging
from utils.plot_utilities import plot_heatmap
from classification.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

#------------------------------
# Functions
#------------------------------

def print_classification_report(classifier_i: BaseClassifier) -> None:

    """
    Print the classification report.
    
    Returns
    -------
    None
    """
    if classifier_i._classification_report is not None:
        print(classifier_i._classification_report)
    else:
        logger.warning("No classification report available to display.")
        print("No classification report available.")

#----------------------------------------

def print_accuracy(classifier_i: BaseClassifier) -> None:
    """
    Print the accuracy.
    
    Returns
    -------
    None
    """
    if classifier_i._accuracy is not None:
        print('Accuracy: %.4f' % classifier_i._accuracy)
    else:
        logger.warning("No accuracy score available to display.")
        print("No accuracy available.")

#----------------------------------------

def plot_confution_matrix(classifier_i: BaseClassifier) -> None:
    """
    Plot the confusion matrix.
    
    Returns
    -------
    None
    """
    if classifier_i._confusion_matrix is not None:
        plot_heatmap(classifier_i._confusion_matrix, title_i='Confusion Matrix', xlabel_i='Predicted Label', ylabel_i='True Label')
    else:
        logger.warning("No confusion matrix available to plot.")
        
    
    
    