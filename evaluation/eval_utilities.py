"""
Module: 			eval_utilities.py
Project: 			ML_DL_Exam
Author: 			Calogero Forte
Revision: 		    1.12
Last modify date: 	09/11/2026
"""

#------------------------------
# Imports
#------------------------------

import logging
# pyrefly: ignore [missing-import]
from utils.plot_utilities import plot_heatmap
# pyrefly: ignore [missing-import]
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

    print(classifier_i.classification_report)

#----------------------------------------

def print_accuracy(classifier_i: BaseClassifier) -> None:
    """
    Print the accuracy.
    
    Returns
    -------
    None
    """

    print('Accuracy: %.4f' % classifier_i.accuracy)
   
#----------------------------------------

def plot_confution_matrix(classifier_i: BaseClassifier) -> None:
    """
    Plot the confusion matrix.
    
    Returns
    -------
    None
    """

    plot_heatmap(classifier_i.confusion_matrix, title_i='Confusion Matrix', xlabel_i='Predicted Label', ylabel_i='True Label')

    
    
    