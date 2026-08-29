"""
This module contains functions to evaluate the performance of a classification model.
"""

#------------------------------
# Imports
#------------------------------


from utils.plot_utilities import plot_heatmap
from classification.base_classifier import BaseClassifier


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
    if(classifier_i._classification_report != None):
        print(classifier_i._classification_report)
    else:
        print("No classification report available.")

#----------------------------------------

def print_accuracy(classifier_i: BaseClassifier) -> None:
    """
    Print the accuracy.
    
    Returns
    -------
    None
    """
    if(classifier_i._accuracy != None):
        print('Accuracy: ', classifier_i._accuracy)
    else:
        print("No accuracy available.")

#----------------------------------------

def plot_confution_matrix(classifier_i: BaseClassifier) -> None:
    """
    Plot the confusion matrix.
    
    Returns
    -------
    None
    """
    plot_heatmap(classifier_i._confusion_matrix, title_i='Confusion Matrix', xlabel_i='Predicted Label', ylabel_i='True Label')
        
    
    
    