#------------------------------
# Import
#------------------------------

import logging
from sklearn.ensemble import RandomForestClassifier
# pyrefly: ignore [missing-import]
from .base_classifier import BaseClassifier
from typing import Tuple
import numpy as np 
import pandas as pd

logger = logging.getLogger(__name__)

#------------------------------
# Classes
#------------------------------
class RandomForest(BaseClassifier):

    def __init__(self, n_estimators_i: int = 400, max_features_i: str = 'sqrt', 
        min_samples_split_i: int = 10, bootstrap_i: bool = True, oob_score_i: bool = True):
        """
        Constructor of the Random Forest class.
        
        Parameters
        ----------
        n_estimators_i : int, default=400
            The number of trees in the forest.
        max_features_i : str, default='sqrt'
            The number of features to consider when looking for the best split.
        min_samples_split_i : int, default=10
            The minimum number of samples required to split an internal node.
        bootstrap_i : bool, default=True
            Whether bootstrap samples are used when building trees.
        oob_score_i : bool, default=True
            Whether to use out-of-bag samples to estimate the generalization error.
        """
        super().__init__()

        # Classifier parameters
        self._n_estimators = n_estimators_i
        self._max_features = max_features_i
        self._min_samples_split = min_samples_split_i
        self._bootstrap = bootstrap_i
        self._oob_score = oob_score_i

        logger.info(
            f"Initializing RandomForest classifier (n_estimators={self._n_estimators}, "
            f"max_features={self._max_features}, min_samples_split={self._min_samples_split}, "
            f"bootstrap={self._bootstrap}, oob_score={self._oob_score})"
        )

        # Classifier instance
        self._classifier = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_features=self._max_features,
            min_samples_split=self._min_samples_split,
            bootstrap=self._bootstrap,
            oob_score=self._oob_score
        )
        
        