"""
hem.py

Implements a HybridEnsembleModel that combines predictions from multiple
sub-models (Bagging, Boosting, Stacking, Dagging) via a meta-learner.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

class HybridEnsembleModel:
    """
    Combines the predictions from multiple trained ensemble models
    (bagging, boosting, stacking, dagging, etc.) using a meta-learner.
    """
    def __init__(self):
        self.meta_learner = LinearRegression()

    def fit(self, predictions_list, y_train):
        """
        predictions_list: list of arrays from each sub-model's predictions on the TRAIN set
        y_train: actual target values for the corresponding training set
        """
        # Concatenate predictions side by side:
        meta_features = np.column_stack(predictions_list)
        self.meta_learner.fit(meta_features, y_train)

    def predict(self, predictions_list):
        """
        predictions_list: list of arrays from each sub-model's predictions on the TEST set
        """
        meta_features = np.column_stack(predictions_list)
        return self.meta_learner.predict(meta_features)
