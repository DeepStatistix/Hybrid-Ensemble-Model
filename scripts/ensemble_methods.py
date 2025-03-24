"""
ensemble_methods.py

A SINGLE file containing Bagging, Boosting, Stacking, and Dagging logic.
Adapted from Melding2 (1).ipynb so results match the original notebook.
"""

import numpy as np
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# ===========================
#         BAGGING
# ===========================
def train_bagging_model(
    X_train, 
    y_train, 
    n_estimators=10, 
    max_depth=5, 
    random_state=42
):
    """
    Trains a Bagging ensemble of Decision Trees.

    Modify these defaults to match exactly what's in your new notebook.
    """
    # If your new notebook sets other hyperparameters, do so here as well
    base_estimator = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    bagging_model = BaggingRegressor(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=random_state
        # additional params if used in your notebook
    )
    bagging_model.fit(X_train, y_train)
    return bagging_model

# ===========================
#         BOOSTING
# ===========================
def train_boosting_model(
    X_train, 
    y_train, 
    n_estimators=50, 
    max_depth=3,
    learning_rate=0.1,
    random_state=42
):
    """
    Trains an AdaBoost ensemble of Decision Trees (or XGBoost, if your notebook does so).
    Adjust to match Melding2 (1).ipynb exactly.
    """
    base_estimator = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    boosting_model = AdaBoostRegressor(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    boosting_model.fit(X_train, y_train)
    return boosting_model

# If your new notebook uses xgboost, replace the above with something like:
# def train_boosting_model(...):
#     model = XGBRegressor(...).fit(X_train, y_train)
#     return model

# ===========================
#         STACKING
# ===========================
def train_stacking_model(
    X_train, 
    y_train, 
    random_state=42
):
    """
    Trains a Stacking ensemble, e.g. with base DecisionTree/RandomForest + LinearRegression meta-learner
    exactly as in your new notebook.
    """
    # Base estimators in your new notebook:
    estimators = [
        ('dt', DecisionTreeRegressor(max_depth=5, random_state=random_state)),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=random_state))
        # If your new notebook uses different base learners, copy them here
    ]
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        passthrough=False
    )
    stacking_model.fit(X_train, y_train)
    return stacking_model

# ===========================
#         DAGGING
# ===========================
class DaggingRegressor:
    """
    A custom Dagging approach for disjoint aggregating.
    Copy any logic or variations from your new notebook.
    """
    def __init__(self, k_splits=5, max_depth=3, random_state=42):
        self.k_splits = k_splits
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = []
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        subset_size = n_samples // self.k_splits
        self.models = []
        
        for k in range(self.k_splits):
            start = k * subset_size
            end = (k+1) * subset_size if k < self.k_splits - 1 else n_samples
            subset_idx = indices[start:end]

            X_subset = X[subset_idx]
            y_subset = y[subset_idx]

            model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + k)
            model.fit(X_subset, y_subset)
            self.models.append(model)

    def predict(self, X):
        predictions = [m.predict(X) for m in self.models]
        return np.mean(predictions, axis=0)

def train_dagging_model(
    X_train, 
    y_train, 
    k_splits=5, 
    max_depth=3, 
    random_state=42
):
    """
    Trains the custom DaggingRegressor.
    """
    dagging = DaggingRegressor(k_splits=k_splits, max_depth=max_depth, random_state=random_state)
    dagging.fit(X_train, y_train)
    return dagging
