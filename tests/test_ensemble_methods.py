import numpy as np
from scripts.ensemble_methods import (
    train_bagging_model,
    train_boosting_model,
    train_stacking_model,
    train_dagging_model
)

def test_bagging_model():
    X_train = np.random.rand(20, 3)
    y_train = np.random.rand(20)
    model = train_bagging_model(X_train, y_train, n_estimators=5, max_depth=2)
    preds = model.predict(X_train)
    assert len(preds) == 20, "Bagging predictions should match training size."

def test_boosting_model():
    X_train = np.random.rand(20, 3)
    y_train = np.random.rand(20)
    model = train_boosting_model(X_train, y_train, n_estimators=5, max_depth=2, learning_rate=0.1)
    preds = model.predict(X_train)
    assert len(preds) == 20, "Boosting predictions should match training size."

def test_stacking_model():
    X_train = np.random.rand(20, 3)
    y_train = np.random.rand(20)
    model = train_stacking_model(X_train, y_train)
    preds = model.predict(X_train)
    assert len(preds) == 20, "Stacking predictions should match training size."

def test_dagging_model():
    X_train = np.random.rand(20, 3)
    y_train = np.random.rand(20)
    model = train_dagging_model(X_train, y_train, k_splits=2, max_depth=2)
    preds = model.predict(X_train)
    assert len(preds) == 20, "Dagging predictions should match training size."
