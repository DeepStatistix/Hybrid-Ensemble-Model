"""
main.py

Example pipeline using the unified ensemble methods in ensemble_methods.py
"""

import os
import numpy as np
from scripts.data_preprocessing import load_data, create_lag_features, split_data
from scripts.ensemble_methods import (
    train_bagging_model,
    train_boosting_model,
    train_stacking_model,
    train_dagging_model
)
from scripts.hem import HybridEnsembleModel
from scripts.evaluation import mse, r2_score, smape, directional_accuracy

def run_pipeline():
    # 1) Load data
    df = load_data(os.path.join("data", "raw", "meta_stock_data.csv"))
    
    # 2) Create lag features if you do that in your new notebook
    df = create_lag_features(df, target_col='Close', num_lags=5)
    
    features = [col for col in df.columns if 'lag' in col]
    target = 'Close'
    
    X_train, X_test, y_train, y_test = split_data(df, features, target, test_size=0.2, shuffle=False)
    
    # 3) Train sub-models
    bagging_model = train_bagging_model(X_train, y_train)
    boosting_model = train_boosting_model(X_train, y_train)
    stacking_model = train_stacking_model(X_train, y_train)
    dagging_model = train_dagging_model(X_train, y_train)
    
    # 4) Train HEM meta-learner
    hem = HybridEnsembleModel()
    bagging_train_preds = bagging_model.predict(X_train)
    boosting_train_preds = boosting_model.predict(X_train)
    stacking_train_preds = stacking_model.predict(X_train)
    dagging_train_preds = dagging_model.predict(X_train)
    
    hem.fit(
        predictions_list=[
            bagging_train_preds, 
            boosting_train_preds, 
            stacking_train_preds, 
            dagging_train_preds
        ],
        y_train=y_train
    )
    
    # 5) Evaluate on test set
    bagging_test_preds = bagging_model.predict(X_test)
    boosting_test_preds = boosting_model.predict(X_test)
    stacking_test_preds = stacking_model.predict(X_test)
    dagging_test_preds = dagging_model.predict(X_test)
    
    hem_preds = hem.predict([
        bagging_test_preds, 
        boosting_test_preds, 
        stacking_test_preds, 
        dagging_test_preds
    ])
    
    print("=== Final HEM Performance on Test Set ===")
    print(f"MSE:   {mse(y_test, hem_preds):.4f}")
    print(f"R^2:   {r2_score(y_test, hem_preds):.4f}")
    print(f"sMAPE: {smape(y_test, hem_preds):.4f}")
    print(f"DA:    {directional_accuracy(y_test, hem_preds):.2f}%")

if __name__ == "__main__":
    run_pipeline()
