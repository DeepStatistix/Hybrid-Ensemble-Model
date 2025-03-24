"""
data_preprocessing.py

Handles data loading, cleaning, feature creation, and train/test splitting.
Replicate the logic from Melding2 (1).ipynb so results match exactly.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    """
    Loads a CSV file from the provided path.

    :param path: Path to the CSV file.
    :return: DataFrame containing loaded & cleaned data.
    """
    df = pd.read_csv(path)
    # If your notebook drops NaN, or does other cleaning, replicate those steps:
    df.dropna(inplace=True)
    return df


def create_lag_features(df: pd.DataFrame, target_col: str, num_lags=5) -> pd.DataFrame:
    """
    Creates lag features for time-series forecasting.

    :param df: Original DataFrame
    :param target_col: Name of the column to create lags from (e.g. 'Close')
    :param num_lags: Number of lags to create
    :return: DataFrame with added lag columns
    """
    for lag in range(1, num_lags + 1):
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    # Remove rows with NaN introduced by shifting:
    df.dropna(inplace=True)
    return df


def split_data(df: pd.DataFrame, features: list, target: str, test_size=0.2, shuffle=False):
    """
    Splits the dataset into train/test sets.

    :param df: DataFrame
    :param features: List of feature column names
    :param target: Target column name
    :param test_size: Fraction of data for testing
    :param shuffle: Whether to shuffle data before splitting (False if time-series)
    :return: (X_train, X_test, y_train, y_test)
    """
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test
