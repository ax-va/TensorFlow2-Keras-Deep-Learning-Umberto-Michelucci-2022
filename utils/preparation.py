import numpy as np
import pandas as pd


def split_into_complete_and_incomplete_data(df):
    """Split data into data without missing values and with missing values"""
    df_complete = df.dropna()
    df_incomplete = df.loc[list(set(df.index) - set(df_complete.index))]
    return df_complete, df_incomplete


def normalize_data(data):
    """Normalize features in the form of numpy.ndarray or pandas.DataFrame"""
    np_data = data if isinstance(data, np.ndarray) else data.to_numpy()
    mu = np.mean(np_data, axis=0)
    sigma = np.std(np_data, axis=0)
    data_normalized = (np_data - mu) / sigma  # numpy.ndarray
    if not isinstance(data, np.ndarray):
        # Return pandas.DataFrame
        data_normalized = pd.DataFrame(
            data=data_normalized,
            columns=data.columns,
            index=data.index
        )
    return data_normalized, mu, sigma


def split_into_train_and_dev_data(features, target, train_proportion=0.8, seed=42):
    """
    Split data into training and development data.
    Return two datasets for features and two datasets for a target.
    """
    np.random.seed(seed)  # reproducible random
    rnd = np.random.rand(len(target)) < train_proportion
    features_train = features[rnd]
    target_train = target[rnd]
    features_dev = features[~rnd]
    target_dev = target[~rnd]
    return (features_train, target_train), (features_dev, target_dev)
