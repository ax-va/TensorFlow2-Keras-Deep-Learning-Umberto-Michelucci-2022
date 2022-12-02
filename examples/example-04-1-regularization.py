#!/usr/bin/python3
"""
-- Regularization
"""

import numpy as np
import pandas as pd
# from scikit-learn
import sklearn.linear_model as sk

# # Import the dataset
# from sklearn.datasets import load_boston
# boston = load_boston()
# features = np.array(boston.data)
# target = np.array(boston.target)

# # Alternatively
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
features = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # 13 features
target = raw_df.values[1::2, 2]  # house price

# This dataset contains information collected by the U.S. Census Bureau concerning housing around the Boston area.
# Each record in the database describes a Boston suburb or town.
# The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970.

# - CRIM: Per capita crime rate by town
# - ZN: Proportion of residential land zoned for lots over 25,000 square feet
# - INDUS: Proportion of non-retail business acres per town
# - CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# - NOX: Nitric oxides concentration (parts per 10 million)
# - RM: Average number of rooms per dwelling
# - AGE: Proportion of owner-occupied units built prior to 1940
# - DIS: Weighted distances to five Boston employment centers
# - RAD: Index of accessibility to radial highways
# - TAX: Full-value property-tax rate per $10,000
# - PTRATIO: Pupil-teacher ratio by town
# - (B − 1000 * (B_k − 0.63)^2 − B_k): Proportion of African Americans by town
# - LSTAT: % lower status of the population
# - MEDV: Median value of owner-occupied homes in $1000s


def normalize_dataset(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    normalized_dataset = (dataset - mu) / sigma
    return normalized_dataset


features_norm = normalize_dataset(features)
np.random.seed(42)
rnd = np.random.rand(len(features_norm)) < 0.8
train_x = features_norm[rnd]
train_y = target[rnd]
dev_x = features_norm[~rnd]
dev_y = target[~rnd]
print(train_x.shape)
# (399, 13)
print(train_y.shape)
# (399,)
print(dev_x.shape)
# (107, 13)
print(dev_y.shape)
# (107,)

