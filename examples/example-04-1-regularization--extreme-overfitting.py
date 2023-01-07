#!/usr/bin/python3
"""
-- Regularization
Issue: Boston area housing and house pricing
"""
import pathlib
import sys
# Get the package directory
package_dir = str(pathlib.Path(__file__).resolve().parents[1])
# Add the package directory into sys.path if necessary
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# tensorflow libraries
from tensorflow import keras
from keras import layers
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling

import importlib
set_style = importlib.import_module("ADL-Book-2nd-Ed.modules.style_setting").set_style

# my modules
from utils.feed_forward import build_keras_model, fit_model
from utils.preparation import normalize_data, split_into_train_and_dev_data

# # Import the dataset from scikit-learn
# from sklearn.datasets import load_boston
# boston = load_boston()
# features = np.array(boston.data)
# target = np.array(boston.target)
# print(boston['DESCR'])

# # Alternatively
data_url = r"http://lib.stat.cmu.edu/datasets/boston"
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


n_training_samples = features.shape[0]
n_dim = features.shape[1]
print('The dataset has', n_training_samples, 'training samples.')
# The dataset has 506 training samples.
print('The dataset has', n_dim, 'features.')
# The dataset has 13 features.

#
# def normalize_dataset(dataset):
#     mu = np.mean(dataset, axis=0)
#     sigma = np.std(dataset, axis=0)
#     normalized_dataset = (dataset - mu) / sigma
#     return normalized_dataset


features_norm, mu, sigma = normalize_data(features)
(train_x, train_y), (dev_x, dev_y) = split_into_train_and_dev_data(features_norm, target)
# np.random.seed(42)  # reproducible random
# rnd = np.random.rand(len(features_norm)) < 0.8
# train_x = features_norm[rnd]
# train_y = target[rnd]
# dev_x = features_norm[~rnd]
# dev_y = target[~rnd]
print(train_x.shape)
# (399, 13)
print(train_y.shape)
# (399,)
print(dev_x.shape)
# (107, 13)
print(dev_y.shape)
# (107,)

STRUCTURE = "20-20-20-20-1"
INPUTS = train_x.shape[1],  # 13
EPOCHS = 10_000
BATCH_SIZE = train_x.shape[0]  # 399

model = build_keras_model(
        num_inputs=INPUTS,
        structure=STRUCTURE,
        hidden_activation="relu",
        initializer=tf.keras.initializers.HeNormal(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=("mse", ),
        model_name="feed-forward-model"
)

learning_history, learning_time = fit_model(
    model, train_x, train_y,
    features_dev=dev_x,
    target_dev=dev_y,
    batch_size=BATCH_SIZE,
    num_epochs=EPOCHS
)
# Cost function at epoch of 0:
# Training MSE = 584.1921997070312
# Dev MSE = 554.349609375
# Cost function at epoch of 10000:
# Training MSE = 0.033795516937971115
# Dev MSE = 25.265743255615234
# Learning time = 3.33 minutes

learning_history.to_parquet(f"../histories/history-04-1-batch_size-{BATCH_SIZE}-structure-{STRUCTURE}.parquet")
# Save the trained model
model.save(f"../models/model-04-1-structure-{STRUCTURE}")

learning_history = pd.read_parquet(f"../histories/history-04-1-batch_size-{BATCH_SIZE}-structure-{STRUCTURE}.parquet")
model = tf.keras.models.load_model(f"../models/model-04-1-structure-{STRUCTURE}")

fp = set_style().set_general_style_parameters()
plt.figure()
plt.plot(learning_history['loss'], ls='-', color='black', lw=3, label='Training MSE')
plt.plot(learning_history['val_loss'], ls='--', color='blue', lw=2, label='Dev MSE')
plt.ylabel('Cost Function (MSE)', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Epochs', fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 50)
plt.legend(loc='best')
plt.axis(True)
plt.title("Extreme Overfitting without Regularization", fontproperties=fm.FontProperties(fname=fp))
# plt.show()
plt.savefig('../figures/figure-04-1-1.svg', bbox_inches='tight')
plt.close()

# predictions
pred_y_train = model.predict(train_x).flatten()
pred_y_dev = model.predict(dev_x).flatten()

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(121)
ax.scatter(train_y, pred_y_train, s=50, color='blue', label=f"MSE Training = {learning_history['loss'].values[-1]:5.4f}")
ax.plot([np.min(np.array(dev_y)), np.max(np.array(dev_y))], [np.min(np.array(dev_y)), np.max(np.array(dev_y))], 'k--', lw=3)
ax.set_xlabel('Measured Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylabel('Predicted Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylim(0, 55)
ax.set_xlim(0, 55)
ax.legend(loc='best')

ax = fig.add_subplot(122)
ax.scatter(dev_y, pred_y_dev, s=50, color='blue', label=f"MSE Dev = {learning_history['val_loss'].values[-1]:5.2f}")
ax.plot([np.min(np.array(dev_y)), np.max(np.array(dev_y))], [np.min(np.array(dev_y)), np.max(np.array(dev_y))], 'k--', lw=3)
ax.set_xlabel('Measured Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylim(0, 55)
ax.set_xlim(0, 55)
ax.legend(loc='best')
plt.suptitle("Extreme Overfitting without Regularization", fontproperties=fm.FontProperties(fname=fp))
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-04-1-2.svg', bbox_inches='tight')
plt.close()
