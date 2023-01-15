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
import tensorflow as tf

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

features_norm, mu, sigma = normalize_data(features)
(train_x, train_y), (dev_x, dev_y) = split_into_train_and_dev_data(features_norm, target)
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
l2_lambdas = [0.0, 10.0]

for l2_lambda in l2_lambdas:
    print("*" * 65)
    print(f"L2 lambda: {l2_lambda}")

    model = build_keras_model(
            num_inputs=INPUTS,
            structure=STRUCTURE,
            hidden_activation="relu",
            initializer=tf.keras.initializers.HeNormal(),
            regularizer=tf.keras.regularizers.l2(l2=l2_lambda),
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
    learning_history.to_parquet(f"../histories/history-04-2-structure-{STRUCTURE}-l2_lambda-{l2_lambda}.parquet")
    # Save the trained model
    model.save(f"../models/model-04-2-structure-{STRUCTURE}-l2_lambda-{l2_lambda}")
# *****************************************************************
# L2 lambda: 0.0
# ...
# Cost function at epoch of 0:
# Training MSE = 522.5780639648438
# Dev MSE = 491.3822021484375
# Cost function at epoch of 10000:
# Training MSE = 0.020333461463451385
# Dev MSE = 16.777416229248047
# Learning time = 3.34 minutes
# *****************************************************************
# L2 lambda: 10.0
# ...
# Cost function at epoch of 0:
# Training MSE = 2112.442626953125
# Dev MSE = 2078.31201171875
# Cost function at epoch of 10000:
# Training MSE = 43.4915885925293
# Dev MSE = 44.756187438964844
# Learning time = 3.29 minutes

learning_history_dict = {}
model_dict = {}
for l2_lambda in l2_lambdas:
    learning_history_dict[l2_lambda] = pd.read_parquet(f"../histories/history-04-2-structure-{STRUCTURE}-l2_lambda-{l2_lambda}.parquet")
    model_dict[l2_lambda] = tf.keras.models.load_model(f"../models/model-04-2-structure-{STRUCTURE}-l2_lambda-{l2_lambda}")

fp = set_style().set_general_style_parameters()
plt.figure()
for l2_lambda in l2_lambdas:
    label_train = 'Training: $\lambda = ' + str(l2_lambda) + '$'
    plt.plot(learning_history_dict[l2_lambda]['loss'], ls='-', lw=3, label=label_train)
    label_dev = 'Dev: $\lambda = ' + str(l2_lambda) + '$'
    plt.plot(learning_history_dict[l2_lambda]['val_loss'], ls='--', lw=2, label=label_dev)
plt.ylabel('Cost Function (MSE)', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Epochs', fontproperties=fm.FontProperties(fname=fp))
title = f'$L_2$ Regularization'
plt.title(title, fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 200)
plt.legend(loc='upper right', fontsize='xx-small')
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-04-2-1.svg', bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(121)
for l2_lambda in list(reversed(l2_lambdas)):
    # predictions for training data
    pred_y_train = model_dict[l2_lambda].predict(train_x).flatten()
    label = f"MSE Training = {learning_history_dict[l2_lambda]['loss'].values[-1]:5.4f}; $\lambda = {l2_lambda}$"
    ax.scatter(train_y, pred_y_train, s=50, label=label)
ax.plot([np.min(np.array(dev_y)), np.max(np.array(dev_y))], [np.min(np.array(dev_y)), np.max(np.array(dev_y))], 'k--', lw=3)
ax.set_xlabel('Measured Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylabel('Predicted Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylim(0, 55)
ax.set_xlim(0, 55)
ax.legend(loc='upper left', fontsize='xx-small')

ax = fig.add_subplot(122)
for l2_lambda in list(reversed(l2_lambdas)):
    # predictions for dev data
    pred_y_dev = model_dict[l2_lambda].predict(dev_x).flatten()
    label = f"MSE Dev = {learning_history_dict[l2_lambda]['val_loss'].values[-1]:5.2f}; $\lambda = {l2_lambda}$"
    ax.scatter(dev_y, pred_y_dev, s=50, label=label)
ax.plot([np.min(np.array(dev_y)), np.max(np.array(dev_y))], [np.min(np.array(dev_y)), np.max(np.array(dev_y))], 'k--', lw=3)
ax.set_xlabel('Measured Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylim(0, 55)
ax.set_xlim(0, 55)
ax.legend(loc='upper left', fontsize='xx-small')
title = f'$L_2$ Regularization'
plt.suptitle(title, fontproperties=fm.FontProperties(fname=fp))
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-04-2-2.svg', bbox_inches='tight')
plt.close()

for l2_lambda in l2_lambdas:
    print("*" * 65)
    print(f"L2 lambda: {l2_lambda}")
    # layer weights
    weights_layer_1 = model_dict[l2_lambda].layers[1].get_weights()
    weights_layer_2 = model_dict[l2_lambda].layers[2].get_weights()
    weights_layer_3 = model_dict[l2_lambda].layers[3].get_weights()
    weights_layer_4 = model_dict[l2_lambda].layers[4].get_weights()
    print(type(weights_layer_1[0]))  # <class 'numpy.ndarray'>
    print(type(weights_layer_1[1]))  # <class 'numpy.ndarray'>
    print(weights_layer_1[0].shape)  # 13 features to each of 20 neurons  # (13, 20)
    print(weights_layer_1[1].shape)  # 1 bias to each of 20 neurons  # (20,)
    print(weights_layer_2[0].shape)  # 20 inputs to each of 20 neurons  # (20, 20)
    print(weights_layer_2[1].shape)  # 1 bias to each of 20 neurons  # (20,)
    print(weights_layer_3[0].shape)  # 20 inputs to each of 20 neurons  # (20, 20)
    print(weights_layer_3[1].shape)  # 1 bias to each of 20 neurons  # (20,)
    print(weights_layer_4[0].shape)  # 20 inputs to each of 20 neurons  # (20, 20)
    print(weights_layer_4[1].shape)  # 1 bias to each of 20 neurons  # (20,)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(221)
alpha = 0.25
bins = 10
for l2_lambda in l2_lambdas:
    weights = model_dict[l2_lambda].layers[1].get_weights()
    plt.hist(weights[0].flatten(), alpha=alpha, bins=bins, color='black')
    alpha += 0.25
    bins += 10
ax.set_ylabel('Count', fontproperties=fm.FontProperties(fname=fp))
ax.text(-1, 150, 'Layer 1', fontproperties=fm.FontProperties(fname=fp))
plt.xticks(fontproperties= fm.FontProperties(fname=fp))
plt.yticks(fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 350)

ax = fig.add_subplot(222)
alpha = 0.25
bins = 10
for l2_lambda in l2_lambdas:
    weights = model_dict[l2_lambda].layers[2].get_weights()
    plt.hist(weights[0].flatten(), alpha=alpha, bins=bins, color='black')
    alpha += 0.25
    bins += 10
ax.text(-1.25, 150, 'Layer 2', fontproperties=fm.FontProperties(fname=fp))
plt.xticks(fontproperties=fm.FontProperties(fname=fp))
plt.yticks(fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 350)

ax = fig.add_subplot(223)
alpha = 0.25
bins = 10
for l2_lambda in l2_lambdas:
    weights = model_dict[l2_lambda].layers[3].get_weights()
    plt.hist(weights[0].flatten(), alpha=alpha, bins=bins, color='black')
    alpha += 0.25
    bins += 10
ax.set_ylabel('Count', fontproperties=fm.FontProperties(fname=fp))
ax.set_xlabel('Weights', fontproperties=fm.FontProperties(fname=fp))
ax.text(-2.30, 150, 'Layer 3', fontproperties=fm.FontProperties(fname=fp))
plt.xticks(fontproperties=fm.FontProperties(fname=fp))
plt.yticks(fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 400)

ax = fig.add_subplot(224)
alpha = 0.25
bins = 10
for l2_lambda in l2_lambdas:
    weights = model_dict[l2_lambda].layers[4].get_weights()
    plt.hist(weights[0].flatten(), alpha=alpha, bins=bins, color='black')
    alpha += 0.25
    bins += 10
ax.set_xlabel('Weights', fontproperties=fm.FontProperties(fname=fp))
ax.text(-2.30, 150, 'Layer 4', fontproperties=fm.FontProperties(fname=fp))
plt.xticks(fontproperties=fm.FontProperties(fname=fp))
plt.yticks(fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 400)
title = f'$L_2$ Regularization'
plt.suptitle(title, fontproperties=fm.FontProperties(fname=fp))
plt.savefig('../figures/figure-04-2-3.svg', bbox_inches='tight')
plt.close()

for l2_lambda in l2_lambdas:
    weights_layer_1 = model_dict[l2_lambda].layers[1].get_weights()
    weights_layer_2 = model_dict[l2_lambda].layers[2].get_weights()
    weights_layer_3 = model_dict[l2_lambda].layers[3].get_weights()
    weights_layer_4 = model_dict[l2_lambda].layers[4].get_weights()
    print("*" * 65)
    print(f"λ = {l2_lambda}")
    print(f"Percent of weight less than 1e−3")
    print('First hidden layer:')
    print(f"{(np.sum(np.abs(weights_layer_1[0]) < 1e-3)) / weights_layer_1[0].size * 100.0:.2f}%")
    print('Second hidden layer:')
    print(f"{(np.sum(np.abs(weights_layer_2[0]) < 1e-3)) / weights_layer_2[0].size * 100.0:.2f}%")
    print('Third hidden layer:')
    print(f"{(np.sum(np.abs(weights_layer_3[0]) < 1e-3)) / weights_layer_3[0].size * 100.0:.2f}%")
    print('Fourth hidden layer:')
    print(f"{(np.sum(np.abs(weights_layer_4[0]) < 1e-3)) / weights_layer_4[0].size * 100.0:.2f}%")
# *****************************************************************
# λ = 0.0
# Percent of weight less than 1e−3
# First hidden layer:
# 0.00%
# Second hidden layer:
# 0.25%
# Third hidden layer:
# 0.00%
# Fourth hidden layer:
# 0.50%
# *****************************************************************
# λ = 10.0
# Percent of weight less than 1e−3
# First hidden layer:
# 20.00%
# Second hidden layer:
# 31.75%
# Third hidden layer:
# 46.50%
# Fourth hidden layer:
# 61.00%
