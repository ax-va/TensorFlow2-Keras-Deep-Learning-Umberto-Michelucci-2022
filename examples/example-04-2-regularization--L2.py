#!/usr/bin/python3
"""
-- Regularization
---- L2 Regularization
Issue: Boston area housing and house pricing
"""
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


def normalize_dataset(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    normalized_dataset = (dataset - mu) / sigma
    return normalized_dataset


features_norm = normalize_dataset(features)
np.random.seed(42)  # reproducible random
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


def create_and_train_l2_model(data_train_norm, labels_train, data_dev_norm, labels_dev,
                              num_neurons, num_layers, num_epochs, reg_lambda):
    """
    This function builds and trains a feed-forward neural network model and evaluates it on the training and dev sets.
    """
    # Build model
    inputs = keras.Input(shape=data_train_norm.shape[1])  # input layer
    # He initialization
    initializer = tf.keras.initializers.HeNormal()
    # L2 regularization
    reg = tf.keras.regularizers.l2(l2=reg_lambda)
    layer = inputs
    # customized number of layers and neurons per layer
    for i in range(num_layers):
        layer = layers.Dense(
            num_neurons,
            activation='relu',
            kernel_initializer=initializer,
            kernel_regularizer=reg
        )(layer)  # hidden layers
    # output layer: one neuron with the identity activation function
    outputs = layers.Dense(1)(layer)
    keras_model = keras.Model(inputs=inputs, outputs=outputs, name='model')
    # Set optimizer and loss
    opt = keras.optimizers.Adam(learning_rate=0.001)
    keras_model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    # Train model
    result = keras_model.fit(
        data_train_norm, labels_train,
        epochs=num_epochs, verbose=0,
        batch_size=data_train_norm.shape[0],
        validation_data=(data_dev_norm, labels_dev),
        callbacks=[tfdocs.modeling.EpochDots()]
    )
    history = pd.DataFrame(result.history)
    history['epoch'] = result.epoch
    # print performances
    print('Cost function at epoch of 0:')
    print(f"Training MSE = {history['loss'].values[0]}")
    print(f"Dev MSE = {history['val_loss'].values[0]}")
    print(f'Cost function at epoch of {num_epochs}:')
    print(f"Training MSE = {history['loss'].values[-1]}")
    print(f"Dev MSE = {history['val_loss'].values[-1]}")
    return history, keras_model


num_l, num_n = 4, 20
reg_lambdas = [0.0, 10.0]
# for reg_lambda in reg_lambdas:
#     hist, model = create_and_train_l2_model(
#         train_x, train_y, dev_x, dev_y,
#         num_neurons=num_n, num_layers=num_l, num_epochs=10000, reg_lambda=reg_lambda
#     )
#     hist.to_csv(f"../histories/history-04-2-num_n-{num_n}-num_l-{num_l}-L2_reg-{reg_lambda}.csv")
#     model.save(f"../models/model-04-2-num_n-{num_n}-num_l-{num_l}-L2_reg-{reg_lambda}")


hist = pd.read_csv(f"../histories/history-04-2-num_n-{num_n}-num_l-{num_l}-L2_reg-10.0.csv")
model_reg = tf.keras.models.load_model(f"../models/model-04-2-num_n-{num_n}-num_l-{num_l}-L2_reg-10.0")
model_not_reg = tf.keras.models.load_model(f"../models/model-04-2-num_n-{num_n}-num_l-{num_l}-L2_reg-0.0")

fp = set_style().set_general_style_parameters()
plt.figure()
plt.plot(hist['loss'], ls='-', color='black', lw=3, label='Training MSE')
plt.plot(hist['val_loss'], ls='--', color='blue', lw=2, label='Dev MSE')
plt.ylabel('Cost Function (MSE)', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Number of Iterations', fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 200)
plt.legend(loc='best')
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-04-2-1.svg', bbox_inches='tight')

# predictions
pred_y_train = model_reg.predict(train_x).flatten()
pred_y_dev = model_reg.predict(dev_x).flatten()

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(121)
ax.scatter(train_y, pred_y_train, s=50, color='blue', label=f"MSE Training = {hist['loss'].values[-1]:5.4f}")
ax.plot([np.min(np.array(dev_y)), np.max(np.array(dev_y))], [np.min(np.array(dev_y)), np.max(np.array(dev_y))], 'k--', lw=3)
ax.set_xlabel('Measured Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylabel('Predicted Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylim(0, 55)
ax.legend(loc='best')

ax = fig.add_subplot(122)
ax.scatter(dev_y, pred_y_dev, s=50, color='blue', label=f"MSE Dev = {hist['val_loss'].values[-1]:5.2f}")
ax.plot([np.min(np.array(dev_y)), np.max(np.array(dev_y))], [np.min(np.array(dev_y)), np.max(np.array(dev_y))], 'k--', lw=3)
ax.set_xlabel('Measured Target Value', fontproperties=fm.FontProperties(fname=fp))
ax.set_ylim(0, 55)
ax.legend(loc='best')
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-04-2-2.svg', bbox_inches='tight')

# weights of non-regularized network
weights1_not_reg = model_not_reg.layers[1].get_weights()
weights2_not_reg = model_not_reg.layers[2].get_weights()
weights3_not_reg = model_not_reg.layers[3].get_weights()
weights4_not_reg = model_not_reg.layers[4].get_weights()

print(type(weights1_not_reg[0]))  # <class 'numpy.ndarray'>
print(type(weights1_not_reg[1]))  # <class 'numpy.ndarray'>

print(weights1_not_reg[0].shape)  # 13 features to each of 20 neurons
# (13, 20)
print(weights1_not_reg[1].shape)  # 1 bias to each of 20 neurons
# (20,)

print(weights2_not_reg[0].shape)  # 20 inputs to each of 20 neurons
# (20, 20)
print(weights2_not_reg[1].shape)  # 1 bias to each of 20 neurons
# (20,)

print(weights3_not_reg[0].shape)  # 20 inputs to each of 20 neurons
# (20, 20)
print(weights3_not_reg[1].shape)  # 1 bias to each of 20 neurons
# (20,)

print(weights4_not_reg[0].shape)  # 20 inputs to each of 20 neurons
# (20, 20)
print(weights4_not_reg[1].shape)  # 1 bias to each of 20 neurons
# (20,)

# weights of regularized network
weights1_reg = model_reg.layers[1].get_weights()
weights2_reg = model_reg.layers[2].get_weights()
weights3_reg = model_reg.layers[3].get_weights()
weights4_reg = model_reg.layers[4].get_weights()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(221)
plt.hist(weights1_not_reg[0].flatten(), alpha=0.25, bins=10, color='black')
plt.hist(weights1_reg[0].flatten(), alpha=0.5, bins=5, color='black')
ax.set_ylabel('Count', fontproperties=fm.FontProperties(fname=fp))
ax.text(-1, 150, 'Layer 1', fontproperties=fm.FontProperties(fname=fp))
plt.xticks(fontproperties= fm.FontProperties(fname=fp))
plt.yticks(fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 350)

ax = fig.add_subplot(222)
plt.hist(weights2_not_reg[0].flatten(), alpha=0.25, bins=10, color='black')
plt.hist(weights2_reg[0].flatten(), alpha=0.5, bins=5, color='black')
ax.text(-1.25, 150, 'Layer 2', fontproperties=fm.FontProperties(fname=fp))
plt.xticks(fontproperties=fm.FontProperties(fname=fp))
plt.yticks(fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 350)

ax = fig.add_subplot(223)
plt.hist(weights3_not_reg[0].flatten(), alpha=0.25, bins=10, color='black')
plt.hist(weights3_reg[0].flatten(), alpha=0.5, bins=5, color='black')
ax.set_ylabel('Count', fontproperties=fm.FontProperties(fname=fp))
ax.set_xlabel('Weights', fontproperties=fm.FontProperties(fname=fp))
ax.text(-2.30, 150, 'Layer 3', fontproperties=fm.FontProperties(fname=fp))
plt.xticks(fontproperties=fm.FontProperties(fname=fp))
plt.yticks(fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 400)

ax = fig.add_subplot(224)
plt.hist(weights4_not_reg[0].flatten(), alpha=0.25, bins=10, color='black')
plt.hist(weights4_reg[0].flatten(), alpha=0.5, bins=5, color='black')
ax.set_xlabel('Weights', fontproperties=fm.FontProperties(fname=fp))
ax.text(-2.30, 150, 'Layer 4', fontproperties=fm.FontProperties(fname=fp))
plt.xticks(fontproperties=fm.FontProperties(fname=fp))
plt.yticks(fontproperties=fm.FontProperties(fname=fp))
plt.ylim(0, 400)
plt.savefig('../figures/figure-04-2-3.svg', bbox_inches='tight')

print(f"Percent of weight less than 1e−3 for λ = {reg_lambdas[0]}")
print('First hidden layer:')
print(f"{(np.sum(np.abs(weights1_not_reg[0]) < 1e-3)) / weights1_not_reg[0].size * 100.0:.2f}%")
print('Second hidden layer:')
print(f"{(np.sum(np.abs(weights2_not_reg[0]) < 1e-3)) / weights2_not_reg[0].size * 100.0:.2f}%")
print('Third hidden layer:')
print(f"{(np.sum(np.abs(weights3_not_reg[0]) < 1e-3)) / weights3_not_reg[0].size * 100.0:.2f}%")
print('Fourth hidden layer:')
print(f"{(np.sum(np.abs(weights4_not_reg[0]) < 1e-3)) / weights4_not_reg[0].size * 100.0:.2f}%")
# Percent of weight less than 1e−3 for λ = 0.0
# First hidden layer:
# 0.00%
# Second hidden layer:
# 0.50%
# Third hidden layer:
# 0.00%
# Fourth hidden layer:
# 0.25%

print(f"Percent of weight less than 1e−3 for λ = {reg_lambdas[1]}")
print('First hidden layer:')
print(f"{(np.sum(np.abs(weights1_reg[0]) < 1e-3)) / weights1_reg[0].size * 100.0:.2f}%")
print('Second hidden layer:')
print(f"{(np.sum(np.abs(weights2_reg[0]) < 1e-3)) / weights2_reg[0].size * 100.0:.2f}%")
print('Third hidden layer:')
print(f"{(np.sum(np.abs(weights3_reg[0]) < 1e-3)) / weights3_reg[0].size * 100.0:.2f}%")
print('Fourth hidden layer:')
print(f"{(np.sum(np.abs(weights4_reg[0]) < 1e-3)) / weights4_reg[0].size * 100.0:.2f}%")
# Percent of weight less than 1e−3 for λ = 10.0
# First hidden layer:
# 3.85%
# Second hidden layer:
# 61.50%
# Third hidden layer:
# 78.25%
# Fourth hidden layer:
# 75.25%
