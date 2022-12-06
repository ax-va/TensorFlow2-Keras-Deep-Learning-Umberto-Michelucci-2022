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


def create_and_train_l2_reg_model(data_train_norm, labels_train, data_dev_norm, labels_dev,
                                  num_neurons, num_layers, num_epochs, l2_reg_lambda):
    """
    This function builds and trains a feed-forward neural network model and evaluates it on the training and dev sets.
    """
    # Build model
    inputs = keras.Input(shape=data_train_norm.shape[1])  # input layer
    # He initialization
    initializer = tf.keras.initializers.HeNormal()
    # L2 regularization
    reg = tf.keras.regularizers.l2(l2=l2_reg_lambda)
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
reg_lambdas = np.arange(0.0, 30.5, 0.5)

# train_mse_values = []
# dev_mse_values = []
# for reg_lambda in reg_lambdas:
#     print("L2 regularization lambda:", reg_lambda)
#     hist, model = create_and_train_l2_reg_model(
#         train_x, train_y, dev_x, dev_y,
#         num_neurons=num_n, num_layers=num_l, num_epochs=10000, l2_reg_lambda=reg_lambda
#     )
#     train_mse_loss = hist['loss'].values[-1]
#     dev_mse_loss = hist['val_loss'].values[-1]
#     train_mse_values.append(train_mse_loss)
#     dev_mse_values.append(dev_mse_loss)
#
# mse_loss_data = {
#     "L2_lambda": reg_lambdas,
#     "training_MSE": train_mse_values,
#     "dev_MSE": dev_mse_values
# }
# df_mse_loss = pd.DataFrame(mse_loss_data).set_index("L2_lambda")
# df_mse_loss.to_csv("../histories/history-04-3-mse-loss-for-different-L2-lambdas.csv")

df_mse_loss = pd.read_csv(f"../histories/history-04-3-mse-loss-for-different-L2-lambdas.csv").set_index("L2_lambda")
lambda_values = df_mse_loss.index
train_mse = df_mse_loss["training_MSE"]
dev_mse = df_mse_loss["dev_MSE"]
fp = set_style().set_general_style_parameters()
plt.figure()
plt.plot(lambda_values, train_mse, ls='-', color='black', lw=3, label='Training MSE')
plt.plot(lambda_values, dev_mse, ls='--', color='blue', lw=2, label='Dev MSE')
plt.ylabel('Cost Function (MSE)', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('$\lambda$', fontproperties=fm.FontProperties(fname=fp))
plt.legend(loc='best')
plt.axis(True)
plt.savefig('../figures/figure-04-3.svg', bbox_inches='tight')
