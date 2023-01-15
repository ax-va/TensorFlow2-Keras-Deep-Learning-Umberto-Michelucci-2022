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
l2_lambdas = np.arange(0.0, 21.0, 1.0)

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
    learning_history.to_parquet(f"../histories/history-04-3-structure-{STRUCTURE}-l2_lambda-{l2_lambda}.parquet")
    # Save the trained model
    model.save(f"../models/model-04-3-structure-{STRUCTURE}-l2_lambda-{l2_lambda}")
# *****************************************************************
# L2 lambda: 0.0
# ...
# Cost function at epoch of 0:
# Training MSE = 626.89404296875
# Dev MSE = 600.2142333984375
# Cost function at epoch of 10000:
# Training MSE = 0.03501132130622864
# Dev MSE = 22.486377716064453
# Learning time = 3.35 minutes
# *****************************************************************
# L2 lambda: 1.0
# ...
# Cost function at epoch of 0:
# Training MSE = 733.9525146484375
# Dev MSE = 705.9791259765625
# Cost function at epoch of 10000:
# Training MSE = 11.44808578491211
# Dev MSE = 21.361793518066406
# Learning time = 3.29 minutes
# *****************************************************************
# L2 lambda: 2.0
# ...
# Cost function at epoch of 0:
# Training MSE = 895.7713623046875
# Dev MSE = 866.9102172851562
# Cost function at epoch of 10000:
# Training MSE = 17.690807342529297
# Dev MSE = 24.95559310913086
# Learning time = 3.31 minutes
# *****************************************************************
# L2 lambda: 3.0
# ...
# Cost function at epoch of 0:
# Training MSE = 1062.906005859375
# Dev MSE = 1033.38818359375
# Cost function at epoch of 10000:
# Training MSE = 21.513370513916016
# Dev MSE = 28.429540634155273
# Learning time = 3.25 minutes
# *****************************************************************
# L2 lambda: 4.0
# ...
# Cost function at epoch of 0:
# Training MSE = 1226.1402587890625
# Dev MSE = 1195.1376953125
# Cost function at epoch of 10000:
# Training MSE = 27.658634185791016
# Dev MSE = 33.05376434326172
# Learning time = 3.26 minutes
# *****************************************************************
# L2 lambda: 5.0
# ...
# Cost function at epoch of 0:
# Training MSE = 1432.65576171875
# Dev MSE = 1398.798583984375
# Cost function at epoch of 10000:
# Training MSE = 32.09239959716797
# Dev MSE = 36.42091751098633
# Learning time = 3.26 minutes
# *****************************************************************
# L2 lambda: 6.0
# ...
# Cost function at epoch of 0:
# Training MSE = 1556.39208984375
# Dev MSE = 1518.285888671875
# Cost function at epoch of 10000:
# Training MSE = 33.26327133178711
# Dev MSE = 37.150428771972656
# Learning time = 3.24 minutes
# *****************************************************************
# L2 lambda: 7.0
# ...
# Cost function at epoch of 0:
# Training MSE = 1714.01123046875
# Dev MSE = 1680.162109375
# Cost function at epoch of 10000:
# Training MSE = 35.58831787109375
# Dev MSE = 38.702423095703125
# Learning time = 3.26 minutes
# *****************************************************************
# L2 lambda: 8.0
# ...
# Cost function at epoch of 0:
# Training MSE = 1797.3507080078125
# Dev MSE = 1764.950927734375
# Cost function at epoch of 10000:
# Training MSE = 41.39790725708008
# Dev MSE = 43.11688232421875
# Learning time = 3.25 minutes
# *****************************************************************
# L2 lambda: 9.0
# ...
# Cost function at epoch of 0:
# Training MSE = 2154.796142578125
# Dev MSE = 2116.3916015625
# Cost function at epoch of 10000:
# Training MSE = 43.16845703125
# Dev MSE = 44.47845458984375
# Learning time = 3.25 minutes
# *****************************************************************
# L2 lambda: 10.0
# ...
# Cost function at epoch of 0:
# Training MSE = 2266.801513671875
# Dev MSE = 2229.30419921875
# Cost function at epoch of 10000:
# Training MSE = 46.478858947753906
# Dev MSE = 47.169822692871094
# Learning time = 3.27 minutes
# *****************************************************************
# L2 lambda: 11.0
# ...
# Cost function at epoch of 0:
# Training MSE = 2447.818359375
# Dev MSE = 2404.572265625
# Cost function at epoch of 10000:
# Training MSE = 46.241233825683594
# Dev MSE = 46.96966552734375
# Learning time = 3.28 minutes
# *****************************************************************
# L2 lambda: 12.0
# ...
# Cost function at epoch of 0:
# Training MSE = 2466.9013671875
# Dev MSE = 2429.027099609375
# Cost function at epoch of 10000:
# Training MSE = 49.73603439331055
# Dev MSE = 49.32621765136719
# Learning time = 3.25 minutes
# *****************************************************************
# L2 lambda: 13.0
# ...
# Cost function at epoch of 0:
# Training MSE = 2567.984619140625
# Dev MSE = 2532.5068359375
# Cost function at epoch of 10000:
# Training MSE = 52.43196487426758
# Dev MSE = 51.6077880859375
# Learning time = 3.27 minutes
# *****************************************************************
# L2 lambda: 14.0
# ...
# Cost function at epoch of 0:
# Training MSE = 2797.908935546875
# Dev MSE = 2755.8134765625
# Cost function at epoch of 10000:
# Training MSE = 54.16257095336914
# Dev MSE = 52.76136779785156
# Learning time = 3.24 minutes
# *****************************************************************
# L2 lambda: 15.0
# ...
# Cost function at epoch of 0:
# Training MSE = 2939.24072265625
# Dev MSE = 2902.709716796875
# Cost function at epoch of 10000:
# Training MSE = 55.112220764160156
# Dev MSE = 53.0665397644043
# Learning time = 3.30 minutes
# *****************************************************************
# L2 lambda: 16.0
# ...
# Cost function at epoch of 0:
# Training MSE = 3277.182861328125
# Dev MSE = 3235.850830078125
# Cost function at epoch of 10000:
# Training MSE = 58.7789306640625
# Dev MSE = 56.36931228637695
# Learning time = 3.29 minutes
# *****************************************************************
# L2 lambda: 17.0
# ...
# Cost function at epoch of 0:
# Training MSE = 3408.79296875
# Dev MSE = 3373.200927734375
# Cost function at epoch of 10000:
# Training MSE = 60.382625579833984
# Dev MSE = 58.199501037597656
# Learning time = 3.28 minutes
# *****************************************************************
# L2 lambda: 18.0
# ...
# Cost function at epoch of 0:
# Training MSE = 3364.732666015625
# Dev MSE = 3324.93896484375
# Cost function at epoch of 10000:
# Training MSE = 60.67520523071289
# Dev MSE = 58.06932830810547
# Learning time = 3.27 minutes
# *****************************************************************
# L2 lambda: 19.0
# ...
# Cost function at epoch of 0:
# Training MSE = 3635.0615234375
# Dev MSE = 3588.9677734375
# Cost function at epoch of 10000:
# Training MSE = 65.37531280517578
# Dev MSE = 62.287384033203125
# Learning time = 3.28 minutes
# *****************************************************************
# L2 lambda: 20.0
# ...
# Cost function at epoch of 0:
# Training MSE = 3860.35107421875
# Dev MSE = 3814.7392578125
# Cost function at epoch of 10000:
# Training MSE = 85.10855102539062
# Dev MSE = 81.89582061767578
# Learning time = 3.27 minutes

train_mse = []
dev_mse = []
for l2_lambda in l2_lambdas:
    learning_history = pd.read_parquet(
        f"../histories/history-04-3-structure-{STRUCTURE}-l2_lambda-{l2_lambda}.parquet")
    train_mse_loss = learning_history['loss'].values[-1]
    dev_mse_loss = learning_history['val_loss'].values[-1]
    train_mse.append(train_mse_loss)
    dev_mse.append(dev_mse_loss)

fp = set_style().set_general_style_parameters()
plt.figure()
plt.plot(l2_lambdas, train_mse, ls='-', color='black', lw=3, label='Training MSE')
plt.plot(l2_lambdas, dev_mse, ls='--', color='blue', lw=2, label='Dev MSE')
plt.ylabel('Cost Function (MSE)', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('$\lambda$', fontproperties=fm.FontProperties(fname=fp))
plt.title("Different $L_2$ Regularization $\lambda$", fontproperties=fm.FontProperties(fname=fp))
plt.legend(loc='best')
plt.axis(True)
plt.savefig('../figures/figure-04-3.svg', bbox_inches='tight')
