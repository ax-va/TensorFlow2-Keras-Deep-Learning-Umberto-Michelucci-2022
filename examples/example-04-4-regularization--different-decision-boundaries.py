#!/usr/bin/python3
"""
-- Regularization
---- L2 Regularization
Issue: Random two-features data with the binary target
"""
import pathlib
import sys
# Get the package directory
package_dir = str(pathlib.Path(__file__).resolve().parents[1])
# Add the package directory into sys.path if necessary
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# general libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# tensorflow libraries
from tensorflow import keras
import tensorflow as tf

import importlib
set_style = importlib.import_module("ADL-Book-2nd-Ed.modules.style_setting").set_style

# my modules
from utils.feed_forward import build_keras_model, fit_model

# Create training data
nobs = 30  # number of observations
np.random.seed(42)  # making results reproducible
# first set of observations
features_0_0 = np.random.normal(0.3, 0.15, nobs)
features_0_1 = np.random.normal(0.3, 0.15, nobs)
# second set of observations
features_1_0 = np.random.normal(0.1, 0.1, nobs)
features_1_1 = np.random.normal(0.3, 0.1, nobs)
# Create 2-value dataset
features_0 = np.c_[features_0_0, features_0_1]
print(features_0)
# [[0.37450712 0.20974401]
#  [0.27926035 0.57784173]
#  [0.39715328 0.29797542]
# ...
#  [0.3563547  0.25361814]
#  [0.2099042  0.34968951]
#  [0.25624594 0.44633177]]
# Create 2-value dataset
features_1 = np.c_[features_1_0, features_1_1]
print(features_1)
# [[ 0.05208258  0.30970775]
#  [ 0.0814341   0.3968645 ]
#  [-0.0106335   0.22979469]
# ...
#  [ 0.13287511  0.1831322 ]
#  [ 0.04702398  0.41428228]
#  [ 0.15132674  0.3751933 ]]
# Concatenate both features
features = np.concatenate([features_0, features_1])
print(features)
# [[ 0.37450712  0.20974401]
#  [ 0.27926035  0.57784173]
#  [ 0.39715328  0.29797542]
# ...
#  [ 0.13287511  0.1831322 ]
#  [ 0.04702398  0.41428228]
#  [ 0.15132674  0.3751933 ]]
# creating the labels
target_0 = np.full(nobs, 0, dtype=int)
print(target_0)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
target_1 = np.full(nobs, 1, dtype=int)
print(target_1)
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
target = np.concatenate((target_0, target_1), axis=0)
print(target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

# defining training points and labels
train_x = features
train_y = target

# def create_and_train_regularized_model(data_train_norm, labels_train, num_neurons,
#                                        num_layers, num_epochs, l2_reg_lambda):
#     # Build model
#     # input layer
#     inputs = keras.Input(shape=data_train_norm.shape[1])
#     # He initialization
#     initializer = tf.keras.initializers.HeNormal()
#     # regularization
#     reg = tf.keras.regularizers.l2(l2=l2_reg_lambda)
#     layer = inputs
#     # customized number of layers and neurons per layer
#     for i in range(num_layers):
#         layer = layers.Dense(
#             num_neurons,
#             activation='relu',
#             kernel_initializer=initializer,
#             kernel_regularizer=reg
#         )(layer)
#     # output layer
#     outputs = layers.Dense(1, activation='sigmoid')(layer)
#     model = keras.Model(inputs=inputs, outputs=outputs, name='model')
#     # Set optimizer and loss
#     opt = keras.optimizers.Adam(learning_rate=0.005)
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=opt,
#         metrics=['accuracy']
#     )
#     # Train model
#     history = model.fit(
#         data_train_norm, labels_train,
#         epochs=num_epochs, verbose=0,
#         batch_size=data_train_norm.shape[0])
#     #
#     #
#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch
#     return hist, model

def predict_for_mesh(h, c1, clf):
    """
    Inputs:
    h: mesh step (0.001 is a good value if you move between 0 and 1)
    c1: your training data
    clf: your model

    Outputs:
    xx: x values of the mesh
    yy: y values of the mesh
    z: the prediction (the color of each point)
    """
    # point in the mesh [x_min, x_max] x [y_min, y_max].
    x_min, x_max = c1[:, 0].min(), c1[:, 0].max()
    y_min, y_max = c1[:, 1].min(), c1[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max+h, h), np.arange(y_min, y_max+h, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z

# xx, yy = np.meshgrid(np.arange(-1, 1 + 0.01, 0.01), np.arange(-2, 2 + 0.02, 0.02))
# print(xx)
# # [[-1.   -0.99 -0.98 ...  0.98  0.99  1.  ]
# #  [-1.   -0.99 -0.98 ...  0.98  0.99  1.  ]
# #  [-1.   -0.99 -0.98 ...  0.98  0.99  1.  ]
# #  ...
# #  [-1.   -0.99 -0.98 ...  0.98  0.99  1.  ]
# #  [-1.   -0.99 -0.98 ...  0.98  0.99  1.  ]
# #  [-1.   -0.99 -0.98 ...  0.98  0.99  1.  ]]
# print(yy)
# # [[-2.   -2.   -2.   ... -2.   -2.   -2.  ]
# #  [-1.98 -1.98 -1.98 ... -1.98 -1.98 -1.98]
# #  [-1.96 -1.96 -1.96 ... -1.96 -1.96 -1.96]
# #  ...
# #  [ 1.96  1.96  1.96 ...  1.96  1.96  1.96]
# #  [ 1.98  1.98  1.98 ...  1.98  1.98  1.98]
# #  [ 2.    2.    2.   ...  2.    2.    2.  ]]
# print(np.c_[xx.ravel(), yy.ravel()])
# # [[-1.   -2.  ]
# #  [-0.99 -2.  ]
# #  [-0.98 -2.  ]
# #  ...
# #  [ 0.98  2.  ]
# #  [ 0.99  2.  ]
# #  [ 1.    2.  ]]


def plot_decision_boundary(xx, yy, z, features, target, file, title):
    """
    Inputs:
    xx: values for the mesh (coming from make_mesh_predict())
    yy: values for the mesh (coming from make_mesh_predict())
    z: prediction for each point (that will be mapped to the color)
    features, target: training points
    """
    plt.figure(figsize=(9, 7))
    plt.pcolormesh(xx, yy, z, cmap='Greys', alpha=0.5)
    # Plot also the training points
    plt.scatter(features[:, 0], features[:, 1], c=target, edgecolors='k', s=40, cmap='gray')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig(file, bbox_inches='tight')


STRUCTURE = "20-20-20-20-1"
INPUTS = train_x.shape[1]
print("INPUTS", INPUTS)
EPOCHS = 1000
BATCH_SIZE = train_x.shape[0]
print("BATCH_SIZE", INPUTS)

l2_lambdas = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
for index, l2_lambda in enumerate(l2_lambdas):
    print("*" * 65)
    print(f"L2 lambda: {l2_lambda}")

    model = build_keras_model(
            num_inputs=INPUTS,
            structure=STRUCTURE,
            hidden_activation="relu",
            output_activation="sigmoid",
            initializer=tf.keras.initializers.HeNormal(),
            regularizer=tf.keras.regularizers.l2(l2=l2_lambda),
            optimizer=keras.optimizers.Adam(learning_rate=0.005),
            loss="binary_crossentropy",
            metrics=["accuracy"],
            model_name="feed-forward-model"
    )

    learning_history, learning_time = fit_model(
        model, train_x, train_y,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS
    )
    learning_history.to_parquet(f"../histories/history-04-4-structure-{STRUCTURE}-l2_lambda-{l2_lambda}.parquet")
    # Save the trained model
    model.save(f"../models/model-04-4-structure-{STRUCTURE}-l2_lambda-{l2_lambda}")

    xx, yy, z = predict_for_mesh(0.001, train_x, model)
    plot_decision_boundary(
        xx, yy, z, train_x, train_y,
        file=f'../figures/figure-04-4-{index + 1}.png',
        title=f"Decision Boundary: $L_2$ Regularization with $\lambda={l2_lambda}$"
    )

# Compare with logistic regression
log_reg = linear_model.LogisticRegression()
log_reg.fit(train_x, train_y)
xx, yy, z = predict_for_mesh(0.001, train_x, log_reg)
plot_decision_boundary(
    xx, yy, z, train_x, train_y,
    file=f'../figures/figure-04-4-{len(l2_lambdas) + 1}.png',
    title=f"Decision Boundary with Logistic Regression"
)
