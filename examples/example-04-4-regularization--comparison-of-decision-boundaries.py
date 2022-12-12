#!/usr/bin/python3
"""
-- Regularization
---- L2 Regularization
Issue: Random two-features data with the binary target
"""
# general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# tensorflow libraries
from tensorflow import keras
from keras import layers
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling

import importlib
set_style = importlib.import_module("ADL-Book-2nd-Ed.modules.style_setting").set_style

# Create training data
nobs = 30  # number of observations
np.random.seed(42)  # making results reproducible
# first set of observations
features_x0_0 = np.random.normal(0.3, 0.15, nobs)
features_x1_0 = np.random.normal(0.3, 0.15, nobs)
# second set of observations
features_x0_1 = np.random.normal(0.1, 0.1, nobs)
features_x1_1 = np.random.normal(0.3, 0.1, nobs)
# concatenating observations
features_0 = np.c_[features_x0_0, features_x1_0]
print(features_0)
# [[0.37450712 0.20974401]
#  [0.27926035 0.57784173]
#  [0.39715328 0.29797542]
# ...
#  [0.3563547  0.25361814]
#  [0.2099042  0.34968951]
#  [0.25624594 0.44633177]]
features_1 = np.c_[features_x0_1, features_x1_1]
print(features_1)
# [[ 0.05208258  0.30970775]
#  [ 0.0814341   0.3968645 ]
#  [-0.0106335   0.22979469]
# ...
#  [ 0.13287511  0.1831322 ]
#  [ 0.04702398  0.41428228]
#  [ 0.15132674  0.3751933 ]]
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


def create_and_train_regularized_model(data_train_norm, labels_train, num_neurons,
                                       num_layers, num_epochs, l2_reg_lambda):
    # Build model
    # input layer
    inputs = keras.Input(shape=data_train_norm.shape[1])
    # He initialization
    initializer = tf.keras.initializers.HeNormal()
    # regularization
    reg = tf.keras.regularizers.l2(l2=l2_reg_lambda)
    layer = inputs
    # customized number of layers and neurons per layer
    for i in range(num_layers):
        layer = layers.Dense(
            num_neurons,
            activation='relu',
            kernel_initializer=initializer,
            kernel_regularizer=reg
        )(layer)
    # output layer
    outputs = layers.Dense(1, activation='sigmoid')(layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name='model')
    # Set optimizer and loss
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    # Train model
    history = model.fit(
        data_train_norm, labels_train,
        epochs=num_epochs, verbose=0,
        batch_size=data_train_norm.shape[0])

    # save performances
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    return hist, model


def make_mesh_predict(h, c1, clf):
    """
    Inputs:
    h -> mesh step (0.001 is a good value if you move between 0 and 1)
    c1 -> your training data
    clf -> your model

    Outputs:
    xx -> x values of the mesh
    yy -> y values of the mesh
    z -> the prediction (the color of each point)
    """
    # point in the mesh [x_min, x_max] x [y_min, y_max].
    x_min, x_max = c1[:, 0].min(), c1[:, 0].max()
    y_min, y_max = c1[:, 1].min(), c1[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max+h, h), np.arange(y_min, y_max+h, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    return xx, yy, z

# xx, yy = np.meshgrid(np.arange(-1, 1+0.01, 0.01), np.arange(-2, 2+0.02, 0.02))
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
    xx: the values for the mesh (coming from make_mesh_predict())
    yy: the values for the mesh (coming from make_mesh_predict())
    z: the prediction for each point (that will be mapped to the color)
    features, target: the training points
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


num_n, num_l, num_epochs = 20, 4, 100

l2_reg_lambdas = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
index = 1
for l2_reg_lambda in l2_reg_lambdas:
    hist_reg, model_reg = create_and_train_regularized_model(
        train_x, train_y,
        num_neurons=num_n, num_layers=num_l,
        num_epochs=num_epochs,
        l2_reg_lambda=l2_reg_lambda
    )
    xx, yy, z = make_mesh_predict(0.001, train_x, model_reg)
    plot_decision_boundary(
        xx, yy, z, train_x, train_y,
        file=f'../figures/figure-04-4-{index}.png',
        title=f"Decision Boundary by $L_2$ Regularization with $\lambda={l2_reg_lambda}$"
    )
    index += 1

# Compare with logistic regression
logreg = linear_model.LogisticRegression()
logreg.fit(train_x, train_y)
xx, yy, z = make_mesh_predict(0.001, train_x, logreg)
plot_decision_boundary(
    xx, yy, z, train_x, train_y,
    file=f'../figures/figure-04-4-{index}.png',
    title=f"Decision Boundary by Logistic Regression"
)
