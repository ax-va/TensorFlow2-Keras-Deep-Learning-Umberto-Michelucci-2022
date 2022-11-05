#!/usr/bin/python3
"""
-- Linear Regression Model with NumPy: Classical Solution
"""

# general libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import importlib
read_radon_dataset = importlib.import_module("ADL-Book-2nd-Ed.modules.read_radon_dataset")
style_setting = importlib.import_module("ADL-Book-2nd-Ed.modules.style_setting")
read_data = read_radon_dataset.read_data
set_style = style_setting.set_style

# inputs to download the dataset
DATASET_DIR = os.path.join('../datasets', 'radon')
url_base = 'http://www.stat.columbia.edu/~gelman/arm/examples/radon/'
# Alternative source:
# url_base = ('https://raw.githubusercontent.com/pymc-devs/uq_chapter/master/reference/data/')

rd = read_data(DATASET_DIR, url_base)
radon_features, radon_labels, county_name = rd.create_dataset()

# Data splitting
np.random.seed(42)
rnd = np.random.rand(len(radon_features)) < 0.8
# 80%
train_x = radon_features[rnd]  # training dataset (features, i.e. inputs)
train_y = radon_labels[rnd]  # training dataset (labels, i.e. outputs)
# 20%
test_x = radon_features[~rnd]  # testing dataset (features, i.e inputs)
test_y = radon_labels[~rnd]  # testing dataset (labels, i.e. outputs)
print('The training dataset dimensions are: ', train_x.shape)
# The training dataset dimensions are:  (733, 4)
print('The testing dataset dimensions are: ', test_x.shape)
# The testing dataset dimensions are:  (186, 4)

# Solution:
# weights = (X^T * X)^{-1} * X^T * y

# Get NumPy arrays from the DataFrame's columns
X_train = train_x.values
X_test = test_x.values
y_train = train_y.values
print(X_train)
# [[ 1.        0.        0.502054  9.7     ]
#  [ 0.        0.        0.502054  9.6     ]
#  [ 0.        0.        0.502054 24.3     ]
#  ...
#  [ 0.       83.        0.913909  8.3     ]
#  [ 0.       83.        0.913909  5.2     ]
#  [ 0.       84.        1.42659   8.      ]]
print(X_train.shape)  # (733, 4)

# Take into account a bias
x_train_bias = np.ones((X_train.shape[0], 1))
X_train = np.hstack((X_train, x_train_bias))
X_train_T = np.transpose(X_train)
x_test_bias = np.ones((X_test.shape[0], 1))
X_test = np.hstack((X_test, x_test_bias))
print(X_train)
# [[ 1.        0.        0.502054  9.7       1.      ]
#  [ 0.        0.        0.502054  9.6       1.      ]
#  [ 0.        0.        0.502054 24.3       1.      ]
#  ...
#  [ 0.       83.        0.913909  8.3       1.      ]
#  [ 0.       83.        0.913909  5.2       1.      ]
#  [ 0.       84.        1.42659   8.        1.      ]]

weights = (np.linalg.inv(X_train_T @ X_train) @ X_train_T) @ y_train
print(weights)
# [-6.40856082e-01  2.70005349e-03  2.58521725e+00 -2.08834593e-01
#   4.58390008e+00]

# Predict radon activities with the built linear regression model
test_predictions = X_test @ weights
# Predictions vs. True Values PLOT
f = set_style().set_general_style_parameters()
fig = plt.figure()
plt.scatter(test_y, test_predictions, marker='o', c='blue')
plt.plot([-5, 20], [-5, 20], color='black', ls='--')
plt.ylabel('Predictions [activity]', fontproperties=fm.FontProperties(fname=f))
plt.xlabel('True Values [activity]', fontproperties=fm.FontProperties(fname=f))
plt.title('Linear Regression with NumPy', fontproperties=fm.FontProperties(fname=f))
plt.ylim(-5, 20)
plt.xlim(-5, 20)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-2-2-1.svg', bbox_inches='tight')
