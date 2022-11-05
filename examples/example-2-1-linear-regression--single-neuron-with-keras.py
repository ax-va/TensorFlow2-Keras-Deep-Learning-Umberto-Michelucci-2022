#!/usr/bin/python3
"""
-- Hands-on with a Single Neuron
---- Linear Regression Model with Keras
"""
# general libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers
# To install tensorflow_docs use:
# pip install git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling

# Ignore warnings
import warnings
warnings.simplefilter('ignore')

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

num_counties = len(county_name)
num_observations = len(radon_features)
print('Number of counties included in the dataset: ', num_counties)
# Number of counties included in the dataset:  85
print('Number of total samples: ', num_observations)
# Number of total samples:  919
print(radon_features)
#    floor  county  log_uranium_ppm  pcterr
# 0        1       0         0.502054     9.7
# 1        0       0         0.502054    14.5
# 2        0       0         0.502054     9.6
# 3        0       0         0.502054    24.3
# 4        0       1         0.428565    13.8
# ..     ...     ...              ...     ...
# 922      0      83         0.913909     4.5
# 923      0      83         0.913909     8.3
# 924      0      83         0.913909     5.2
# 925      0      84         1.426590     9.6
# 926      0      84         1.426590     8.0
#
# [919 rows x 4 columns]

# The radon activity label (the target variable) is the measured radon concentration in pCi/L
print(radon_labels)
# 0      2.2
# 1      2.2
# 2      2.9
# 3      1.0
# 4      3.1
#       ...
# 922    6.4
# 923    4.5
# 924    5.0
# 925    3.7
# 926    2.9
# Name: radon, Length: 919, dtype: float64

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


def build_model():
    single_neuron = keras.Sequential(
        [
            layers.Dense(
                1,  # one neuron
                input_shape=[len(train_x.columns)]  # number of input features
            )
        ]
    )
    single_neuron.compile(
        loss='mse',  # Mean Squared Error (MSE) function as the loss function
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),  # RMSProp optimizer like gradient descent
        metrics=['mse']
    )
    return single_neuron


model = build_model()
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 1)                 5
#
# =================================================================
# Total params: 5
# Trainable params: 5
# Non-trainable params: 0
# _________________________________________________________________

#  5 parameters to be trained: the weights
#  associated with the 4 features, plus the bias

EPOCHS = 1000
result = model.fit(
    train_x, train_y,
    epochs=EPOCHS,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

hist = pd.DataFrame(result.history)
hist['epoch'] = result.epoch
print(hist.tail())
#           loss        mse  epoch
# 995  15.961001  15.961001    995
# 996  15.957549  15.957549    996
# 997  15.951381  15.951381    997
# 998  15.963511  15.963511    998
# 999  15.976029  15.976029    999

f = set_style().set_general_style_parameters()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(hist['epoch'], hist['mse'], color='blue')
plt.ylabel('Cost Function (MSE)', fontproperties=fm.FontProperties(fname=f))
plt.xlabel('Number of Iterations', fontproperties=fm.FontProperties(fname=f))
plt.ylim(0, 50)
plt.xlim(0, 1000)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-2-1-1.svg', bbox_inches='tight')

# Return a numpy list of weights
weights = model.get_weights()
# Weights consisting of a 4-weights vector and a bias scalar
print(weights)
# [
#   array([[-6.5743500e-01],
#        [ 1.4790290e-03],
#        [ 2.7411327e+00],
#        [-2.0764650e-01]], dtype=float32),
#   array([4.3961654], dtype=float32)
#   ]

# Predict radon activities with the built linear regression model
test_predictions = model.predict(test_x).flatten()
# Predictions vs. True Values PLOT
fig = plt.figure()
plt.scatter(test_y, test_predictions, marker='o', c='blue')
plt.plot([-5, 20], [-5, 20], color='black', ls='--')
plt.ylabel('Predictions [activity]', fontproperties=fm.FontProperties(fname=f))
plt.xlabel('True Values [activity]', fontproperties=fm.FontProperties(fname=f))
plt.title('Linear Regression with One Neuron', fontproperties=fm.FontProperties(fname=f))
plt.ylim(-5, 20)
plt.xlim(-5, 20)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-2-1-2.svg', bbox_inches='tight')
