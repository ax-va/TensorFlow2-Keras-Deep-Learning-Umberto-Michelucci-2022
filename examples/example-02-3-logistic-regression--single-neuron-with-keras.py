#!/usr/bin/python3
"""
-- Hands-on with a Single Neuron
---- Logistic Regression with a Single Neuron
Issue: Blood Cells Detection
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
read_bccd_dataset = importlib.import_module("ADL-Book-2nd-Ed.modules.read_bccd_dataset")
style_setting = importlib.import_module("ADL-Book-2nd-Ed.modules.style_setting")
read_data = read_bccd_dataset.read_data
set_style = style_setting.set_style

rd = read_data()
dataset = rd.preprocess_bccd_dataset()
dataset_reduced = dataset.loc[(dataset['cell_type'] == 'RBC') | (dataset['cell_type'] == 'WBC')]
# the location of the edges of the bounding box of the blood cell on a picture
bccd_features = dataset_reduced[['xmin', 'xmax', 'ymin', 'ymax']]
bccd_labels = dataset_reduced['cell_type']

num_observations = len(bccd_features)
print('Number of total samples: ', num_observations)
# Number of total samples:  4527
print(bccd_features.head())
#    xmin  xmax  ymin  ymax
# 0   260   491   177   376
# 1    78   184   336   435
# 2    63   169   237   336
# 3   214   320   362   461
# 4   414   506   352   445

print(bccd_labels.head())
# 0    WBC
# 1    RBC
# 2    RBC
# 3    RBC
# 4    RBC

# Split the dataset
np.random.seed(42) # reproducible random
rnd = np.random.rand(len(bccd_features)) < 0.8
# 80% of samples for training
train_x = bccd_features[rnd]  # training dataset (features)
train_y = bccd_labels[rnd]  # training dataset (labels)
# 20% of samples for test
test_x = bccd_features[~rnd]  # testing dataset (features)
test_y = bccd_labels[~rnd]  # testing dataset (labels)
print('The training dataset dimensions are: ', train_x.shape)
# The training dataset dimensions are:  (3631, 4)
print('The testing dataset dimensions are: ', test_x.shape)
# The testing dataset dimensions are:  (896, 4)

# Replace the types of blood cells, 'RBC' or 'WBC' strings, by 0 and 1 integers, respectively
train_y_bin = np.zeros(len(train_y))
train_y_bin[train_y == 'WBC'] = 1
test_y_bin = np.zeros(len(test_y))
test_y_bin[test_y == 'WBC'] = 1

# activation function := sigmoid function
# cost function := cross-entropy
# L_i (y_pred_i, y_train_i) = -(y_train_i * (log y_pred_i) + (1 - y_train_i) * log(1 - y_pred_i))


def build_model():
    neuron_model = keras.Sequential(
        [
            layers.Dense(1, input_shape=[len(train_x.columns)], activation='sigmoid')
        ]
    )
    neuron_model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=['binary_crossentropy', 'binary_accuracy']
    )
    return neuron_model


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

EPOCHS = 500
result = model.fit(
  train_x, train_y_bin,
  epochs=EPOCHS, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])
hist = pd.DataFrame(result.history)
hist['epoch'] = result.epoch
print(hist.tail())
#          loss  binary_crossentropy  binary_accuracy  epoch
# 495  0.068241             0.068241         0.983751    495
# 496  0.067719             0.067719         0.981272    496
# 497  0.070205             0.070205         0.982374    497
# 498  0.068242             0.068242         0.981548    498
# 499  0.068884             0.068884         0.982374    499

fp = set_style().set_general_style_parameters()
plt.figure()
plt.plot(hist['epoch'], hist['binary_crossentropy'], color='blue')
plt.ylabel('Cost Function (cross-entropy)', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Number of Iterations', fontproperties=fm.FontProperties(fname=fp))
plt.ylim(-5, 20)
plt.xlim(0, 500)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-02-3.svg', bbox_inches='tight')

# Predict cell type with the built logistic regression model
test_predictions = model.predict(test_x).flatten()

# The following lines compute the accuracy on the test set
test_predictions_bin = test_predictions > 0.5
true_positive = np.sum((test_predictions_bin == 1) & (test_y_bin == 1))
true_negative = np.sum((test_predictions_bin == 0) & (test_y_bin == 0))
accuracy_test = (true_positive + true_negative) / len(test_y)
print('The accuracy on the test set is equal to: ' + str(int(accuracy_test * 100)) + '%')
# The accuracy on the test set is equal to: 97%

