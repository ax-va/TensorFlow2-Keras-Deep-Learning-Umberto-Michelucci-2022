#!/usr/bin/python3
"""
-- Feed-Forward Neural Networks
---- Multiclass Classification with Feed-Forward Neural Networks
------ Comparing Different Networks
 Issue: Clothes images recognition

The dataset has 785 columns, where the first column is the class label
(an integer going from 0 to 9) and the remaining 784 contain the pixel
gray value of the image (you can calculate that 28x28=784).

{0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
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
from keras.datasets import fashion_mnist
import tensorflow as tf

import importlib
set_style = importlib.import_module("ADL-Book-2nd-Ed.modules.style_setting").set_style

# my module
import utils.feed_forward as feed_forward

# train_x, test_x contain pixels.
# train_y, test_y contain digits corresponding to Zalando clothes.
# Both are numpy arrays.
((train_x, train_y), (test_x, test_y)) = fashion_mnist.load_data()

label_dict = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

data_train = train_x.reshape(60000, 784)
data_test = test_x.reshape(10000, 784)

# Normalize the input so that it has only values between 0 and 1
data_train_norm = np.array(data_train/255.0)
data_test_norm = np.array(data_test/255.0)

# Hot-one encoding for labels:
labels_train = np.zeros((60000, 10))
labels_train[np.arange(60000), train_y] = 1
labels_test = np.zeros((10000, 10))
labels_test[np.arange(10000), test_y] = 1

# The last number is the number of outputs
structures = ["1-10", "2-10", "5-10", "10-10", "20-10", "50-10", "100-10"]
EPOCHS = 1000
MB_SIZE = 20

for structure in structures:
    print("*" * 65)
    print(f"Structure: {structure}")
    model = feed_forward.build_keras_model(
        num_inputs=784,
        structure=structure,
        hidden_activation="relu",
        output_activation="softmax",
        optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    learning_history, learning_time = feed_forward.fit_model(
        model, data_train_norm, labels_train,
        batch_size=MB_SIZE, num_epochs=EPOCHS
    )

    learning_history.to_parquet(f"../histories/history-03-6-mb_size-{MB_SIZE}-structure-{structure}.parquet")
    # Save the trained model
    model.save(f"../models/model-03-6-structure-{structure}")
# *****************************************************************
# Structure: 1-10
# ...
# Cost function at epoch of 0:
# Training MSE = 2.244680166244507
# Cost function at epoch of 1000:
# Training MSE = 1.3493092060089111
# Learning time = 46.30 minutes
# *****************************************************************
# Structure: 2-10
# ...
# Cost function at epoch of 0:
# Training MSE = 2.053353786468506
# Cost function at epoch of 1000:
# Training MSE = 0.6999201774597168
# Learning time = 42.89 minutes
# *****************************************************************
# Structure: 5-10
# ...
# Cost function at epoch of 0:
# Training MSE = 2.100836992263794
# Cost function at epoch of 1000:
# Training MSE = 0.403066486120224
# Learning time = 42.72 minutes
# *****************************************************************
# Structure: 10-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.648625135421753
# Cost function at epoch of 1000:
# Training MSE = 0.3379696011543274
# Learning time = 43.39 minutes
# *****************************************************************
# Structure: 20-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.447582721710205
# Cost function at epoch of 1000:
# Training MSE = 0.23862901329994202
# Learning time = 44.91 minutes
# *****************************************************************
# Structure: 50-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.3464939594268799
# Cost function at epoch of 1000:
# Training MSE = 0.16387705504894257
# Learning time = 58.55 minutes
# *****************************************************************
# Structure: 100-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.194445013999939
# Cost function at epoch of 1000:
# Training MSE = 0.1119411364197731
# Learning time = 62.40 minutes

learning_history_dict = {}
for structure in structures:
    learning_history = pd.read_parquet(f"../histories/history-03-6-mb_size-20-structure-{structure}.parquet")
    learning_history_dict[structure] = learning_history

fp = set_style().set_general_style_parameters()
fig = plt.figure()
for structure in structures[1:]:
    learning_history = learning_history_dict[structure]
    neurons_per_layer = [int(num_str) for num_str in structure.split("-")]
    num_layers = len(neurons_per_layer)
    num_hidden_layers = num_layers - 1
    num_neurons = neurons_per_layer[0]
    if num_hidden_layers == 1:
        text_num_layers = f'{num_hidden_layers} hidden layer'
        text_num_neurons = f'{num_neurons} neuron' if neurons_per_layer[0] == 1 else f'{num_neurons} neurons'
    else:
        text_num_layers = f'{num_hidden_layers} hidden layers'
        text_num_neurons = f'{num_neurons} neuron' if neurons_per_layer[0] == 1 else f'{num_neurons} neurons'
    plt.plot(learning_history['epoch'], learning_history['loss'], label=f'{text_num_layers}, {text_num_neurons}')
plt.ylabel('Cost function $J$', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Epochs', fontproperties=fm.FontProperties(fname=fp))
plt.title('Different number of neurons', fontproperties=fm.FontProperties(fname=fp))
plt.legend(loc='best')
plt.ylim(0.0, 1.0)
plt.xlim(0, 1000)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-6-2.svg', bbox_inches='tight')
plt.close()
