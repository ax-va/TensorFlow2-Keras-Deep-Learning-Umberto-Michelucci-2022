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
structures = ["2-10", "5-10", "10-10", "20-10", "50-10", "100-10", "200-10"]
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
# Structure: 2-10
# ...
# Cost function at epoch of 0:
# Training MSE = 2.071779251098633
# Cost function at epoch of 1000:
# Training MSE = 0.7749960422515869
# Learning time = 42.19 minutes
# *****************************************************************
# Structure: 5-10
# ...
# Cost function at epoch of 0:
# Training MSE = 2.0799577236175537
# Cost function at epoch of 1000:
# Training MSE = 0.40162768959999084
# Learning time = 42.13 minutes
# *****************************************************************
# Structure: 10-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.793076992034912
# Cost function at epoch of 1000:
# Training MSE = 0.30377396941185
# Learning time = 43.54 minutes
# *****************************************************************
# Structure: 20-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.3631904125213623
# Cost function at epoch of 1000:
# Training MSE = 0.2382696568965912
# Learning time = 45.23 minutes
# *****************************************************************
# Structure: 50-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.345389723777771
# Cost function at epoch of 1000:
# Training MSE = 0.14930659532546997
# Learning time = 53.24 minutes
# *****************************************************************
# Structure: 100-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.2164252996444702
# Cost function at epoch of 1000:
# Training MSE = 0.11624709516763687
# Learning time = 61.50 minutes
# *****************************************************************
# Structure: 200-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.2146618366241455
# Cost function at epoch of 1000:
# Training MSE = 0.08334076404571533
# Learning time = 68.83 minutes

learning_history_dict = {}
for structure in structures:
    learning_history = pd.read_parquet(f"../histories/history-03-6-mb_size-20-structure-{structure}.parquet")
    learning_history_dict[structure] = learning_history

fp = set_style().set_general_style_parameters()
fig = plt.figure()
for structure in structures:
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
plt.title('Different number of neurons per hidden layer', fontproperties=fm.FontProperties(fname=fp))
plt.legend(loc='best')
plt.ylim(0.0, 1.0)
plt.xlim(0, EPOCHS)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-6.svg', bbox_inches='tight')
plt.close()
