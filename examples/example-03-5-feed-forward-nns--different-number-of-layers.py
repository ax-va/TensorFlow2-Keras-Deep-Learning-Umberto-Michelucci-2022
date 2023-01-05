#!/usr/bin/python3
"""
-- Feed-Forward Neural Networks
---- Multiclass Classification with Feed-Forward Neural Networks
------ Adding Many Layers Efficiently
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
structures = ["10-10", "10-10-10", "10-10-10-10", "10-10-10-10-10", "100-100-100-100-10"]
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

    learning_history.to_parquet(f"../histories/history-03-5-mb_size-{MB_SIZE}-structure-{structure}.parquet")
    # Save the trained model
    model.save(f"../models/model-03-5-structure-{structure}")
# *****************************************************************
# Structure: 10-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.4168392419815063
# Cost function at epoch of 1000:
# Training MSE = 0.3320516049861908
# Learning time = 43.77 minutes
# *****************************************************************
# Structure: 10-10-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.8717634677886963
# Cost function at epoch of 1000:
# Training MSE = 0.2803069055080414
# Learning time = 46.32 minutes
# *****************************************************************
# Structure: 10-10-10-10
# ...
# Cost function at epoch of 0:
# Training MSE = 2.0893638134002686
# Cost function at epoch of 1000:
# Training MSE = 0.28231358528137207
# Learning time = 48.00 minutes
# *****************************************************************
# Structure: 10-10-10-10-10
# ...
# Cost function at epoch of 0:
# Training MSE = 2.0132534503936768
# Cost function at epoch of 1000:
# Training MSE = 0.2618364095687866
# Learning time = 49.71 minutes
# *****************************************************************
# Structure: 100-100-100-100-100-10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.7035552263259888
# Cost function at epoch of 1000:
# Training MSE = 0.00025658542290329933
# Learning time = 73.97 minutes

learning_history_dict = {}
for structure in structures:
    learning_history = pd.read_parquet(f"../histories/history-03-5-mb_size-20-structure-{structure}.parquet")
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
plt.ylabel('Cost function $J$', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Epochs', fontproperties=fm.FontProperties(fname=fp))
plt.title('Different number of layers', fontproperties=fm.FontProperties(fname=fp))
plt.legend(loc='best')
plt.ylim(0.0, 1.0)
plt.xlim(0, 1000)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-5.svg', bbox_inches='tight')
plt.close()

# The model with 4 hidden layers and 100 neurons per layer is over fitted
