#!/usr/bin/python3
"""
-- Feed-Forward Neural Networks
---- Multiclass Classification with Feed-Forward Neural Networks
------ Weight Initialization
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

# # Weight initialization:
# # Normal distribution wit the mean of 0 and the standard deviation sigma:
# # - Xavier initialization for sigmoid;
# # - He initialization for ReLU.

mb_sizes = [200, 100, 50, 20, 10, 5]
EPOCHS = 1000

learning_time_dict = {}
for mb_size in mb_sizes:
    print("*" * 65)
    print("Mini-batch size:", mb_size)

    model = feed_forward.build_keras_model(
        num_inputs=784,
        structure="15-10",
        hidden_activation="relu",
        output_activation="softmax",
        initializer= tf.keras.initializers.HeNormal(),  # He initialization for ReLU
        optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    learning_history, learning_time = feed_forward.fit_model(
        model, data_train_norm, labels_train,
        batch_size=mb_size, num_epochs=EPOCHS
    )

    learning_time_dict[mb_size] = learning_time
    # Save history.
    # Install the module 'fastparquet' for working with the parquet binary format.
    learning_history.to_parquet(f"../histories/history-03-4-mb_size-{mb_size}.parquet")
# Save learning time.
# Install the module 'fastparquet' for working with the parquet binary format.
pd.DataFrame(
    data=learning_time_dict.items(),
    columns=['mini_batch_size', 'learning_time_in_minutes'],
).set_index('mini_batch_size').to_parquet("../histories/history-03-4-mb_size-learning_time.parquet")

# *****************************************************************
# Mini-batch size: 200
# ...
# Cost function at epoch of 0:
# Training MSE = 2.229443311691284
# Cost function at epoch of 1000:
# Training MSE = 0.36729559302330017
# Learning time = 6.94 minutes
# *****************************************************************
# Mini-batch size: 100
# ...
# Cost function at epoch of 0:
# Training MSE = 2.044503927230835
# Cost function at epoch of 1000:
# Training MSE = 0.3306983709335327
# Learning time = 12.92 minutes
# *****************************************************************
# Mini-batch size: 50
# ...
# Cost function at epoch of 0:
# Training MSE = 1.9720191955566406
# Cost function at epoch of 1000:
# Training MSE = 0.30878108739852905
# Learning time = 22.37 minutes
# *****************************************************************
# Mini-batch size: 20
# ...
# Cost function at epoch of 0:
# Training MSE = 1.4063093662261963
# Cost function at epoch of 1000:
# Training MSE = 0.2639326751232147
# Learning time = 45.69 minutes
# *****************************************************************
# Mini-batch size: 10
# ...
# Cost function at epoch of 0:
# Training MSE = 1.2854679822921753
# Cost function at epoch of 1000:
# Training MSE = 0.23946025967597961
# Learning time = 85.34 minutes
# *****************************************************************
# Mini-batch size: 5
# ...
# Cost function at epoch of 0:
# Training MSE = 0.9960035681724548
# Cost function at epoch of 1000:
# Training MSE = 0.23836740851402283
# Learning time = 167.16 minutes

# Load histories
learning_history_dict = {}
for mb_size in mb_sizes:
    learning_history = pd.read_parquet(f"../histories/history-03-4-mb_size-{mb_size}.parquet")
    learning_history_dict[mb_size] = learning_history
learning_time_dict = pd.read_parquet(
    "../histories/history-03-4-mb_size-learning_time.parquet"
)["learning_time_in_minutes"].to_dict()

# Plot results
fp = set_style().set_general_style_parameters()
plt.figure()
for mb_size in mb_sizes:
    history = learning_history_dict[mb_size]
    plt.plot(history['epoch'], history['loss'], label=f'Mini-batch size: {mb_size}')
plt.ylabel('Cost function $J$', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Epochs', fontproperties=fm.FontProperties(fname=fp))
plt.title('Different mini-batch sizes with the He initialization', fontproperties=fm.FontProperties(fname=fp))
plt.legend(loc='best')
plt.ylim(0.0, 1.0)
plt.xlim(0, EPOCHS)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-4-1.svg', bbox_inches='tight')
plt.close()

times = []
costs = []
labels = []
for mb_size in mb_sizes:
    history = learning_history_dict[mb_size]
    times.append(learning_time_dict[mb_size])
    costs.append(history["loss"].values[-1])
    labels.append(mb_size)

plt.figure()
plt.scatter(times, costs,  color='blue')
plt.ylabel(f'Cost function $J$ after {EPOCHS} epochs', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Time in minutes', fontproperties=fm.FontProperties(fname=fp))
plt.title('Different mini-batch sizes with the He initialization', fontproperties=fm.FontProperties(fname=fp))
for i, txt in enumerate(labels):
    plt.annotate(txt, (times[i] + 5, costs[i]))
plt.ylim(0.20, 0.40)
plt.xlim(0, 200)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-4-2.svg', bbox_inches='tight')
plt.close()
