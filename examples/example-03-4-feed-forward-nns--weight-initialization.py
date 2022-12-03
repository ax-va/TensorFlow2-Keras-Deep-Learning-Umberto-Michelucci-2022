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
# general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time

# tensorflow libraries
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras import layers
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling

import importlib
set_style = importlib.import_module("ADL-Book-2nd-Ed.modules.style_setting").set_style

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

# Weight initialization:
# Normal distribution wit the mean of 0 and the standard deviation sigma:
# - Xavier initialization for sigmoid;
# - He initialization for ReLU.


# ReLU activation function and He initialization:
# Keras implementation:
def build_model(optimizer):
    # Create model
    feed_forward_model = keras.Sequential()
    # Use the He initialization for ReLU
    initializer = tf.keras.initializers.HeNormal()
    # Add first hidden layer and set input dimensions
    feed_forward_model.add(layers.Dense(
        15,  # number of neurons
        input_dim=784,
        activation='relu',
        kernel_initializer=initializer))
    # Add output layer
    feed_forward_model.add(layers.Dense(
        10,  # number of neurons
        activation='softmax'))
    # Compile model
    feed_forward_model.compile(
        loss='categorical_crossentropy',  # loss function
        optimizer=optimizer,
        metrics=['categorical_accuracy']  # accuracy metrics
    )
    return feed_forward_model


def execute_mini_batch_gradient_descent(mini_batch_size):
    # Build model
    model_mbgd = build_model(tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.0001))
    # Set number of epochs
    EPOCHS = 100
    # Train model
    result = model_mbgd.fit(
        data_train_norm, labels_train,
        epochs=EPOCHS, verbose=0,
        batch_size=mini_batch_size,
        callbacks=[tfdocs.modeling.EpochDots()]
    )
    # Save performances
    hist_mbgd = pd.DataFrame(result.history)
    hist_mbgd['epoch'] = result.epoch
    return hist_mbgd


mb_sizes = [200, 100, 50, 20, 10, 5]

# history_dict = {}
# execution_time_dict = {}
# for mb_size in mb_sizes:  # [200, 100, 50, 20, 10, 5]
#     print("*" * 65)
#     print("Mini-batch size:", mb_size)
#     start = time.time()
#     df_history = execute_mini_batch_gradient_descent(mb_size)
#     history_dict[mb_size] = df_history
#     execution_time = (time.time() - start) / 60
#     execution_time_dict[mb_size] = execution_time
#     print(f"\nExecution time in minutes: {execution_time:.2f}")
#     df_history.to_csv(f"../histories/history-03-4-mb_size-{mb_size}.csv")
#     pd.DataFrame(
#         data=execution_time_dict.items(),
#         columns=['mini_batch_size', 'exec_time_in_minutes'],
#     ).set_index('mini_batch_size').to_csv("../histories/history-03-4-mb_size-exec_time.csv")

history_dict = {}
for mb_size in mb_sizes:
    df_history = pd.read_csv(f"../histories/history-03-4-mb_size-{mb_size}.csv")
    history_dict[mb_size] = df_history
execution_time_dict = pd.read_csv(
    "../histories/history-03-4-mb_size-exec_time.csv",
    index_col="mini_batch_size"
)["exec_time_in_minutes"].to_dict()

fp = set_style().set_general_style_parameters()
fig = plt.figure()
ax = fig.add_subplot(111)
for mb_size, color in zip(mb_sizes, ["black", "blue", "red", "green", "magenta", "yellow"]):
    history = history_dict[mb_size]
    ax.plot(history['epoch'], history['loss'], color=color, label=f'Mini-batch size: {mb_size}')
plt.ylabel('Cost function $J$', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Epochs', fontproperties=fm.FontProperties(fname=fp))
plt.title('ReLU with He init', fontproperties=fm.FontProperties(fname=fp))
plt.legend(loc='best')
plt.ylim(0.3, 1.0)
plt.xlim(0, 100)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-4-1.svg', bbox_inches='tight')

times = []
costs = []
labels = []
for mb_size in [200, 100, 50, 20, 10, 5]:
    history = history_dict[mb_size]
    times.append(execution_time_dict[mb_size])
    costs.append(history["loss"].values[-1])
    labels.append(mb_size)

fig = plt.figure()
plt.scatter(times, costs,  color='blue')
plt.ylabel('Cost function $J$ after 100 epochs', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Time (minutes)', fontproperties=fm.FontProperties(fname=fp))
plt.title('ReLU with He init', fontproperties=fm.FontProperties(fname=fp))
for i, txt in enumerate(labels):
    plt.annotate(txt, (times[i] + 1.0, costs[i]))
plt.ylim(0.30, 0.60)
plt.xlim(0, 20)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-4-2.svg', bbox_inches='tight')
