#!/usr/bin/python3
"""
-- Feed-Forward Neural Networks
---- Multiclass Classification with Feed-Forward Neural Networks
------ Adding Many Layers Efficiently

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


def build_and_train_model_with_layers(num_neurons, num_layers):
    """ Build model """
    inputs = keras.Input(shape=784)  # input layer
    # customized number of layers and neurons per layer
    layer = inputs
    for i in range(num_layers):
        layer = layers.Dense(num_neurons, activation='relu')(layer)  # hidden layers
    # output layer
    outputs = layers.Dense(10, activation='softmax')(layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name='model')
    # Set optimizer and loss
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.0001),
        metrics=['categorical_accuracy']
    )
    # Train model
    result = model.fit(
        data_train_norm, labels_train,
        epochs=1000, verbose=0,
        batch_size=20,
        callbacks=[tfdocs.modeling.EpochDots()]
    )
    # Save performances
    df_history = pd.DataFrame(result.history)
    df_history['epoch'] = result.epoch
    return df_history, model


history_dict = {}
for num_n, num_l in [(10, 1), (10, 2), (10, 3), (10, 4), (100, 4)]:
    df_history, model = build_and_train_model_with_layers(num_neurons=num_n, num_layers=num_l)
    history_dict[(num_n, num_l)] = df_history
    df_history.to_csv(f"../histories/history-03-5-num_n-{num_n}-num_l-{num_l}.csv")


# history_dict = {}
# for num_n, num_l in [(10, 1), (10, 2), (10, 3), (10, 4), (100, 4)]:
#     df_history = pd.read_csv(f"../histories/history-03-5-num_n-{num_n}-num_l-{num_l}.csv")
#     history_dict[(num_n, num_l)] = df_history


fp = set_style().set_general_style_parameters()
fig = plt.figure()
ax = fig.add_subplot(111)
for params, color in [((10, 1), "black"), ((10, 2), "blue"), ((10, 3), "red"), ((10, 4), "green"), ((100, 4), "magenta")]:
    history = history_dict[params]
    if params[1] == 1:
        text_num_layers = f'{params[1]} layer'
        text_num_neurons = f'{params[0]} neurons'
    else:
        text_num_layers = f'{params[1]} layers'
        text_num_neurons = f'{params[0]} neurons pro layer'
    ax.plot(history['epoch'], history['loss'], color=color, label=f'{text_num_layers}, {text_num_neurons}')
plt.ylabel('Cost function $J$', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('Epochs', fontproperties=fm.FontProperties(fname=fp))
plt.legend(loc='best')
plt.ylim(0.0, 1.0)
plt.xlim(0, 1000)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-5.svg', bbox_inches='tight')

# The model with 4 layers, 100 neurons pro layer, is overfitted
