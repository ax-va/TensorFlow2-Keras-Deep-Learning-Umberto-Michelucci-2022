#!/usr/bin/python3
"""
-- Feed-Forward Neural Networks
---- Multiclass Classification with Feed-Forward Neural Networks

The dataset has 785 columns, where the first column is the class label
(an integer going from 0 to 9) and the remaining 784 contain the pixel
gray value of the image (you can calculate that 28x28=784).

{0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
"""
# general libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from random import *
import time

# tensorflow libraries
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras import layers
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling

# train_x, test_x contain pixels.
# train_y, test_y contain digits corresponding to Zalando clothes.
# Both are numpy arrays.
((train_x, train_y), (test_x, test_y)) = fashion_mnist.load_data()
print('Dimensions of the training dataset: ', train_x.shape)
# Dimensions of the training dataset:  (60000, 28, 28)
print('Dimensions of the test dataset: ', test_x.shape)
# Dimensions of the test dataset:  (10000, 28, 28)
print('Dimensions of the training labels: ', train_y.shape)
# Dimensions of the training labels:  (60000,)
print('Dimensions of the test labels: ', test_y.shape)
# Dimensions of the test labels:  (10000,)

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

idx = 5
plt.imshow(data_train_norm[idx, :].reshape(28, 28), cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis("on")
plt.title(label_dict[train_y[idx]])
# plt.show()
plt.savefig('../figures/figure-03-2-1.svg', bbox_inches='tight')


def get_random_element_with_label(data, lbls, lbl):
    """Returns one numpy array (one column) with an example of a chosen label."""
    tmp = lbls == lbl
    subset = data[tmp.flatten(), :]
    return subset[randint(1, subset.shape[1]), :]


# The following code create a numpy array where in column 0 you will find
# an example of label 0, in column 1 of label 1, and so on.
labels_overview = np.empty([784, 10])
for i in range(0, 10):
    col = get_random_element_with_label(data_train_norm, train_y, i)
    labels_overview[:, i] = col


f = plt.figure(figsize=(15, 15))
count = 1
for i in range(0, 10):
    plt.subplot(5, 2, count)
    count += 1
    plt.subplots_adjust(hspace=0.5)
    plt.title(label_dict[i])
    digit_image = labels_overview[:, i].reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
plt.savefig('../figures/figure-03-2-2.svg', bbox_inches='tight')


# Keras implementation:
def build_model(opt):
    # Create model
    neuron_model = keras.Sequential()
    # Add first hidden layer and set input dimensions
    neuron_model.add(layers.Dense(
        15,  # number of neurons
        input_dim=784,
        activation='relu'))
    # Add output layer
    neuron_model.add(layers.Dense(
        10,  # number of neurons
        activation='softmax'))
    # Compile model
    neuron_model.compile(
        loss='categorical_crossentropy',  # loss function
        optimizer=opt,
        metrics=['categorical_accuracy']  # accuracy metrics
    )
    return neuron_model


EPOCHS = 100
gd_dict = {  # (batch_size, momentum, learning_rate)
    # "standard gradient descent": (data_train_norm.shape[0], 0.0, 0.01),  # (60000, 0.0, 0.01)
    # "stochastic gradient descent": (1, 0.9, 0.0001),
    "mini-batch gradient descent": (20, 0.9, 0.01)
}
for gd_name, hyperparams in gd_dict.items():
    batch_size, momentum, learning_rate = hyperparams
    print("Type of gradient descent:", gd_name)
    start_time = time.time()
    model = build_model(
        tf.keras.optimizers.SGD(momentum=momentum, learning_rate=learning_rate)
    )
    result = model.fit(
      data_train_norm, labels_train,
      epochs=EPOCHS, verbose=0,
      batch_size=batch_size,
      callbacks=[tfdocs.modeling.EpochDots()]
    )
    training_time = (time.time() - start_time) / 60
    print("\n*********************************")
    model.summary()
    hist = pd.DataFrame(result.history)
    hist['epoch'] = result.epoch
    print(hist.tail())
    print(f'This took {training_time:.2f} minutes')

    # Calculate accuracy on the dev dataset
    test_loss, test_accuracy = model.evaluate(data_test_norm, labels_test, verbose=0)
    print(f"The accuracy on the dev set is equal to: {int(test_accuracy*100)}%")
    print("*********************************")

# Type of gradient descent: standard gradient descent
# ...
# *********************************
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 15)                11775
#
#  dense_1 (Dense)             (None, 10)                160
#
# =================================================================
# Total params: 11,935
# Trainable params: 11,935
# Non-trainable params: 0
# _________________________________________________________________
#         loss  categorical_accuracy  epoch
# 95  1.666536              0.407100     95
# 96  1.662127              0.409333     96
# 97  1.657738              0.411350     97
# 98  1.653368              0.413550     98
# 99  1.649016              0.415300     99
# This took 0.15 minutes
# The accuracy on the dev set is equal to: 41%
# *********************************
# Type of gradient descent: stochastic gradient descent
# ...
# *********************************
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense_2 (Dense)             (None, 15)                11775
#
#  dense_3 (Dense)             (None, 10)                160
#
# =================================================================
# Total params: 11,935
# Trainable params: 11,935
# Non-trainable params: 0
# _________________________________________________________________
#         loss  categorical_accuracy  epoch
# 95  0.280242              0.898583     95
# 96  0.279999              0.899350     96
# 97  0.279683              0.898167     97
# 98  0.279734              0.898333     98
# 99  0.279381              0.899233     99
# This took 82.29 minutes
# The accuracy on the dev set is equal to: 86%
# *********************************
# Type of gradient descent: mini-batch
# ...
# *********************************
# Model: "sequential_2"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense_4 (Dense)             (None, 15)                11775
#
#  dense_5 (Dense)             (None, 10)                160
#
# =================================================================
# Total params: 11,935
# Trainable params: 11,935
# Non-trainable params: 0
# _________________________________________________________________
#         loss  categorical_accuracy  epoch
# 95  0.294306              0.890850     95
# 96  0.294935              0.892567     96
# 97  0.295630              0.890567     97
# 98  0.294350              0.891633     98
# 99  0.292305              0.891050     99
# This took 4.67 minutes
# The accuracy on the dev set is equal to: 84%
# *********************************
