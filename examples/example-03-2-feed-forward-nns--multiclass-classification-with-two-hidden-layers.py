#!/usr/bin/python3
"""
-- Feed-Forward Neural Networks
---- Multiclass Classification with Feed-Forward Neural Networks
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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import *

# tensorflow libraries
from keras.datasets import fashion_mnist
import tensorflow as tf

# my module
import utils.feed_forward as feed_forward

# train_x, test_x contain pixels.
# train_y, test_y contain digits corresponding to Zalando clothes.
# Both are numpy arrays.
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
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
print(labels_train)
# [[0. 0. 0. ... 0. 0. 1.]
#  [1. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
labels_test = np.zeros((10000, 10))
labels_test[np.arange(10000), test_y] = 1
print(labels_test)
# [[0. 0. 0. ... 0. 0. 1.]
#  [0. 0. 1. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 1. 0.]
#  [0. 1. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]

idx = 5
plt.imshow(data_train_norm[idx, :].reshape(28, 28), cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis("on")
plt.title(label_dict[train_y[idx]])
# plt.show()
plt.savefig('../figures/figure-03-2-1.svg', bbox_inches='tight')
plt.close()


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
plt.close()


EPOCHS = 100
gd_dict = {  # (batch_size, momentum, learning_rate)
    "standard": (data_train_norm.shape[0], 0.0, 0.01),  # (60000, 0.0, 0.01)
    "stochastic": (1, 0.9, 0.0001),
    "mini-batch": (20, 0.9, 0.01)
}
for gd_name, hyper_params in gd_dict.items():
    batch_size, momentum, learning_rate = hyper_params
    print("Type of gradient descent:", gd_name)

    model = feed_forward.build_keras_model(
        num_inputs=784,
        structure="15-10",
        hidden_activation="relu",
        output_activation="softmax",
        optimizer=tf.keras.optimizers.SGD(momentum=momentum, learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    learning_history, learning_time = feed_forward.fit_model(
        model, data_train_norm, labels_train,
        batch_size=batch_size, num_epochs=EPOCHS
    )

    # Calculate accuracy on the dev dataset
    test_loss, test_accuracy = model.evaluate(data_test_norm, labels_test, verbose=0)
    print(f"The accuracy on the dev set is equal to: {int(test_accuracy * 100)}%")
    print("*" * 65)

# FIRST RUN

# Type of gradient descent: standard
# ...
# Cost function at epoch of 0:
# Training MSE = 2.345994234085083
# Cost function at epoch of 100:
# Training MSE = 1.544346570968628
# Learning time = 0.16 minutes
# The accuracy on the dev set is equal to: 55%
# *****************************************************************
# Type of gradient descent: stochastic
# ...
# Cost function at epoch of 0:
# Training MSE = 0.6423079371452332
# Cost function at epoch of 100:
# Training MSE = 0.27205148339271545
# Learning time = 83.68 minutes
# The accuracy on the dev set is equal to: 86%
# *****************************************************************
# Type of gradient descent: mini-batch
# ...
# Cost function at epoch of 0:
# Training MSE = 0.560768187046051
# Cost function at epoch of 100:
# Training MSE = 0.264372318983078
# Learning time = 5.01 minutes
# The accuracy on the dev set is equal to: 86%
# *****************************************************************

# SECOND RUN

# Type of gradient descent: standard
# ...
# Cost function at epoch of 0:
# Training MSE = 2.382349967956543
# Cost function at epoch of 100:
# Training MSE = 1.8391923904418945
# Learning time = 0.15 minutes
# The accuracy on the dev set is equal to: 36%
# *****************************************************************
# Type of gradient descent: stochastic
# ...
# Cost function at epoch of 0:
# Training MSE = 0.6920946836471558
# Cost function at epoch of 100:
# Training MSE = 0.2798094153404236
# Learning time = 81.14 minutes
# The accuracy on the dev set is equal to: 85%
# *****************************************************************
# Type of gradient descent: mini-batch
# ...
# Cost function at epoch of 0:
# Training MSE = 0.58244389295578
# Cost function at epoch of 100:
# Training MSE = 0.2661474347114563
# Learning time = 4.57 minutes
# The accuracy on the dev set is equal to: 85%
# *****************************************************************

# THIRD RUN

# Type of gradient descent: standard
# ...
# Cost function at epoch of 0:
# Training MSE = 2.5165092945098877
# Cost function at epoch of 100:
# Training MSE = 1.6146700382232666
# Learning time = 0.17 minutes
# The accuracy on the dev set is equal to: 58%
# *****************************************************************
# Type of gradient descent: stochastic
# ...
# Cost function at epoch of 0:
# Training MSE = 0.6294620037078857
# Cost function at epoch of 100:
# Training MSE = 0.27525442838668823
# Learning time = 79.17 minutes
# The accuracy on the dev set is equal to: 84%
# *****************************************************************
# Type of gradient descent: mini-batch
# ...
# Cost function at epoch of 0:
# Training MSE = 0.5544220209121704
# Cost function at epoch of 100:
# Training MSE = 0.27743008732795715
# Learning time = 4.81 minutes
# The accuracy on the dev set is equal to: 83%
# *****************************************************************
