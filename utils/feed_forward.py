# general libraries
import time
import pandas as pd

# tensorflow libraries
from tensorflow import keras
from keras import layers
import tensorflow as tf
# To install tensorflow_docs use:
# pip install --target=<side_packages directory> git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling


def build_keras_model(
        num_inputs,
        structure="4-3-2-1",
        hidden_activation="relu",
        output_activation=None,
        initializer="glorot_uniform",
        regularizer=None,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=("mse", ),
        model_name="feed-forward-model"
):
    """Build a customized feed-forward NN model using Keras"""
    # Build model
    inputs = keras.Input(shape=num_inputs)  # input layer
    # Add layers
    layer = inputs
    # customized number of layers and neurons per layer
    neurons_per_layer = [int(num_str) for num_str in structure.split("-")]
    for num_neurons in neurons_per_layer[:-1]:
        layer = layers.Dense(
            num_neurons,
            activation=hidden_activation,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )(layer)  # hidden layers
    # output layer
    outputs = layers.Dense(
        neurons_per_layer[-1],
        activation=output_activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )(layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def build_keras_sequential(
        num_inputs,
        structure="4-3-2-1",
        hidden_activation="relu",
        output_activation=None,
        initializer="glorot_uniform",
        regularizer=None,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=("mse",)
):
    """Build a customized feed-forward NN model using Keras"""
    # Create model
    model = keras.Sequential()
    neurons_per_layer = [int(num_str) for num_str in structure.split("-")]
    # Add first hidden layer and set input dimensions
    model.add(
        layers.Dense(
            neurons_per_layer[0],  # number of neurons
            input_dim=num_inputs,
            activation=hidden_activation,
            kernel_initializer=initializer
        )
    )
    # Add next hidden layers
    for num_neurons in neurons_per_layer[1:-1]:
        model.add(
            layers.Dense(
                num_neurons,  # number of neurons
                activation=hidden_activation,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer
            )
        )
    # Add output layer
    model.add(
        layers.Dense(
            neurons_per_layer[-1],  # number of neurons
            activation=output_activation,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )
    )
    # Compile model
    model.compile(
        loss=loss,  # loss function
        optimizer=optimizer,
        metrics=metrics  # accuracy metrics
    )
    return model


def fit_model(
        model,
        features_train,
        target_train,
        features_dev=None,
        target_dev=None,
        batch_size=50,
        num_epochs=1000
):
    """Fit a feed-forward NN model using Keras and validate it on the training and dev datasets"""
    start_time = time.time()
    result = model.fit(
        features_train, target_train,
        epochs=num_epochs, verbose=0,
        batch_size=batch_size,
        validation_data=(features_dev, target_dev) if (features_dev, target_dev) != (None, None) else None,
        callbacks=[tfdocs.modeling.EpochDots()]
    )
    learning_time = (time.time() - start_time) / 60
    learning_history = pd.DataFrame(result.history)
    learning_history['epoch'] = result.epoch
    print("\n")
    print('Cost function at epoch of 0:')
    print(f"Training MSE = {learning_history['loss'].values[0]}")
    if (features_dev, target_dev) != (None, None):
        print(f"Dev MSE = {learning_history['val_loss'].values[0]}")
    print(f'Cost function at epoch of {num_epochs}:')
    print(f"Training MSE = {learning_history['loss'].values[-1]}")
    if (features_dev, target_dev) != (None, None):
        print(f"Dev MSE = {learning_history['val_loss'].values[-1]}")
    print(f'Learning time = {learning_time:.2f} minutes')
    # model.summary()
    return model, learning_history, learning_time
