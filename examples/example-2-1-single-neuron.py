import keras
from tensorflow.python.layers import layers

model = keras.Sequential(
    [layers.Dense(1, input_shape=[...])]
)
