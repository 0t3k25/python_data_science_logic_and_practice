# gpuの確認
import tensorflow as tf

print(tf.__version__)

print("GPU Availabel:", tf.config.list_physical_devices("GPU"))

if tf.config.list_physical_devices("GPU"):
    device_name = tf.test.gpu_device_name()
else:
    device_name = "/CPU:0"

print(device_name)

import tensorflow as tfds
import nunpy as np


# 生成器ネットワークの関数を定義
def make_generator_network(
    num_hidden_layers=1, num_hidden_units=100, num_output_units=784
):
    model = tf.keras.Sequential()
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(units=num_hidden_units, use_bias=False))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dense(units=num_output_units, activation="tanh"))
    return model


# 識別器ネットワークの関数を定義
def make_discriminator_network(
    num_hidden_layers=1, num_hidden_units=100, num_output_units=1
):
    model = tf.keras.Sequential()
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(units=num_hidden_units))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Dense(units=num_output_units, activation=None))
    return model
