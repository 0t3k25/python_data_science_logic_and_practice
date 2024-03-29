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


image_size = (28, 28)
z_size = 20
mode_z = "uniform"  # 'uniform'か'normal'のどちらか
gen_hidden_layers = 1
gen_hidden_size = 100
disc_hidden_layers = 1
disc_hidden_size = 100
tf.random.set_seed(1)
gen_model = make_generator_network(
    num_hidden_layers=gen_hidden_layers,
    num_hidden_units=gen_hidden_size,
    num_output_units=np.prod(image_size),
)
gen_model.build(input_shape=(None, z_size))
gen_model.summary()
disc_model = make_discriminator_network(
    num_hidden_layers=disc_hidden_layers, num_hidden_units=disc_hidden_size
)
disc_model.build(input_shape=(None, np.prod(image_size)))
disc_model.summary()


mnist_bldr = tfds.builder("mnist")
mnist_bldr.download_and_prepare()
mnist = mnist_bldr.as_dataset(shuffle_files=False)


def preprocess(ex, mode="uniform"):
    image = ex["image"]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [-1])
    image = image * 2 - 1.0
    if mode == "uniform":
        input_z = tf.random.uniform(shape=(z_size,), minval=-1.0, maxval=1.0)
    elif mode == "normal":
        input_z = tf.random.normal(shape=(z_size,))
    return input_z, image


mnist_trainset = mnist["train"]
mnist_trainset = mnist_trainset.map(preprocess)
