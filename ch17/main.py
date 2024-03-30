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

mnist_trainset = mnist_trainset.batch(32, drop_remainder=True)
input_z, input_real = next(iter(mnist_trainset))
print("input-z -- shape:", input_z.shape)
print("input-real -- shape:", input_real.shape)

# fake画像と本物の画像
g_output = gen_model(input_z)
print("Output of G -- shape:", g_output.shape)
d_logits_real = disc_model(input_real)
d_logits_fake = disc_model(g_output)
print("Disc.(real) -- shape:", d_logits_real.shape)
print("Disc.(fake) -- shape:", d_logits_fake.shape)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# 生成器ネットワークの損失関数
g_labels_real = tf.ones_like(d_logits_fake)
g_loss = loss_fn(y_true=g_labels_real, y_pred=d_logits_fake)
print("Generator Loss: {:.4f}".format(g_loss))

# 識別器ネットワークの損失関数
d_labels_real = tf.ones_like(d_logits_real)
d_labels_fake = tf.zeros_like(d_logits_fake)
d_loss_real = loss_fn(y_true=d_labels_real, y_pred=d_logits_real)
d_loss_fake = loss_fn(y_true=d_labels_fake, y_pred=d_logits_fake)
print(
    "Discriminator Losses: Real {:.4f} Fake {:.4f}".format(
        d_loss_real.numpy(), d_loss_fake.numpy()
    )
)

import time

num_epochs = 100
batch_size = 64
image_size = (28, 28)
z_size = 20
mode_z = "uniform"
gen_hidden_layers = 1
gen_hidden_size = 100
disc_hidden_layers = 1
disc_hidden_size = 100
tf.random.set_seed(1)
np.random.seed(1)

if mode_z == "uniform":
    fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)
elif mode_z == "normal":
    fixed_z = tf.random.normal(shape=(batch_size, z_size))


def create_samples(g_model, input_z):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0


# データセットの準備
mnist_trainset = mnist["train"]
mnist_trainset = mnist_trainset.map(lambda ex: preprocess(ex, mode=mode_z))
mnist_trainset = mnist_trainset.shuffle(10000)
mnist_trainset = mnist_trainset.batch(batch_size, drop_remainder=True)
# モデルの準備
with tf.device(device_name):
    gen_model = make_generator_network(
        num_hidden_layers=gen_hidden_layers,
        num_hidden_units=gen_hidden_size,
        num_output_units=np.prod(image_size),
    )
    gen_model.build(input_shape=(None, z_size))
    disc_model = make_discriminator_network(
        num_hidden_layers=disc_hidden_layers, num_hidden_units=disc_hidden_size
    )
    disc_model.build(input_shape=(None, np.prod(image_size)))

# 損失関数とオプティマイザ
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = tf.keras.optimizers.Adam()
d_optimizer = tf.keras.optimizers.Adam()

all_losses = []
all_d_vals = []
epoch_samples = []

start_time = time.time()
for epoch in range(1, num_epochs + 1):
    epoch_lossses, epoch_d_vals = [], []
    for i, (input_z, input_real) in enumerate(mnist_trainset):
        # 生成器ネットワークの損失関数を計算
        with tf.GradientTape() as g_tape:
            g_output = gen_model(input_z)
            d_logits_fake = disc_model(g_output, training=True)
            labels_real = tf.ones_like(d_logits_fake)
            g_loss = loss_fn(y_true=labels_real, y_pred=d_logits_fake)

        # g_lossの勾配を計算
        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
        # 最適化・勾配を適用
        g_optimizer.apply_gradients(
            grads_and_vars=zip(g_grads, gen_model.trainable_variables)
        )
        # 識別器ネットワークの損失関数を計算
        with tf.GradientTape() as d_tape:
            d_logits_real = disc_model(input_real, training=True)
            d_labals_real = tf.ones_like(d_logits_real)
            print(d_logits_real.shape)
            d_loss_real = loss_fn(y_true=d_labels_real, y_pred=d_logits_real)
            d_logits_fake = disc_model(g_output, training=True)
            d_labels_fake = tf.zeros_like(d_logits_fake)
            d_loss_fake = loss_fn(y_ture=d_labels_fake, y_pred=d_logits_fake)
            d_loss = d_loss_real + d_loss_fake

        # d_lossの勾配を計算
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        # 最適化：勾配を適用
        d_optimizer.apply_gradients(
            grads_and_vars=zip(d_grads, disc_model.trainable_variables)
        )

        epoch_losses.append(
            (g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy())
        )
        d_probas_real = tf.reduce_mean(tf.sigmoid(d_logits_real))
        d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logtis_fake))
        epoch_d_vals.appennd((d_probas_real.numpy(), d_probas_fake.numpy()))
    all_losses.append(epoch_losses)
    all_d_vals.append(epoch_d_vals)
    print(
        "Epoch {:03d} | ET{:.2f} min| Avg Losses >> G/D {:.4f}/{:.4f}"
        "[D-Real:{:.4f} D-Fake:{:.4f}]".format(
            epoch,
            (time.time() - start_time) / 60,
            *list(np.mean(all_losses[-1], axis=0))
        )
    )
    epoch_samples.append(create_samples(gen_model, fixed_z).numpy())
