# 畳み込み演算の実施
import numpy as np


def convld(x, w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
    res = []
    for i in range(0, int((len(x_padded) - len(w_rot)) / s) + 1, s):
        res.append(np.sum(x_padded[i : i + w_rot.shape[0]] * w_rot))
    return np.array(res)


# テスト
x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]
print("Convld Implementation", convld(x, w, p=2, s=1))
print()

# 2次元の畳み込みを実施
import numpy as np
import scipy.signal


def conv2d(X, W, p=(0, 0), s=(1, 1)):
    W_rot = np.array(W)[::-1, ::-1]
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2 * p[0]
    n2 = X_orig.shape[1] + 2 * p[1]
    X_padded = np.zeros(shape=(n1, n2))
    X_padded[p[0] : p[0] + X_orig.shape[0], p[1] : p[1] + X_orig.shape[1]] = X_orig
    res = []
    for i in range(0, int((X_padded.shape[0] - W_rot.shape[0]) / s[0]) + 1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1] - W_rot.shape[1]) / s[1]) + 1, s[1]):
            X_sub = X_padded[i : i + W_rot.shape[0], j : j + W_rot.shape[1]]

            res[-1].append(np.sum(X_sub * W_rot))
    return np.array(res)


X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]
print("conv2d implementation:\n", conv2d(X, W, p=(1, 1), s=(1, 1)))

print("scipy result:\n", scipy.signal.convolve2d(X, W, mode="same"))

from tensorflow import keras

conv_layer = keras.layers.Conv2D(
    filters=16, kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(0.001)
)
fc_layer = keras.layers.Dense(units=16, kernel_regularizer=keras.regularizers.l2(0.001))

import tensorflow as tf

# 分類の損失関数、logit logistic
# binarycrossentropy
bce_probas = tf.keras.losses.BinaryCrossentropy(from_logits=False)
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
logits = tf.constant([0.8])
probas = tf.keras.activations.sigmoid(logits)
tf.print(
    "BCE(w Probas): {:.4f}".format(bce_probas(y_true=[1], y_pred=probas)),
    "(w Logits): {:.4f}".format(bce_logits(y_true=[1], y_pred=logits)),
)

# categoricalcrossentropy
cce_probas = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
cce_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
logits = tf.constant([[1.5, 0.8, 2.1]])
probas = tf.keras.activations.softmax(logits)
tf.print(
    "CCE(w probas): {:.4f}".format(cce_probas(y_true=[[0, 0, 1]], y_pred=probas)),
    "(w logits): {:.4f}".format(cce_logits(y_true=[[0, 0, 1]], y_pred=logits)),
)

# sparseCategoricalCrossentropy
sp_cce_probas = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
sp_cce_logits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
tf.print(
    "Sparse CCE (w probas): {:.4f}".format(sp_cce_probas(y_true=[2], y_pred=probas)),
    "(w logits): {:.4f}".format(sp_cce_logits(y_true=[2], y_pred=logits)),
)

# mnistデータの読み込み
import tensorflow_datasets as tfds

# 手順1
mnist_bldr = tfds.builder("mnist")
# 手順2
mnist_bldr.download_and_prepare()
# 手順3
datasets = mnist_bldr.as_dataset(shuffle_files=False)
# 訓練データセットとテストデータセットを取得
mnist_train_orig = datasets["train"]
mnist_test_orig = datasets["test"]

import tensorflow as tf

# 訓練データセットと検証データセットの分割
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20
mnist_train = mnist_train_orig.map(
    lambda item: (
        tf.cast(item["image"], tf.float32) / 255.0,
        tf.cast(item["label"], tf.int32),
    )
)
mnist_test = mnist_test_orig.map(
    lambda item: (
        tf.cast(item["image"], tf.float32) / 255.0,
        tf.cast(item["label"], tf.int32),
    )
)
tf.random.set_seed(1)
mnist_train = mnist_train.shuffle(
    buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False
)
mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)

# kerasでのCNNの構築
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        data_format="channels_last",
        name="conv_1",
        activation="relu",
    )
)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_1"))
model.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        name="conv_2d",
        activation="relu",
    )
)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_2"))

# 特徴量mapのサイズを計算
model.compute_output_shape(input_shape=(16, 28, 28, 1))

# 層の平坦化
model.add(tf.keras.layers.Flatten())
model.compute_output_shape(input_shape=(16, 28, 28, 1))


model.add(tf.keras.layers.Dense(units=1024, name="fc_1", activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(units=10, name="fc_2", activation="softmax"))

tf.random.set_seed(1)
model.build(input_shape=(None, 28, 28, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    mnist_train, epochs=NUM_EPOCHS, validation_data=mnist_valid, shuffle=True
)

# import numpy as np
# 学習曲線を可視化
import matplotlib.pyplot as plt

hist = history.history
x_arr = np.arange(len(hist["loss"])) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist["loss"], "-o", label="Train loss")
ax.plot(x_arr, hist["val_loss"], "--<", label="Validation loss")
ax.set_xlabel("Epoch", size=15)
ax.set_ylabel("Loss", size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist["accuracy"], "-o", label="Train acc")
ax.plot(x_arr, hist["val_accuracy"], "--<", label="Validation acc")
ax.set_xlabel("Epoch", size=15)
ax.set_ylabel("Accuracy", size=15)
ax.legend(fontsize=15)
plt.show()

# 評価
test_results = model.evaluate(mnist_test.batch(20))
print("Test Acc.: {:.2f}%".format(test_results[1] * 100))

# 入力値と予測ラベルの可視化
batch_test = next(iter(mnist_test.batch(12)))
preds = model(batch_test[0])
tf.print(preds.shape)

preds = tf.argmax(preds, axis=1)
print(preds)

fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    img = batch_test[0][i, :, :, 0]
    ax.imshow(img, cmap="gray_r")
    ax.text(
        0.9,
        0.1,
        "{}".format(preds[i]),
        size=15,
        color="blue",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
plt.show()

import tensorflow as tf
import tensorflow_datasets as tfds

celeba_bldr = tfds.builder("celeb_a")
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_file=False)
celeba_train = celeba["train"]
celeba_valid = celeba("validation")
celeba_test = celeba["test"]


def count_items(ds):
    n = 0
    for _ in ds:
        n += 1
    return n


print("Train set: {}".format(count_items(celeba_train)))

print("Validation: {}".format(count_items(celeba_valid)))

print("Test set: {}".format(count_items(celeba_test)))

import tensorflow as tf

tf.random.set_seed(1)
rnn_layer = tf.keras.layers.SimpleRNN(units=2, use_bias=True, return_sequences=True)
rnn_layer.build(input_shape=(None, None, 5))
w_xh, w_oo, b_h = rnn_layer.weights
print("W_xh shape:", w_xh.shape)
print("W_oo shape:", w_oo.shape)
print("b_h shape:", b_h.shape)

x_seq = tf.convert_to_tensor([[1.0] * 5, [2.0] * 5, [3.0] * 5], dtype=tf.float32)
# simpleRNNの出力
output = rnn_layer(tf.reshape(x_seq, shape=(1, 3, 5)))
# 出力を手動で計算
out_man = []
for t in range(len(x_seq)):
    xt = tf.reshape(x_seq[t], (1, 5))
    print("Time step {}=>".format(t))
    print(" Input.    :", xt.numpy())

    ht = tf.matmul(xt, w_xh) + b_h
    print("Hidden:", ht.numpy())

    if t > 0:
        prev_o = out_man[t - 1]
    else:
        prev_o = tf.zeros(shape=(ht.shape))

    ot = ht + tf.matmul(prev_o, w_oo)
    ot = tf.math.tanh(ot)
    out_man.append(ot)
    print(" Output(manual):", ot.numpy())
    print(" SimpleRNN output:".format(t), output[0][t].numpy())
    print()
