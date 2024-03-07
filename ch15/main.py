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
