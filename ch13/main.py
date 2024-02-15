# テンソルの確認
import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)

np.set_printoptions(precision=3)
a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]
t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)
print(t_a)
print(t_b)

t_ones = tf.ones((2, 3))
t_ones.numpy()

const_tensor = tf.constant([1.2, 5, np.pi], dtype=tf.float32)
print(const_tensor)

t_a_new = tf.cast(t_a, tf.int64)
print(t_a_new.dtype)

# テンソルの転置
t = tf.random.uniform(shape=(3, 5))
t_tr = tf.transpose(t)
print(t.shape, "-->", t_tr.shape)

# テンソルの形状変更
t = tf.zeros((30,))
print(t)
t_reshape = tf.reshape(t, shape=(5, 6))
print(t_reshape)

# 不要な次元の削除
t = tf.zeros((1, 2, 1, 4, 1))
t_sqz = tf.squeeze(t, axis=(2, 4))  # 2,4を削除
print(t.shape, "-->", t_sqz.shape)

tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5, 2), minval=-1.0, maxval=1.0)
t2 = tf.random.normal(shape=(5, 2), mean=0.0, stddev=1.0)
# print(t1,t2)

t3 = tf.multiply(t1, t2).numpy()
print(t3)

t4 = tf.math.reduce_mean(t1, axis=0)
print(t4)

t5 = tf.linalg.matmul(t1, t2, transpose_b=True)
print(t5)

t6 = tf.linalg.matmul(t1, t2, transpose_a=True)
print(t6)

norm_t1 = tf.norm(t1, ord=2, axis=1).numpy()
print(norm_t1)

tf.random.set_seed(1)
t = tf.random.uniform((6,))
print(t.numpy())

t_splits = tf.split(t, num_or_size_splits=3)
[item.numpy() for item in t_splits]

tf.random.set_seed(1)
t = tf.random.uniform((5,))

print(t.numpy())

t_splits = tf.split(t, num_or_size_splits=[3, 2])
[item.numpy() for item in t_splits]

A = tf.ones((3,))
B = tf.zeros((2,))
C = tf.concat([A, B], axis=0)
print(C.numpy())

A = tf.ones((3,))
B = tf.zeros((3,))
S = tf.stack([A, B], axis=1)
print(S.numpy())

a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)

for item in ds:
    print(item)

ds_batch = ds.batch(3)
for i, elem in enumerate(ds_batch, 1):
    print(f"batch {i}: {elem.numpy()}")

tf.random.set_seed(1)
t_x = tf.random.uniform([4, 3], dtype=tf.float32)
t_y = tf.range(4)

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)
ds_joint = tf.data.Dataset.zip((ds_x, ds_y))
for example in ds_joint:
    print(f"x: {example[0].numpy()} y:{example[1].numpy()}")

ds_trans = ds_joint.map(lambda x, y: (x * 2 - 1.0, y))
for example in ds_trans:
    print(f"x: {example[0].numpy()} y: {example[1].numpy()}")

ds = ds_joint.batch(batch_size=3, drop_remainder=False)
batch_x, batch_y = next(iter(ds))
print(f"Batch-x:\n{batch_x.numpy()}")
print(f"Batch-y:{batch_y.numpy()}")

#
ds = ds_joint.repeat(count=2).batch(3)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# shuffle->batch->repeat
tf.random.set_seed(1)
ds = ds_joint.shuffle(4).batch(2).repeat(3)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

tf.random.set_seed(1)
ds = ds_joint.batch(2).shuffle(20).repeat(3)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# 画像ファイル取得
import pathlib

imgdir_path = pathlib.Path("/content/cat_dog_images")
file_list = sorted([str(path) for path in imgdir_path.glob("*.jpg")])

print(file_list)

import matplotlib.pyplot as plt
import os

fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print("image shape: ", img.shape)
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)

# labeling
labels = [1 if "dog" in os.path.basename(file) else 0 for file in file_list]
print(labels)

# make dataset
ds_file_labels = tf.data.Dataset.from_tensor_slices((file_list, labels))
for item in ds_file_labels:
    print(item[0].numpy(), item[1].numpy())


# 前処理
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0
    return image, label


img_width, img_height = 120, 80
ds_images_labels = ds_file_labels.map(load_and_preprocess)
fig = plt.figure(figsize=(10, 5))
for i, example in enumerate(ds_images_labels):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])
    ax.set_title(f"{example[1].numpy()}", size=15)
plt.tight_layout()
plt.show()

# celeb_aデータ読み込み
import tensorflow_datasets as tfds

print(len(tfds.list_builders()))

print(tfds.list_builders()[:5])

celeba_bldr = tfds.builder("celeb_a")
print(celeba_bldr.info.features)
print(celeba_bldr.info.features["image"])
print(celeba_bldr.info.features["attributes"].keys())
print(celeba_bldr.info.citation)

# mnistデータ
mnist, mnist_info = tfds.load("mnist", with_info=True, shuffle_files=False)
print(mnist_info)

ds_train = mnist["train"]
ds_train = ds_train.map(lambda item: (item["image"], item["label"]))
ds_train = ds_train.batch(10)
batch = next(iter(ds_train))
print(batch[0].shape, batch[1])

fig = plt.figure(figsize=(15, 6))
for i, (image, label) in enumerate(zip(batch[0], batch[1])):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image[:, :, 0], cmap="gray_r")
    ax.set_title(f"{label}", size=15)

import numpy as np
import matplotlib.pyplot as plt

# 線形回帰
X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])
plt.plot(X_train, y_train, "o", markersize=10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 標準化
import tensorflow as tf

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)

ds_train_orig = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train_norm, tf.float32), tf.cast(y_train, tf.float32))
)


# keras
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = tf.Variable(0.0, name="weight")
        self.b = tf.Bariable(0.0, name="bias")

    def call(self, X):
        return self.w * X + self.b


model = MyModel()
model.build(input_shape=(None, 1))
model.summary()


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
        dW, db = tape.grandient(current_loss, [model.w, model.b])
        model.w.assign_sub(learning_rate * dW)
        model.b.assign_sub(learning_rate * db)


tf.random.set_seed(1)
num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epochs = int(np.ceil(len(y_train) / batch_size))

ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.repeat(count=None)
ds_train = ds_train.batch(1)
Ws, bs = [], []

for i, batch in enumerate(ds_train):
    if i >= steps_per_epochs * num_epochs:
        break  # 無限ループを抜ける
    Ws.append(model.w.numpy())
    bs.append(model.b.numpy())

    bx, by = batch
    loss_val = loss_fn(model(bx), by)

    train(model, bx, by, learning_rate=learning_rate)
    if i % log_steps == 0:
        print(
            "Epoch {:4d} Step {:2d} Loss {:6.4f}".format(
                int(i / steps_per_epochs), i, loss_val
            )
        )

print("Final Paraneters: ", model.w.numpy(), model.b.numpy())

import numpy as np

X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, "o", markersize=10)
plt.plot(X_test_norm, y_pred, "--", lw=3)
plt.legend(["Training examples", "Linear Reg."], fontsize=15)
ax.set_xlabel("x", size=15)
ax.set_ylabel("y", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(["Weight w", "Bias unit b"], fontsize=15)
ax.set_xlabel("Iteration", size=15)
ax.set_ylabel("Value", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)
plt.show()

# compile methodの利用
tf.random.set_seed(1)
model = MyModel()
model.compile(optimizer="sgd", loss=loss_fn, metrics=["mae", "mse"])
model.fit(X_train_norm, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

import tensorflow_datasets as tfds

# iris-datasetの取得
iris, iris_info = tfds.load("iris", with_info=True)
print(iris_info)

# 訓練データとテストデータに分割
tf.random.set_seed(1)
ds_orig = iris["train"]
ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)
ds_train_orig = ds_orig.take(100)
ds_test = ds_orig.skip(100)

ds_train_orig = ds_train_orig.map(lambda x: (x["features"], x["label"]))
ds_test = ds_test.map(lambda x: (x["features"], x["label"]))

iris_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(16, activation="sigmoid", name="fc1", input_shape=(4,)),
        tf.keras.layers.Dense(3, name="fc2", activation="softmax"),
    ]
)
iris_model.summary()

# compile
iris_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
import numpy as np

num_epochs = 100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil((training_size / batch_size))
ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)
history = iris_model.fit(
    ds_train, epochs=num_epochs, steps_per_epoch=steps_per_epoch, verbose=0
)

import matplotlib.pyplot as plt

hist = history.history
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(hist["loss"], lw=3)
ax.set_title("Training loss", size=15)
ax.set_xlabel("Epoch", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(hist["accuracy"], lw=3)
ax.set_title("Training accuracy", size=15)
ax.set_xlabel("Epoch", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)
plt.tight_layout()
plt.show()

results = iris_model.evaluate(ds_test.batch(50), verbose=0)
print("Test loss: {:.4f} Test Acc. {:.4f}".format(*results))

import numpy as np

X = np.array([1, 1.4, 2.5])  # 1つ目の値は1でなければならない
w = np.array([0.4, 0.3, 0.5])


def net_input(X, w):
    return np.dot(X, w)


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)


print(f"P(y=1|x) = {logistic_activation(X,w) :.3f}")

# W : array , shape =[n_output_units, n_hidden_units + 1]
#  この配列の最初の列(W[:][0])はバイアスユニット
W = np.array([[1.1, 1.2, 0.8, 0.4], [0.2, 0.4, 1.0, 0.2], [0.6, 1.5, 1.2, 0.7]])
# A : array, shape = (n_hidden_units+1,n_samples)
#  この配列の最初の列(A[0][0])は1でなければならない

A = np.array([[1, 0.1, 0.4, 0.6]])
# Z : array,shape = [n_output_units, n_samples]
# 出力層の総入力
Z = np.dot(W, A[0])
y_probas = logistic(Z)
print(f"Net Input: \n", Z)
print(f"Output Units:\n", y_probas)

y_class = np.argmax(Z, axis=0)
print(f"Predicted class label: {y_class}")


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


y_probas = softmax(Z)
print(f"Probabilities:\n", y_probas)

np.sum(y_probas)

import tensorflow as tf

Z_tensor = tf.expand_dims(Z, axis=0)
tf.keras.activations.softmax(Z_tensor)

# compare logistc vs hyperbolic tangent
import matplotlib.pyplot as plt


def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5, 1.5])
plt.xlabel("net input $z$")
plt.ylabel("activation $\phi(z)$")
plt.axhline(1, color="black", linestyle=":")
plt.axhline(0.5, color="black", linestyle=":")
plt.axhline(0, color="black", linestyle=":")
plt.axhline(-0.5, color="black", linestyle=":")
plt.axhline(-1, color="black", linestyle=":")
plt.plot(z, tanh_act, linewidth=3, linestyle="--", label="tanh")
plt.plot(z, log_act, linewidth=3, label="logistic")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print(np.tanh(z))

tf.keras.activations.tanh(z)

from scipy.special import expit

expit(z)

tf.keras.activations.sigmoid(z)

# ReLUの適用法
tf.keras.activations.relu(z)
