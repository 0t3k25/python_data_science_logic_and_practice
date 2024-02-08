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
