#  tensorflow 2.x
a = tf.constant(1, name="a")
b = tf.constant(2, name="b")
c = tf.constant(3, name="c")
z = 2 * (a - b) + c
tf.print("Result: z = ", z)


def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z


@tf.function
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z


tf.print("Scalar Inputs:", compute_z(1, 2, 3))
tf.print("Rank 1 Inputs:", compute_z([1], [2], [3]))
tf.print("Rank 2 Inputs:", compute_z([[1]], [[2]], [[3]]))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    )
)
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z


tf.print("Rank 1 Inputs:", compute_z([1], [2], [3]))
tf.print("Rank 1 Inputs:", compute_z([1, 2], [2, 4], [3, 6]))

import tensorflow as tf

a = tf.Variable(initial_value=3.14, name="var_a")
print(a)

b = tf.Variable(initial_value=[1, 2, 3], name="var_b")
print(b)

c = tf.Variable(initial_value=[True, False], dtype=tf.bool)
print(c)

d = tf.Variable(initial_value=["abc"], dtype=tf.string)
print(d)

# 訓練不可変数
w = tf.Variable([1, 2, 3], trainable=False)
print(w.trainable)

print(w.assign([3, 1, 4], read_value=True))
w.assign_add([2, -1, 2], read_value=False)
print(w.value())

# 乱数生成
tf.random.set_seed(1)
init = tf.keras.initializers.GlorotNormal()
tf.print(init(shape=(3,)))
v = tf.Variable(init(shape=(2, 3)))
tf.print(v)


class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2, 3)), trainable=True)
        self.w2 = tf.Variable(init(shape=(1, 2)), trainable=False)


m = MyModule()
print("All module variabels:", [v.shape for v in m.variables])

print("Trainable variable:", [v.shape for v in m.trainable_variables])

import tensorflow as tf

w = tf.Variable(1.0)
b = tf.Variable(0.5)
print(w.trainable, b.trainable)

x = tf.convert_to_tensor([1.4])
y = tf.convert_to_tensor([2.1])

with tf.GradientTape() as tape:
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

dloss_dw = tape.gradient(loss, w)
tf.print("dL/dw:", dloss_dw)

with tf.GradientTape() as tape:
    tape.watch(x)
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

dloss_dx = tape.gradient(loss, x)
tf.print("dL/dx:", dloss_dx)

with tf.GradientTape(persistent=True) as tape:
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

dloss_dw = tape.gradient(loss, w)
tf.print("dL/dw:", dloss_dw)
dloss_db = tape.gradient(loss, b)
tf.print("dL/db:", dloss_db)

optimizer = tf.keras.optimizers.SGD()
optimizer.apply_gradients(zip([dloss_dw, dloss_db], [w, b]))
tf.print("Updated w:", w)
tf.print("Updated bias:", b)

# keras api
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=16, activation="relu"))
model.add(tf.keras.layers.Dense(units=32, activation="relu"))
## 遅延変数作成
model.build(input_shape=(None, 4))
model.summary()

## モデルの変数を出力
for v in model.variables:
    print("{:20s}".format(v.name), v.trainable, v.shape)

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Dense(
        units=16,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.glorot_uniform(),
        bias_initializer=tf.keras.initializers.Constant(2.0),
    )
)
model.add(
    tf.keras.layers.Dense(
        units=32,
        activation=tf.keras.activations.sigmoid,
        kernel_regularizer=tf.keras.regularizers.l1,
    )
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.Accuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ],
)

# 特徴寮の作成方法
# autoMPGデータの読み込み
import tensorflow as tf
import pandas as pd

dataset_path = tf.keras.utils.get_file(
    "auto-mpg.data",
    (
        "http://archive.ics.uci.edu/ml/machine-learning"
        "-databases/auto-mpg/auto-mpg.data"
    ),
)
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "ModelYear",
    "Origin",
]

df = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)

# NA行を削除
df = df.dropna()
df = df.reset_index(drop=True)
# 訓練データセットとテストデータセットに分割
import sklearn
import sklearn.model_selection

df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8)
train_stats = df_train.describe().transpose()
numeric_column_names = [
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
]
df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, "mean"]
    std = train_stats.loc[col_name, "std"]
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean) / std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std
df_train_norm.tail()

# データ変更
numeric_features = []
for col_name in numeric_column_names:
    numeric_features.append(tf.feature_column.numeric_column(key=col_name))

# モデルの年数を細分化
feature_year = tf.feature_column.numeric_column(key="ModelYear")
bucketized_features = []
bucketized_features.append(
    tf.feature_column.bucketized_column(
        source_column=feature_year, boundaries=[73, 76, 79]
    )
)

# 語彙リスト定義
feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(
    key="Origin", vocabulary_list=[1, 2, 3]
)

categorical_indicator_features = []
categorical_indicator_features.append(
    tf.feature_column.indicator_column(feature_origin)
)


def train_input_fn(df_train, batch_size=8):
    de = df_train.copy()
    train_x, train_y = df, df.pop("MPG")
    dataset - tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    return dataset.shuffle(1000).repeat(batch(batch_size))


ds = train_input_fn(df_train_norm)
batch = next(iter(ds))
print("Keys:", batch[0].keys())

print("Batch Model Year", batch[0]["ModelYear"])


def eval_input_fn(df_test, batch_size=8):
    df = df_test.copy()
    test_x, test_y = df, df.pop("MPG")
    dataset = tf.data.Dataset.from_tensor_slices((dict(test_x), test_y))
    return dataset.batch(batch_size)


# 特徴量連結
all_feature_columns = (
    numeric_features + bucketized_features + categorical_indicator_features
)
regressor = tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,
    hidden_units=[32, 10],
    model_dir="models/autompg-dnnregressor/",
)

# モデルの訓練
import numpy as np

EPOCHS = 1000
BATCH_SIZE = 8
total_steps = EPOCHS * int(np.ceil(len(df_train) / BATCH_SIZE))
print("Training Steps:", total_steps)

regressor.train(
    input_fn=lambda: train_input_fn(df_train_norm, batch_size=BATCH_SIZE),
    steps=total_steps,
).reloaded_regressor = tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,
    hidden_units=[32, 10],
    warm_start_from="models/autompg=dnnregressor/",
    model_dir="models/autompg-dnnregressor/",
)

# モデルの性能評価
eval_results = reloaded_regressor.evaluate(
    input_fn=lambda: eval_input_fn(df_test_norm, batch_size=8)
)
print("Aberage-Loss {:.4f}".format(eval_results["average_loss"]))

# モデルを使った予測
pred_res = regressor.predict(input_fn=lambda: eval_input_fn(df_test_norm, batch_size=8))
print(next(iter(pred_res)))
boosted_tree = tf.estimator.BoostedTreesRegressor(
    feature_columns=all_feature_columns, n_batches_per_layer=20, n_trees=200
)
boosted_tree.train(input_fn=lambda: train_input_fn(df_train_norm, batch_size=8))
eval_results = boosted_tree.evaluate(lambda: eval_input_fn(df_test_norm, batch_size=8))
print("Avarage-Loss{:.4f}".format(eval_results["average_loss"]))

# 手書き数字を分類する
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20
steps_per_epoch = np.ceil(60000 / BATCH_SIZE)


# float型に変換
def preprocess(item):
    image = item["image"]
    label = item["label"]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, (-1,))
    return {"image-pixels": image}, label[..., tf.newaxis]


# 訓練用と評価用の2つの入力関数を定義する
# 手順1:入力関数を定義
def train_input_fn():
    datasets = tfds.load(name="mnist")
    mnist_train = datasets["train"]
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.repeat()


def eval_input_fn():
    datasets = tfds.load(name="mnist")
    mnist_test = datasets["test"]
    dataset = mnist_test.map(preprocess).batch(BATCH_SIZE)
    return dataset


# 特徴量列を定義
image_feature_column = tf.feature_column.numeric_column(
    key="image-pixels", shape=(28 * 28)
)
# 画像のサイズを定義

# 手順3:Estimatorをインスタンス化
dnn_classifier = tf.estimator.DNNClassifier(
    feature_columns=[image_feature_column],
    hidden_units=[32, 16],
    n_classes=10,
    model_dir="models/mnist-dnn/",
)

# 手順4:訓練と評価
dnn_classifier.train(input_fn=train_input_fn, steps=NUM_EPOCHS * steps_per_epoch)
eval_result = dnn_classifier.evaluate(input_fn=eval_input_fn)
print(eval_result)

eval_result = dnn_classifier.evaluate(input_fn=eval_input_fn)
print(eval_result)

tf.random.set_seed(1)
np.random.seed(1)
# データを作成
x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0
x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(2,), name="input-features"),
        tf.keras.layers.Dense(units=4, activation="relu"),
        tf.keras.layers.Dense(units=4, activation="relu"),
        tf.keras.layers.Dense(units=4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)


# 手順1 入力関数を定義
def train_input_fn(x_train, y_train, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"input-features": x_train}, y_train.reshape(-1, 1))
    )
    # データのシャッフル、リピート、バッチ
    return dataset.shuffle(100).repeat().batch(batch_size)


def eval_input_fn(x_test, y_test=None, batch_size=8):
    if y_test is None:
        dataset = tf.data.Dataset.from_tensor_slices({"input-features": x_test})
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            {"input-features": x_test}, y_test.reshape(-1, 1)
        )
    # データのバッチ
    return dataset.batch(batch_size)


# 手順2 特徴量列を定義
features = [tf.feature_column.numeric_column(key="input-features:", shape=(2,))]

# モデルをEstimatorに変更
model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)
my_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir="models/estimator-for-XOR/"
)

# 手順4 Estimatorを使う
num_epochs = 200
batch_size = 2
steps_per_epoch = np.ceil(len(x_train) / batch_size)

my_estimator.train(
    input_fn=lambda: train_input_fn(x_train, y_train, batch_size),
    steps=num_epochs * steps_per_epoch,
)


my_estimator.evaluate(input_fn=lambda: eval_input_fn(x_valid, y_valid, batch_size))
