import pandas as pd

df = pd.read_csv("/content/movie_data.csv")

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words="english", max_df=0.1, max_features=5000)
X = count.fit_transform(df["review"].values)

# number of topics
print(lda.components_.shape)

n_top_words = 5
feature_names = count.get_feature_names_out()

# print topics
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx +1}")
    print(
        " ".join([feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]])
    )

horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print("\n Horror movie #%d" % (iter_idx + 1))
    print(df["review"][movie_idx][:300], "...")

# dataset読み込み
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv("movie_data.csv", encoding="utf-8", engine="python")

# 手順1:Datasetを作成
target = df.pop("sentiment")
ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))
# 調査
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])

tf.random.set_seed(1)
ds_raw = ds_raw.shuffle(50000, reshuffle_each_iteration=False)
ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

# 手順2:一意なトークン(単語)を特定
# 手順2:一意なトークン(単語)を特定
from collections import Counter

# TextVectorization レイヤーを初期化
text_vectorization = tf.keras.layers.TextVectorization(
    output_mode="int",  # トークンを整数インデックスにマッピング
    split="whitespace",  # ホワイトスペースでテキストを分割
)

# データセットのテキストデータに基づいてレイヤーを適応
# ds_raw_train.map(lambda x: x[0]) のように、テキスト部分だけを抽出する必要がある場合があります。
# 以下はテキストデータを想定していますが、実際のデータ構造に合わせて調整してください。
# adapt メソッドを使うためには、まずテキストデータを集めたリストが必要です。
all_texts = [example[0].numpy()[0] for example in ds_raw_train]
text_vectorization.adapt(all_texts)

# トークンのカウント
token_counts = Counter()

# データセットをイテレートし、各テキストをトークン化してカウント
for example in ds_raw_train:
    text = example[0].numpy()[0]  # テキストデータの抽出
    print(text)
    tokenized_text = text_vectorization([text])  # テキストをトークン化
    tokens = [
        int(token) for token in tokenized_text[0].numpy() if token != 0
    ]  # 0はパディング/未使用トークンを示す
    token_counts.update(tokens)

print("Vocab-size:", len(token_counts))


# 手順3:一意なトークンを整数にエンコード
encoder = tfds.features.text.TokenTextEncoder(token_counts)
example_str = "This is an example!"
print(encoder.encode(example_str))


# 手順3-A:変換用の関数を定義
def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label


# 3-B:encode関数をラッピングしてTensorFlow演算子に変換
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)
# 訓練データの形状をチェック
tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print("Sequence length:", example[0].shape)

# 小さなサブセットを取得
ds_subset = ds_train.take(8)
for example in ds_subset:
    print("Individual size:", example[0].shape)
ds_batched = ds_subset.padded_batch(4, padded_shapes=([-1], []))
for batch in ds_batched:
    print("Batch dimension:", batch[0].shape)

# add embedding layer
from tensorflow.keras.layers import Embedding

model = tf.keras.Sequential()
model.add(Embedding(input_dim=100, output_dim=6, input_length=20, name="embed-layer"))
model.summary()

# RNNモデルの構築
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.add(Dense(1))
model.summary()

# 感情分析のためのRNNの構築
embedding_dim = 20
vocab_size = len(token_counts) + 2
tf.random.set_seed(1)
# モデルを構築
bi_lstm_model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embegging_dim, name="embed-layer"
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, name="lstm-layer"), name="bidir-lstm"
        ),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
bi_lstm_model.summary()
# コンパイルと訓練
bi_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(le - 3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)
history = bi_lstm_model.fit(train_data, validation_data=valid_data, epochs=10)
# テストデータでの評価
test_results = bi_lstm_model.evaluate(test_data)
print("Test Acc:{:.2f}%".format(test_results[1] * 100))
