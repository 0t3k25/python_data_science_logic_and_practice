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
