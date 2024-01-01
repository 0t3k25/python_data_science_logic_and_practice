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
