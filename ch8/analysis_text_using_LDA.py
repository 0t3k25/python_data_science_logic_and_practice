import pandas as pd

df = pd.read_csv("/content/movie_data.csv")

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words="english", max_df=0.1, max_features=5000)
X = count.fit_transform(df["review"].values)
print(X)
