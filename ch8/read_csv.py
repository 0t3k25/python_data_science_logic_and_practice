# csvファイルデータ取得
from google.colab import files

uploaded = files.upload()

# データロード
import pandas as pd
import io

df = pd.read_csv(io.BytesIO(uploaded["movie_data.csv"]))

df.columns = ["review", "sentiment"]
print(df.columns)

# ちゃんと情報が取れているか確認
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv("movie_data.csv", index=False, encoding="utf-8")
df = pd.read_csv("movie_data.csv", encoding="utf-8")
df.head(3)
df.shape
