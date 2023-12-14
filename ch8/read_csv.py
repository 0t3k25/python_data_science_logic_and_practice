# csvファイルデータ取得
from google.colab import files

uploaded = files.upload()

# データロード
import pandas as pd
import io

df = pd.read_csv(io.BytesIO(uploaded["movie_data.csv"]))

df.columns = ["review", "sentiment"]
print(df.columns)
