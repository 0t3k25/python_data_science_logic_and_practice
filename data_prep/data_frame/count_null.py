import pandas as pd
from io import StringIO

# サンプルデータを作成
csv_data = """A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,"""

# サンプルデータを読み込み
df = pd.read_csv(StringIO(csv_data))
df
# データが大きい場合における処理
# 各特徴量の欠測値をカウント
print(df.isnull().sum())
print(df.values)
