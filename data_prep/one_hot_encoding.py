# 名義特徴量でのone-hotエンコーディング
X = df[["color", "size", "price"]].values
print(X)
color_le = LabelEncoder()
print(X[:, 0])
# 各行の1列目の特徴量を整数に変換
X[:, 0] = color_le.fit_transform(X[:, 0])
X
# このままでは red > green > blueとなってしまう
from sklearn.preprocessing import OneHotEncoder

x = df[["color", "size", "price"]].values
# one-hotエンコーダーの生成
color_ohe = OneHotEncoder()
# one-hotエンコーディングを実行
# reshapeで3行1列の配列となる
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
# 1列だけ値を変えたい場合に使える別の方法
from sklearn.compose import ColumnTransformer

X = df[["color", "size", "price"]].values
c_transf = ColumnTransformer(
    [("one-hot", OneHotEncoder(), [0]), ("nothing", "passthrough", [1, 2])]
)
c_transf.fit_transform(X).astype(float)

# クラスラベルのエンコーディング
import numpy as np

df = pd.DataFrame(
    [
        ["green", "M", "10.1", "class2"],
        ["red", "L", "13.5", "class1"],
        ["blue", "XL", "15.3", "class2"],
    ]
)
df.columns = ["color", "size", "price", "classlabel"]
# lambda式を使った方法
# カラムの値が増える
df["x > M"] = df["size"].apply(lambda x: 1 if x in {"L", "XL"} else 0)
df["x > L"] = df["size"].apply(lambda x: 1 if x == "XL" else 0)
del df["size"]
df
