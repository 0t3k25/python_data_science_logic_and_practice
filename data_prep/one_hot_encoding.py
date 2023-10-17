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
