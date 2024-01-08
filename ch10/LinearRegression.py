# データの取得
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt",
    header=None,
    sep="\s+",
)
df.columns = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
df.head()

# 探索的データ解析EDA
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

cols = ["LSTAT", "INDUS", "NOX", "RM", "MEDV"]
# 変数のペアの関係をプロット:dfはDataFrameのオブジェクト
scatterplotmatrix(df[cols].values, figsize=(10, 8), names=cols, alpha=0.5)
plt.tight_layout()
plt.show()

# ピアソンの積率相関係数
# ヒートマップ表示
from mlxtend.plotting import heatmap
import numpy as np

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()


# LinearRegression
# 基本的な線形回帰モデル
class LinearRegressionGD(object):
    # 初期化
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta  # 学習率
        self.n_iter = n_iter  # 訓練回数

    # 訓練を実行する fit
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])  # 重みを初期化
        self.cost_ = []  # コスト関数の値を初期化
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    # 総入力を計算
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


X = df[["RM"]].values
y = df["MEDV"].values
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()

lr.fit(X_std, y_std)

# エポック数とコスト関数の関係を表す折線グラフのプロット
plt.plot(range(1, lr.n_iter + 1), lr.cost_)
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.show()


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="steelblue", edgecolor="white", s=70)
    plt.plot(X, model.predict(X), color="black", lw=2)
    return None


lin_regplot(X_std, y_std, lr)
plt.xlabel("Average number of rooms [RM]")
plt.ylabel("Price in $1000s [MEDV]")
plt.show()
num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print(f"Price in $1000s: {sc_y.inverse_transform([price_std])}")
