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

from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print(f"Slope: {slr.coef_[0]: .3f}")
print(f"Intercept: {slr.intercept_: .3f}")

lin_regplot(X, y, slr)
plt.xlabel("Average number of room[RM]")
plt.ylabel("Prive in $1000s[MEDV]")
plt.show()

from sklearn.linear_model import RANSACRegressor

# RANSACモデルをインスタンス化
ransac = RANSACRegressor(
    LinearRegression(),
    max_trials=100,
    min_samples=50,
    loss="absolute_error",
    residual_threshold=5.0,
    random_state=0,
)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_  # 正常値を表す真偽値を取得
outlier_mask = np.logical_not(inlier_mask)  # 外れ値を表す真偽値を取得
line_X = np.arange(3, 10, 1)  # 3から9までの整数値を作成
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
# 正常値をプロット
plt.scatter(
    X[inlier_mask],
    y[inlier_mask],
    c="steelblue",
    edgecolors="white",
    marker="o",
    label="Inliers",
)
# 外れ値をプロット
plt.scatter(
    X[outlier_mask],
    y[outlier_mask],
    c="limegreen",
    edgecolors="white",
    marker="s",
    label="Outliers",
)
plt.plot(line_X, line_y_ransac, color="black", lw=2)
plt.xlabel("Average number of room[RM]")
plt.ylabel("Price in $1000s [MEDV]")
plt.legend(loc="upper left")
plt.show()

# Multiple Regression Models
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values
y = df["MEDV"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
# residual plot
plt.scatter(
    y_train_pred,
    y_train_pred - y_train,
    c="steelblue",
    marker="o",
    edgecolor="white",
    label="Training data",
)
plt.scatter(
    y_test_pred,
    y_test_pred - y_test,
    c="limegreen",
    marker="s",
    edgecolor="white",
    label="Test data",
)
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=-10, xmax=50, color="black", lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error

# 平均二乗誤差を出力
print(
    f"MSE train: {mean_squared_error(y_train,y_train_pred): .3f} ,test: {mean_squared_error(y_test,y_test_pred): .3f}"
)

# R^2(決定形数)のスコアを出力
from sklearn.metrics import r2_score

print(
    f"R^2 train {r2_score(y_train, y_train_pred): .3f} test: {r2_score(y_test,y_test_pred):.3f}"
)

# ridge model
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # L2ペナルティ項の影響度合いを表す値を引数に指定

# Lasso model
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0)

# Elastic Net model
from sklearn.linear_model import ElasticNet

elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
