# Principal Component Analysis
# read data set
import pandas as pd

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)
# ①standardization wine data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2列目以降のデータをXに、1列目のデータをyに格納
# print(df_wine)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)
# 平均と標準偏差を用いて標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
# ②computing a covariance matrix
import numpy as np

cov_mat = np.cov(X_train_std.T)
# ③computing eigenvalues and eigenvectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print(f'Eigen_vals \n {eigen_vecs}')
# compute and plot variance explained ratio
# 固有値を合計
tot = sum(eigen_vals)
# 分散説明率を合計
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積和を取得
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt

plt.bar(
    range(1, 14),
    var_exp,
    alpha=0.5,
    align="center",
    label="Individual explained variance",
)
# 分散説明率の累積和の段階グラフを作成
plt.step(
    range(1, 14),
    cum_var_exp,
    alpha=0.5,
    where="mid",
    label="Cumulative explained variance",
)
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal component index")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
# (固有値・固有ベクトル)のタプルのリストを作成
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]
# (固有値・固有ベクトル)のタプルを大きいものから順に並べ替え
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# 射影行列w作成
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print(f"Matrix \n  {w}")
print(X_train_std[0].dot(w))
X_train_pca = X_train_std.dot(w)
colors = ["r", "b", "g"]
markers = ["s", "x", "o"]
# 「クラスラベル」「点の色」「点の種類」の組み合わせからなるリストを生成してプロット
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(
        X_train_pca[y_train == l, 0],
        X_train_pca[y_train == l, 1],
        c=c,
        label=l,
        marker=m,
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
