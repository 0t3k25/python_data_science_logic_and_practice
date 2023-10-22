# PCA主成分抽出
import pandas as pd
import numpy as np

# wineデータを読み込む
df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2列目以降のデータをXに、1列目のデータをyに格納
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)
# 平均と標準偏差を用いて標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 共分散行列の固有対取得
import numpy as np

cov_mat = np.cov(X_train_std.T)  # 共分散行列を作成
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)  # 固有値と固有ベクトルを計算
print(f"Eigenvalue \n {eigen_vals}")
# 分散説明率を求める
# 固有値の合計
tot = sum(eigen_vals)
# print(tot)
# 分散説明率を計算
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積和を取得
# 累積和はその時点での分散の割合を示している
cum_var_exp = np.cumsum(var_exp)
print(var_exp)
import matplotlib.pyplot as plt

# 分散説明率の棒グラフを作成
plt.bar(
    range(1, 14),
    var_exp,
    alpha=0.5,
    align="center",
    label="Individual explained variance",
)
# 分散説明率の累積和の段階グラフを作成
plt.step(range(1, 14), cum_var_exp, where="mid", label="Cumulative explained variance")
plt.ylabel("Explained variance ratio")
plt.xlabel("principal component index")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
