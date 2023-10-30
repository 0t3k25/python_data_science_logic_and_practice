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
print(f"Eigen_vals \n {eigen_vecs}")
