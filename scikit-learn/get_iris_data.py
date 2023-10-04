from sklearn import datasets
import numpy as np

# irisデータセットをロード
iris = datasets.load_iris()
# print(iris)
# 3、4列目の特徴量を抽出
X = iris.data[:, [2, 3]]
# クラスラベルを取得
y = iris.target
# 一意なクラスラベルを出力
print("Class labels: ", np.unique(y))
