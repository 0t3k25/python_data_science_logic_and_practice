# iris-dataの散布図
# setosaとversicolorのみ
import matplotlib.pyplot as plt
import numpy as np

# 1-100行目の目的変数の抽出
# 目的変数は5番目の数値
y = df.iloc[0:100, 4].values
# iris-setosaを-1, iris-versicolorを1に変換
y = np.where(y == "Iris-setosa", -1, 1)
# 1-100行目の1,3列目の抽出
X = df.iloc[0:100, [0, 2]].values
# 品種setosaのプロット(赤のo)
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
# 品種versicolorのプロット(青のx)
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor")
# 軸のラベルの設定
plt.xlabel("sepal length[cm]")
plt.ylabel("petal length[cm]")
# 凡例の設定(どれがどれみたいなのを示しているやつ)
plt.legend(loc="upper left")
# 図の表示
plt.show()
