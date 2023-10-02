# std→標準化
# numpyを用いたデータの標準化
# データのコピー
X_std = np.copy(X)
# 各列の標準化
# 標準偏差を1に平均を0に
# この場合特徴量が2つのため
# xj = (x - μ(平均)) / σ(標準偏差)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
# 標準化を用いることによって
# 学習回数が同じでも標準化したデータの方が誤差平方和が小さい
# 勾配降下法によるADALINEの学習(標準化後、学習率eta = 0.01)
ada_gd = AdalineGD(n_iter=15, eta=0.01)
# モデルの結合
ada_gd.fit(X_std, y)
# 決定領域のプロット
plot_decision_regions(X_std, y, classifier=ada_gd)
# タイトルの設定
plt.title("Adaline - Gradient Descent")
# 軸のラベルの設定
plt.xlabel("sepal length [standarized]")
plt.xlabel("petal length [standarized]")
# 凡例の設定
plt.legend(loc="upper left")
# 余白などの修正
plt.tight_layout()
# 図の表示
plt.show()
# エポック数とコストの関係を表す折れ線グラフのプロット
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker="o")
# 軸ラベルの設定
plt.xlabel("Epochs")
plt.ylabel("Sum-squared-error")
# 余白などの修正
plt.tight_layout()
# 図の表示
plt.show()
