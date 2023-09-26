# perseptron機械学習実行
# 誤分類の数と機械学習実行回数のグラフを表示
ppn = Perceptron(eta=0.1, n_iter=10)
# 訓練データへのモデルの適合、モデルの重みを更新して最適化
ppn.fit(X, y)
# エポックと誤分類の関係を表す折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
# 軸ラベルの設定
# x軸
plt.xlabel("Epoch")
# y軸
plt.ylabel("Number of update")
# 図の表示
plt.show()
