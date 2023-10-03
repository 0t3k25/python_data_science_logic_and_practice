# 確率勾配降下法によるADALINEの学習
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# モデルへの適合
ada_sgd.fit(X_std, y)
# 境界領域のプロット
plot_decision_regions(X_std, y, classifier=ada_sgd)
# タイトルの設定
plt.title("Adaline - Stochastic Gradient Descent")
# 軸のラベルの設定
plt.xlabel("sepal length[standard]")
plt.ylabel("petal length[standard]")
# 凡例の設定(左上に配置)
plt.legend(loc="upper left")
plt.tight_layout()
# プロットの表示
plt.show()
# エポックとコストの折れ線グラフのプロット
plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker="o")
# 軸のラベルの設定
plt.xlabel("Epochs")
plt.ylabel("Average Cost")
# プロットの表示
plt.tight_layout()
plt.show()

# 余談
# # ストリーミングでデータを呼び出してモデルを更新したい場合
ada_sgd.partial_fit(X_std[0, :], y[0])
