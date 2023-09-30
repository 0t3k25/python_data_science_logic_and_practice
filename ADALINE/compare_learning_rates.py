import matplotlib.pyplot as plt

# 学習率が大きすぎる場合の例ax[0]発散,学習率が小さすぎる時の例ax[1]最小値を見つけるまで時間がかかる。
# 描画領域を1行2列に分割
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# 勾配降下法によるADALINEの学習(学習率 eta = 0.01)
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# エポック数とコスト関数の関係を表す折線グラフのプロット(縦軸のコストは常用対数)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker="o")
# 軸のラベルの設定
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(Sum-squared-error)")
# タイトルの設定
ax[0].set_title("Adaline - Learning rate 0.01")
# 勾配降下法によるADALINEの学習(学習率 eta = 0.0001)
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# エポック数とコストの関係を表す折れ線グラフのプロット
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker="o")
# 軸ラベルの設定
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Sum-squared-error")
# タイトルの設定
ax[1].set_title("Adaliine - Learning rate 0.0001")
# 図の表示
plt.show()
