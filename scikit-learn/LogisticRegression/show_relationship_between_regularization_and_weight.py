# 過学習への対応
# 正則化
# 空のリストを生成(重み係数、逆正則化パラメータ)
weights, params = [], []
# 10個の逆正則化パラメータに対応するロジスティック回帰モデルをそれぞれ処理
for c in np.arange(-5, 5):
    print(10.0**c)
    lr = LogisticRegression(
        C=10.0**c, random_state=1, solver="lbfgs", multi_class="ovr"
    )
    lr.fit(X_train_std, y_train)
    # 重み係数を格納
    weights.append(lr.coef_[1])
    print(lr.coef_[1])
    # 逆正則化パラメータを格納
    params.append(10.0**c)
# 重み係数をNumPy配列に変換
weights = np.array(weights)
# 横軸に逆正則化パラメータ、縦軸に重み係数をプロット
plt.plot(params, weights[:, 0], label="petal length")
plt.plot(params, weights[:, 1], label="petal width")
plt.ylabel("weight coefficient")
plt.xlabel("C")
plt.legend(loc="upper left")
# 横軸を対数スケールに設定
plt.xscale("log")
plt.show()
