# L1正則化パスのプロット
import matplotlib.pyplot as plt

# 描画の準備
fig = plt.figure()
ax = plt.subplot(111)
# 各係数の色のリスト
colors = [
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "pink",
    "lightgreen",
    "lightblue",
    "gray",
    "indigo",
    "orange",
]
# 空のリストを生成(重み係数、逆正則化パラメータ)
weights, params = [], []
# 逆正則化パラメータの値ごとに処理
for c in np.arange(-4, 6.0):
    lr = LogisticRegression(
        penalty="l1", C=10.0**c, solver="liblinear", multi_class="ovr", random_state=0
    )
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

# 重み係数をNumpy配列に変換
weights = np.array(weights)
# 各重み係数をプロット
for column, color in zip(range(weights.shape[1]), colors):
    # 横軸を逆正則化パラメータ、縦軸を重み係数とした折線グラフ
    plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)
# y=0に黒い破線を引く
plt.axhline(0, color="black", linestyle="--", linewidth=3)
# 横軸の範囲の設定
plt.xlim([10 ** (-5), 10**5])
# 軸ラベルの設定
plt.ylabel("weight coefficient")
plt.xlabel("C")
# 横軸を対数スケールに設定
plt.xscale("log")
plt.legend(loc="upper left")
ax.legend(loc="upper center", bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()
