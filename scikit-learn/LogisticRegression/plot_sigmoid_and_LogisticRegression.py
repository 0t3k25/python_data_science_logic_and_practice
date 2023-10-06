# y=1のコストを計算する関数
def cost_1(z):
    return -np.log(sigmoid(z))


# y= 0 のコストを計算する関数
def cost_0(z):
    return -np.log(1 - sigmoid(z))


# 0.1間隔で-10以上10未満のデータを生成
z = np.arange(-10, 10, 0.1)
# シグモイド関数を実行
phi_z = sigmoid(z)
# y=1のコストを計算する関数を実行
c1 = [cost_1(x) for x in z]
# 結果をプロット
plt.plot(phi_z, c1, label="J(w) if y =1")
# y=0のコストを計算する関数を生成
c0 = [cost_0(x) for x in z]
# 結果をプロット
plt.plot(phi_z, c0, linestyle="--", label="J(w) if y =0")
# x軸とy軸の上限/下限を設定
plt.ylim(0.0, 5, 1)
plt.xlim([0, 1])
# 軸のラベルを設定
plt.xlabel("$\phi$(z)")
plt.ylabel("J(w)")
# 凡例を設定
plt.legend(loc="upper left")
# グラフを表示
plt.tight_layout()
plt.show()
