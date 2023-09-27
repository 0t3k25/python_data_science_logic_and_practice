# 決定領域のプロット
plot_decision_regions(X, y, classifier=ppn)
# 軸ラベルの設定
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
# 凡例の設定（左上に設定）
plt.legend(loc="upper left")
# 図を表示
plt.show()
