# 3つのクラスの分類を実行
# うまく分類はできていない
# 訓練データとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# 訓練データとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))
# 決定領域のプロット
plot_decision_regions(
    X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150)
)
# 軸のラベルの設定
plt.xlabel("petal length[standardized]")
plt.ylabel("petal width[standardized]")
# 凡例の設定
plt.legend(loc="upper left")
# グラフを表示
plt.tight_layout()
plt.show()
