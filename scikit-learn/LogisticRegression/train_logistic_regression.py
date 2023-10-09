# ovr multinomialそれぞれで実装
# 相互排他的なクラスでは、multinomialが推奨される。
from sklearn.linear_model import LogisticRegression

# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(C=100.0, random_state=1, solver="lbfgs", multi_class="ovr")
# 訓練データをモデルに適合させる。
lr.fit(X_train_std, y_train)
# 決定境界をプロット
plot_decision_regions(
    X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150)
)
plt.xlabel("petal length [standardized]")  # 軸のラベルを設定
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")  # 凡例を設定
plt.tight_layout()  # グラフを表示
plt.show()
lr.predict_proba(X_test_std[:3, :])
# axis=1で列を1行にします
# :3は3行目まで取る、,の右は列を表す:なので全て
lr.predict_proba(X_test_std[:3, :]).sum(axis=1)
# クラスラベル返却一番確率の高いものを返す [2,0,0]など
lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
# クラスラベルの予測↑と同じ値を返す
lr.predict(X_test_std[:3, :])
# 単一データを入れるときの注意点、2次元配列に変換する必要がある。
# reshape(1,-1)を使って2次元に変換
lr.predict(X_test_std[0, :].reshape(1, -1))
