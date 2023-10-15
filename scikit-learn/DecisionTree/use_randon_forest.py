# アンサンブル法のランダムフォレストをつかって決定境界を決定
# および表示
from sklearn.ensemble import RandomForestClassifier

# ジニ不純度を指標とするランダムフォレストのインスタンスを生成
# n_estimatorsで使用するランダムフォレストの数を指定
# n_jobsで使用するコアの数を指定しているこの場合2なので並列処理をしている
forest = RandomForestClassifier(
    criterion="gini", n_estimators=25, random_state=1, n_jobs=2
)
# 訓練データにランダムフォレストのモデルを適合させる
forest.fit(X_train, y_train)
plot_decision_regions(
    X_combined, y_combined, classifier=forest, test_idx=range(105, 150)
)
plt.xlabel("petal length[cm]")
plt.ylabel("petal width[cm]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
