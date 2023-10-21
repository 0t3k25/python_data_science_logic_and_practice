# 実際にsbsを使ってみよう
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# k最近傍法分類器のインスタンスを生成(近傍点線=5)
knn = KNeighborsClassifier(n_neighbors=5)
# 逐次後退選択のインスタンスを生成(特徴量の個数が1になるまで特徴量を選択)
sbs = SBS(knn, k_features=1)
# 逐次後退選択を実行
sbs.fit(X_train_std, y_train)
# knn分類器の正解率の可視化
# 特徴量の個数のリスト(13,12,....,1)
k_feat = [len(k) for k in sbs.subsets_]
# 横軸を特徴量の個数、縦軸をスコアとした折れ線グラフのプロット
plt.plot(k_feat, sbs.scores_, marker="o")
plt.ylim([0.7, 1.02])
plt.ylabel("Accuracy")
plt.xlabel("Number of features")
plt.grid()
plt.tight_layout()
plt.show()
