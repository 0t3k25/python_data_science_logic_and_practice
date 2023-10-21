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

# check why number of features 3 accuracy is 1.00 point
k3 = list(sbs.subsets_[10])
print(k3)
# 1列目にはクラスラベルが存在
print(df_wine.columns[1:][k3])
# 13個全ての特徴量を用いてモデルを適合
knn.fit(X_train_std, y_train)
# 訓練の正解率を出力
print("Training accuracy:", knn.score(X_train_std, y_train))
# テストの正解率を出力
print("Test accuracy:", knn.score(X_test_std, y_test))

# 3つの特徴量を用いてモデルを適合
knn.fit(X_train_std[:, k3], y_train)
# 訓練の正解率を出力
print("Traingin accuracy:", knn.score(X_train_std[:, k3], y_train))
# テストの正解率を出力
print("Test accuracy:", knn.score(X_test_std[:, k3], y_test))
