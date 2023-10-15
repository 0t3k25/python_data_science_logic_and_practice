# k最近傍法を用いた分類
from sklearn.neighbors import KNeighborsClassifier

# k最近傍法のインスタンスを生成
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
# 訓練データにk最近傍法のモデルを適合させる
knn.fit(X_train_std, y_train)
plot_decision_regions(
    X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150)
)
plt.xlabel("petal length[standardized]")
plt.ylabel("petal width[standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
