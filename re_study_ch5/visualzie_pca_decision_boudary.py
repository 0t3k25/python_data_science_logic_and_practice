from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# 主成分数を指定して、PCAのインスタンスを生成
pca = PCA(n_components=2)
# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(multi_class="ovr", random_state=1, solver="lbfgs")
# 次元削減
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# 削減したデータセットでロジスティック回帰モデルを適合
lr.fit(X_train_pca, y_train)
# 決定境界をプロット
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
# 決定境界をプロット
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
# それぞれの分散率に関して表示
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
