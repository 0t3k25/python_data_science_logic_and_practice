# 同心円用のデータを作成してプロット
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="^", alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="o", alpha=0.5)
plt.tight_layout()
plt.show()

# データをPCAで変換してからプロット
scikiit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color="red", marker="^", alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color="blue", marker="o", alpha=0.5)
ax[1].scatter(
    X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02, color="red", marker="^", alpha=0.5
)
ax[1].scatter(
    X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02, color="blue", marker="o", alpha=0.5
)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.tight_layout()
plt.show()

# データをRBFカーネルPCAで変換してからプロット
X_kpca, hoge = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color="red", marker="^", alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color="blue", marker="o", alpha=0.5)
ax[1].scatter(
    X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02, color="red", marker="^", alpha=0.5
)
ax[1].scatter(
    X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02, color="blue", marker="o", alpha=0.5
)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.tight_layout()
plt.show()

# 新しいデータ点の射影
X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
x_new = X[25]
print(x_new)
x_proj = alphas[25]
print(x_proj)


def project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)


x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
print(x_reproj)

plt.scatter(alphas[y == 0, 0], np.zeros((50)), color="red", marker="^", alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)), color="blue", marker="o", alpha=0.5)
plt.scatter(
    x_proj,
    0,
    color="black",
    label="Original projection of point X[25]",
    marker="^",
    s=100,
)
plt.scatter(x_reproj, 0, color="green", label="Remapped point X[25]", marker="x", s=500)
plt.yticks([], [])
plt.legend(scatterpoints=1)
plt.tight_layout()
plt.show()
