import matplotlib.pyplot as plt

# 2つの半月形データを作成してプロット
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="^", alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="o", alpha=0.5)
plt.tight_layout()
plt.show()

# 標準のpcaを使って線形分離
from sklearn.decomposition import PCA

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
# グラフの数と配置、サイズを指定
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# 1番目のグラフ領域に散布図をプロット
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color="red", marker="^", alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color="blue", marker="o", alpha=0.5)
# 2番目のグラフ領域に散布図をプロット
ax[1].scatter(
    X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02, color="red", marker="^", alpha=0.5
)
ax[1].scatter(
    X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02, color="blue", marker="o", alpha=0.5
)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
# y軸のメモリ削除
ax[1].set_yticks([])
ax[1].set_xlabel(["PC1"])
plt.tight_layout()
plt.show()

# kernel_pca
from matplotlib.ticker import FormatStrFormatter

# カーネルPCA関数を実行(データチューニングパラメータ、次元数を指定)
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color="red", marker="^", alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color="blue", marker="o", alpha=0.5)
ax[1].scatter(
    X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02, color="red", marker="^", alpha=0.5
)
ax[1].scatter(
    X_kpca[y == 1, 0], np.zeros((50, 1)) + 0.02, color="blue", marker="o", alpha=0.5
)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.tight_layout()
plt.show()
