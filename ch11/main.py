# 擬似データ作成
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=150,  # データ展の総数
    n_features=2,  # 特徴量の個数
    centers=3,  # クラスタの個数
    cluster_std=0.5,  # クラスタ内の標準偏差
    shuffle=True,  # データ点をシャッフル
    random_state=0,  # 乱数生成器の状態を指定
)
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c="white", marker="o", edgecolor="black", s=50)
plt.grid()
plt.tight_layout()
plt.show()

# k-means分類
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3,  # クラスタの個数
    init="random",  # セントロイドの初期値をランダムに選択
    n_init=10,  # 異なるセントロイドの初期値を用いたk-meansアルゴリズムの実装回数
    max_iter=300,  # k-meansアルゴリズム内の最大イテレーション回数
    tol=1e-04,  # 収束と判定するための相対的な許容誤差
    random_state=0,  # セントロイドの初期化に用いる乱数生成器の状態
)
y_km = km.fit_predict(X)  # クラスタ中心の計算と各データ点のインデックスの予測

plt.scatter(
    X[y_km == 0, 0],  # グラフのxの値
    X[y_km == 0, 1],  # グラフのy軸
    s=50,  # プロットのサイズ
    c="lightgreen",  # プロットの色
    edgecolor="black",  # プロットの線の色
    marker="s",  # マーカーの形
    label="Cluster 1",  # ラベル名
)
plt.scatter(
    X[y_km == 1, 0],
    X[y_km == 1, 1],
    s=50,
    c="orange",
    edgecolor="black",
    marker="o",
    label="Cluster 2",
)
plt.scatter(
    X[y_km == 2, 0],
    X[y_km == 2, 1],
    s=50,
    c="lightblue",
    edgecolor="black",
    marker="v",
    label="Cluseter 3",
)
plt.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    s=250,
    marker="*",
    c="red",
    edgecolor="black",
    label="Centroids",
)
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

# elbow method
print(f"Distortion: {km.inertia_}")

distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i,
        init="k-means++",  # k-means++法によりクラスタ中心を選択
        n_init=10,
        max_iter=300,
        random_state=0,
    )
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.tight_layout()
plt.show()

km = KMeans(
    n_clusters=3, init="k-means++", n_init=10, max_iter=300, tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

cluster_labels = np.unique(y_km)  # y_kmの要素の中で重複をなくす
n_clusters = cluster_labels.shape[0]  # 配列の長さを返す
# シルエット係数を計算
silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)  # 色の値をセット
    plt.barh(
        range(y_ax_lower, y_ax_upper),
        c_silhouette_vals,
        height=1.0,
        edgecolor="none",
        color=color,
    )

# 悪いクラスタリングの表示
km = KMeans(
    n_cluster=2, init="k-means++", n_init=10, max_iter=300, tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
plt.scatter(
    X[y_km == 0, 0],
    X[y_km == 0, 1],
    s=50,
    c="lightgreen",
    edgecolor="black",
    marker="s",
    label="Cluster 1",
)

# 階層的クラスタリング
# データ作成
import pandas as pd
import numpy as np

np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
df
# hierachicak clustering
from scipy.spatial.distance import pdist, squareform

# pdistで距離を計算、squareformで対称行列を生成
row_dist = pd.DataFrame(
    squareform(pdist(df, metric="euclidean")), columns=labels, index=labels
)
row_dist

from scipy.cluster.hierarchy import linkage

row_clusters = linkage(pdist(df, metric="euclidean"), method="complete")
row_clusters

pd.DataFrame(
    row_clusters,
    columns=["row label 1", "row label 2", "distance", "no. of items in clust."],
    index=["cluster %d" % (i + 1) for i in range(row_clusters.shape[0])],
)

from scipy.cluster.hierarchy import dendrogram

# 樹形図を黒で表示する場合
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])
row_dendr = dendrogram(
    row_clusters,
    labels=labels,
    #  樹形図を黒で表示する場合
    #  color_threshold=np.inf
)
plt.ylabel("Euclidean distance")
plt.tight_layout()
plt.show()

# print heatmap
fig = plt.figure(figsize=(8, 8), facecolor="white")
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])  # x軸の位置,y軸の位置,幅,高さ
# 注意：matplotlibがv1.5.1以下の場合は、orientation='right'を使うこと
row_dendr = dendrogram(row_clusters, orientation="left")

df_rowclust = df.iloc[row_dendr["leaves"][::-1]]
axd.set_xticks([])
axd.set_yticks([])

# remove axes spines from dendrogram
for i in axd.spines.values():
    i.set_visible(False)

# plot heatmap
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])  # x-pos, y-pos, width, height
cax = axm.matshow(df_rowclust, interpolation="nearest", cmap="hot_r")
fig.colorbar(cax)
axm.set_xticklabels([""] + list(df_rowclust.columns))
axm.set_yticklabels([""] + list(df_rowclust.index))

# plt.savefig('images/11_12.png', dpi=300)
plt.show()

# implement AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(
    n_clusters=3,  # クラスタの個数
    affinity="euclidean",  # 類似度の指数
    linkage="complete",  # 連結方法（ここでは完全連結法）
)
labels = ac.fit_predict(X)
print(f"Cluster labels {labels}")

ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")
labels = ac.fit_predict(X)
print(f"Cluster labels {labels}")

# comapare clustering analysis using half moon data
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=200, noise=0.05, random_state=0  # 生成する点の個数  # データに追加するガウスノイズの標準偏差
)
plt.scatter(X[:, 0], X[:, 1])
plt.tight_layout()
plt.show()

# k-means,agglomerative
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(
    X[y_km == 0, 0],
    X[y_km == 0, 1],
    c="lightblue",
    edgecolor="black",
    marker="o",
    s=40,
    label="cluster 1",
)
ax1.scatter(
    X[y_km == 1, 0],
    X[y_km == 1, 1],
    c="red",
    edgecolor="black",
    marker="s",
    s=40,
    label="cluster 2",
)
ax1.set_title("K-means clustering")
ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")
y_ac = ac.fit_predict(X)
ax2.scatter(
    X[y_ac == 0, 0],
    X[y_ac == 0, 1],
    c="lightblue",
    edgecolor="black",
    marker="o",
    s=40,
    label="cluster 1",
)
ax2.scatter(
    X[y_ac == 1, 0],
    X[y_ac == 1, 1],
    c="red",
    edgecolor="black",
    marker="s",
    s=40,
    label="cluster 2",
)
ax2.set_title("Agglomerative clustering")
plt.legend()
plt.tight_layout()
plt.show()

# DBSCAN
from sklearn.cluster import DBSCAN

db = DBSCAN(
    eps=0.2,  # 隣接点とみなす2点間の最大距離
    min_samples=5,  # ボーダー点の最小個数
    metric="euclidean",  # 距離の計算方法
)
y_db = db.fit_predict(X)
plt.scatter(
    X[y_db == 0, 0],
    X[y_db == 0, 1],
    c="lightgreen",
    edgecolor="black",
    marker="o",
    s=40,
    label="Cluster 1",
)
plt.scatter(
    X[y_db == 1, 0],
    X[y_db == 1, 1],
    c="red",
    edgecolor="black",
    marker="s",
    s=40,
    label="Cluster 2",
)
plt.legend()
plt.tight_layout()
plt.show()
