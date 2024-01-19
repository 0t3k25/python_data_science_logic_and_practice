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
