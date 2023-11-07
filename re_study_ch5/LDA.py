# LDAの処理の流れ
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    # 行方向（それぞれの特徴量に関して）の平均取得
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f"MV {label}: {mean_vecs[label-1]}\n")
d = 13
# クラス内変動行列の計算
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        # 変動行列Siを合計
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
# クラス内変動行列の次元
print(f"Within-class scatter matrix: {S_W.shape[0]}x{S_W.shape[1]}")
# クラスのラベル存在数
print(f"Class label distribution: {np.bincount(y_train)[1:]}")

# クラスラベルが一様に分布していないためスケーリング実行
# クラス内変動行列
d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print(f"Scaled within-class scatter matrix {S_W.shape[0]}x{S_W.shape[1]}")
# クラス間変動行列の計算
mean_overall = np.mean(X_train_std, axis=0)
print(mean_overall)
d = 13  # 特徴量の個数
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    # 各クラスに属するサンプル数を取得
    n = X_train_std[y_train == i + 1, :].shape[0]
    mena_vec = mean_vec.reshape(d, 1)  # 列ベクトルを作成
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print(f"Between-class scatter matrix: {S_B.shape[0]}x{S_B.shape[1]}")
# inv関数で逆行列、dot関数で行列積、eig関数で固有値を計算
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print(f"Eigenvalues in descending order:\n")
for eigen_val in eigen_pairs:
    print(eigen_val[0])
# 固有値の総和を求める
tot = sum(eigen_vals.real)
# 分散説明率とその累積和を計算
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(
    range(1, 14), discr, alpha=0.5, align="center", label='Individual "discriminability'
)
plt.step(range(1, 14), cum_discr, where="mid", label='Cumlative "discriminability')
plt.ylabel('"Discriminability" ratio')
plt.xlabel("Linear Discriminants")
plt.ylim([-0.1, 1.1])
plt.legend(loc="best")
plt.tight_layout()
plt.show()
