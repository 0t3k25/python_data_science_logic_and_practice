# inv関数で逆行列、dot関数で行列積、eig関数で固有値を計算
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print("Eigenvalues in descending order:\n")
for eigen_val in eigen_pairs:
    print(eigen_val[0])
# 線形判別をプロット
# 固有値の実巣部の総和を求める
tot = sum(eigen_vals.real)
# 分散説明率とその累積和を計算
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discur = np.cumsum(discr)
plt.bar(
    range(1, 14),
    discr,
    alpha=0.5,
    align="center",
    label='Individual "discriminability"',
)
plt.step(range(1, 14), cum_discr, where="mid", label='cumulative "discriminability"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel("Linear Discriminants")
plt.ylim([-0.1, 1.1])
plt.legend(loc="best")
plt.tight_layout()
plt.show()
# 2つの固有ベクトルから変換行列を作成
w = np.hstack(
    (eigen_paris[0][1][:, np.newaxis].real, eigen_paris[1][1][:, np.newaxis].real)
)
print("MatrixW: \n", w)
