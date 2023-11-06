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
