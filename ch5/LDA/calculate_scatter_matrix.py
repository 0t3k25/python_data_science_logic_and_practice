# 平均ベクトルの取得
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    # print('MV %s: %s\n' %(label,mean_vecs[label-1]))

# クラス内変動行列を取得
d = 13  # 特徴量の個数
S_W = np.zeros((d, d))

for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print("Within-class scatter matrix: %sx%s" % (S_W.shape[0], S_W.shape[1]))
# クラスラベルの個数を取得
print(y_train)
# 0を排除するため1:
print(f"Class label distribution {np.bincount(y_train)[1:]}")
# クラス内変動行列計算
# クラスラベルの個数が違う為、スケーリングを実施
d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    # 共分散行列がデータのばらつきを正規化して考慮する性質を持っている
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print(f"Scaked wuthin-class scatter matrix: {S_W.shape[0]}x{S_W.shape[1]}")

# クラス間変動行列計算
mean_overall = np.mean(X_train_std, axis=0)
d = 13  # 特徴量の個数
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    # print(mean_vec)
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print(f"Between-class scatter matrix: {S_B.shape[0]}x{S_B.shape[1]}")
