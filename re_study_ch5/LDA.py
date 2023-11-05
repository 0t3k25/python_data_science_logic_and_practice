# LDAの処理の流れ
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    # 行方向（それぞれの特徴量に関して）の平均取得
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f"MV {label}: {mean_vecs[label-1]}\n")
d = 13
S_W = np.zeros((d, d))
