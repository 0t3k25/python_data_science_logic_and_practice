# 不均衡なデータセットの作成
x_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))
# 多数決法による予測の正解率
y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100
from sklearn.utils import resample

print(f"Number of class 1 examples before: {x_imb[y_imb==1].shape[0]}")

# データ点の個数がクラス0と同じになるまで新しいデータ点を復元抽出
X_upsampled, y_upsampled = resample(
    x_imb[y_imb == 1],
    y_imb[y_imb == 1],
    replace=True,
    n_samples=x_imb[y_imb == 0].shape[0],
    random_state=123,
)

print(f"Number of class 1 examples after: {X_upsampled.shape[0]}")


# サンプルを追加
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))
# 多数決法による予測の正解率
y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100
