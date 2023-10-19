# min-maxスケーリング
from sklearn.preprocessing import MinMaxScaler

# min-maxスケーリングのインスタンスを生成
mms = MinMaxScaler()
# 訓練データをスケーリング
X_train_norm = mms.fit_transform(X_train)
# テストデータをスケーリング
X_test_norm = mms.transform(X_test)
# print(X_train_norm)
# print(X_test_norm)

# 0-5の値に関して標準化と正規化
ex = np.array([0, 1, 2, 3, 4, 5])
print("standardized:", (ex - ex.mean()) / ex.std())
print("normalized:", (ex - ex.min()) / (ex.max() - ex.min()))

from sklearn.preprocessing import StandardScaler

# 標準化のインスタンスを生成（平均=0、標準偏差=1に変換）
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
# print(X_train_std)
# print(X_test_std)
