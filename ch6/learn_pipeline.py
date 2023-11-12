import pandas as pd

df = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    header=None,
)

from sklearn.preprocessing import LabelEncoder

# 行,列
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
# print(y)
# print(df)
le = LabelEncoder()
# 0と1に変換
y = le.fit_transform(y)
# print(y)
# print(le.classes_)
# MとBに変換
# print(le.transform(['M','B']))
# データセットの分割
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=1
)
# 変換ステップなどを結合
# パイプラインを使用して標準化、PCA、学習を結合
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 連結する処理としてスケーリング、主成分分析、ロジスティック回帰を指定
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(random_state=1, solver="lbfgs"),
)
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print(f"Test Acciracu:{pipe_lr.score(X_test,y_test)}")
