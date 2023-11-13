import numpy as np
from sklearn.model_selection import StratifiedKFold

# 分割元データ、分割数、乱数生成機の状態を指定し、
# 層化k分割交差検証イテレータを表すStratifiedKFoldクラスのインスタンス化
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
# イテレータのインデックスと要素をループ処理：（上から順に）
#   データをモデルに適合
#   テストデータの正解率を算出
#   リストに正解率を追加
#   分割の番号、0　以上の要素数、正解率を出力

for k, (train, test) in enumerate(kfold):
    print(train)
    print("train")
    print(test)
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(f"Fold: {k+1}, Class dist.: {np.bincount(y_train[train])}, Acc:{score:.3f}")
# 正解率の平均と標準偏差を出力
print(f"\nCV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

# 層化k分割交差検証を使ったモデルの評価
# より簡単に評価を出すことができる
from sklearn.model_selection import cross_val_score

# 交差検証のcross_val_score缶sぬうでモデルの正解率を算出
# 推定機esrimator、訓練データX、予測値y、分割数cv、CPU数n_jobsを指定
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print(f"CV accuracy scores:{scores}")
print(f"CV accuracy:{np.mean(scores):.3f} +/- {np.std(scores):.3f}")
