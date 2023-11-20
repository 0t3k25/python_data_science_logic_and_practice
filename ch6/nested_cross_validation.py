# nested cross-validation
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring="accuracy", cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)
print(f"CV accuracy: {np.mean(score) :3f} +/- {np.std(scores) :3f}")
from sklearn.tree import DecisionTreeClassifier

# ハイパーパラメータ値として決定機の深さパラメータを指定し、←決定機の深さをチューニングしていない
# グリッドサーチを行うGridSearchCVクラスをインスタンス化
gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[{"max_depth": [1, 2, 3, 4, 5, 6, 7, None]}],
    scoring="accuracy",
    cv=2,
)
scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)
print(f"CV accuracy: {np.mean(scores) : 3f} +/- {np.std(scores):3f}")
