# グリッドサーチを使って最適なハイパーパラメータを求める
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
    {"svc__C": param_range, "svc__kernel": ["linear"]},
    {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]},
]
# ハイパーパラメータ値のリストparam_gridを指定し
# グリッドサーチを行うGridSearchCV
gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring="accuracy",
    cv=10,
    refit=True,
    n_jobs=-1,
)
gs = gs.fit(X_train, y_train)
# モデルの最良スコアを出力
print(gs.best_score_)

print(gs.best_params_)
# テストデータの正解率確認
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f"test accuraby {clf.score(X_test,y_test) : 3f}")
