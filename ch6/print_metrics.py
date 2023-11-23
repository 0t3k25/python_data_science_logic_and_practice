# 適合率、再現率、F1スコアを出力
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

print(f"Precision: {precision_score(y_true=y_test,y_pred=y_pred) : 3f}")
print(f"Recall: {recall_score(y_true=y_test, y_pred=y_pred)}")
print(f"F1: {f1_score(y_true=y_test,y_pred=y_pred)}")

# カスタム性の指標設定出力
from sklearn.metrics import make_scorer, f1_score

c_gamma_range = [0.01, 0.1, 1, 10.0]
param_grid = [
    {"svc__C": c_gamma_range, "svc__kernel": ["linear"]},
    {"svc__C": c_gamma_range, "svc__gamma": c_gamma_range, "svc__kernel": ["rbf"]},
]
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(
    estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10, n_jobs=-1
)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
