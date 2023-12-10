# MajorityVoteClassifierを使って多数決でクラスラベルを予測
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ["Majority voting"]
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(
        estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc"
    )
    print(f"ROC AUC: {scores.mean(): .3f} (+/- {scores.std(): .3f}) [{label}]")

# evaluate and tuning ensemble classifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

colors = ["black", "orange", "blue", "green"]
linestyles = [":", "--", "-.", "-"]
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # 陽性クラスのラベルは1である事が前提
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label=f"{label} (auc = {roc_auc: .2f})")
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate(TPr)")
plt.show()


# visualize decision boundary
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product

# 決定領域を描画する最小値、最大値を生成
print(X_train_std[:, 1])
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
# グリッドポイントを生成
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# 描画領域を2行2列に分割
f, axarr = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", figsize=(7, 5))
# 決定領域のプロット、青や赤の散布図の作成などを実行
# 変数idxは各分類器を描画する行と列の位置を表すタプル
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(
        X_train_std[y_train == 0, 0],
        X_train_std[y_train == 0, 1],
        c="blue",
        marker="^",
        s=50,
    )
    axarr[idx[0], idx[1]].scatter(
        X_train_std[y_train == 1, 0], X_train_std[y_train == 1, 1], c="green", s=50
    )
    axarr[idx[0], idx[1]].set_title(tt)
plt.text(
    -3.5, -5.0, s="Sepal width [Standardized]", ha="center", va="center", fontsize=12
)
plt.text(
    -12.5,
    4.5,
    s="Petal length[Standardized]",
    ha="center",
    va="center",
    fontsize=12,
    rotation=90,
)
plt.show()

# access param
mv_clf.get_params()

# grid search
from sklearn.model_selection import GridSearchCV

params = {
    "decisiontreeclassifier__max_depth": [1, 2],
    "pipeline-1__clf__C": [0.001, 0.1, 100.0],
}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring="roc_auc")
grid.fit(X_train, y_train)


# compare roc auc
for r, _ in enumerate(grid.cv_results_["mean_test_score"]):
    print(
        f'{grid.cv_results_["mean_test_score"][r]: .3f} +/- {grid.cv_results_["std_test_score"][r] / 2.0: 0.2f} {grid.cv_results_["params"][r]}'
    )

print(f"Best parameters: {grid.best_params_}")
print(f"Accuracy: {grid.best_score_: .2f}")
