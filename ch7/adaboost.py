# adapt adboost using scikit-learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

tree = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=1)
ada = AdaBoostClassifier(
    base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1
)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f"Dexision tree train/test accuracies {tree_train: .3f} {tree_test: .3f}")

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print(f"AdaBoost train/test accuracies {ada_train: .3f} / {ada_test: .3f}")

import numpy as np
import matplotlib.pyplot as plt

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2, sharex="col", sharey="row", figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, ada], ["Decision tree", "AdaBoost"]):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(
        X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="blue", marker="^"
    )
    axarr[idx].scatter(
        X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="green", marker="o"
    )
    axarr[idx].set_title(tt)

axarr[0].set_ylabel("Alcohol", fontsize=12)
plt.tight_layout()
plt.text(
    0,
    -0.2,
    s="OD280/OD315 of diluted wines",
    ha="center",
    va="center",
    fontsize=12,
    transform=axarr[1].transAxes,
)
plt.show()
