# prepare data
import pandas as pd

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)
df_wine.columns = [
    "Class label",
    "Alcohol",
    "Magic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]
# クラス1を削除
df_wine = df_wine[df_wine["Class label"] != 1]
y = df_wine["Class label"].values
X = df_wine[["Alcohol", "OD280/OD315 of diluted wines"]].values

# split train and test data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Instantiation
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 決定木深さなし
tree = DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=1)
# バギング
# base estimator is decision tree
bag = BaggingClassifier(
    base_estimator=tree,
    n_estimators=500,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    n_jobs=1,
    random_state=1,
)

# compare train and test accuracy
from sklearn.metrics import accuracy_score

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f"Decision tree train/test accuracies {tree_train: .3f} / {tree_test: .3f}")

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print(f"Bagging train/test accuracies {bag_train: .3f}/{bag_test: .3f}")

# visualize decision boundary
import numpy as np
import matplotlib.pyplot as plt

# compare decision boundary
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex="col", sharey="row", figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, bag], ["Decision tree", "Bagging"]):
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
