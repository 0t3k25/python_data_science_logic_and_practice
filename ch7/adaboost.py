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
