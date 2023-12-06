# irisデータセット取得
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
# 50%をテストデータに分類
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1, stratify=y
)
# ロジスティック回帰
# 決定木分類器
# k最近傍法分類器
# を使用
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(penalty="l2", C=0.001, solver="lbfgs", random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")
pipe1 = Pipeline([["sc", StandardScaler()], ["clf", clf1]])
pipe3 = Pipeline([["sc", StandardScaler()], ["clf", clf3]])
clf_labels = ["Logistic regression", "Decision tree", "KNN"]
print("10-fold criss validation:\n")
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(
        estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc"
    )
    # 予測性能の出力
    print("ROC AUC %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
