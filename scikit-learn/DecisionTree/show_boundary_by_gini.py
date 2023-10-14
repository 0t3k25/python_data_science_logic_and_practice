# 深さが4の決定木を訓練する
# ジニ不純度を使用
from sklearn.tree import DecisionTreeClassifier

# ジニ不純度を指標とする決定木のインスタンスを生成
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
# 決定木のモデルを訓練データに適合させる
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(
    X_combined, y_combined, classifier=tree_model, test_idx=range(105, 150)
)
plt.xlabel("petel length[cm]")
plt.ylabel("petal width[cm]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# 決定木モデルを可視化
# 決定木を見ることができる。
from sklearn import tree

tree.plot_tree(tree_model)
plt.show()
