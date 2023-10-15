from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(
    tree_model,
    filled=True,
    rounded=True,
    class_names=["Setosa", "Versicolor", "Virginica"],
    feature_names=["petal length", "petal width"],
    out_file=None,
)
graph = graph_from_dot_data(dot_data)
graph.write_png("tree.png")
