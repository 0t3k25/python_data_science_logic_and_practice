from sklearn.linear_model import SGDClassifier

# 確率的勾配降下法バージョンのパーセプトロン生成
ppn = SGDClassifier(loss="perceptron")
# 確率的勾配降下法バージョンのロジスティック回帰を生成
lr = SGDClassifier(loss="log")
# 確率的勾配降下法バージョンのSVM(損失関数=ヒンジ関数)を生成
svm = SGDClassifier(loss="hinge")
