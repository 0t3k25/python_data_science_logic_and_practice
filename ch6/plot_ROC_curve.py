# ROC曲線のプロット
from sklearn.metrics import roc_curve, auc
from scipy import interp

# スケーリング、主成分分析、ロジスティック回帰を指定して、Pieplineクラスをインスタンス化
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(penalty="l2", random_state=1, slover="lbfgs", C=100.0),
)
#
