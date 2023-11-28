# ROC曲線のプロット
from sklearn.metrics import roc_curve, auc
from scipy import interp

# スケーリング、主成分分析、ロジスティック回帰を指定して、Pieplineクラスをインスタンス化
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(penalty="l2", random_state=1, solver="lbfgs", C=100.0),
)
# 2つの特徴量を抽出
X_train2 = X_train[:, [4, 14]]
# 層化k分割交差検証イテレータを表すStratifiedKFoldクラスをインスタンス化
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
# 0から1までの間で100個の要素を生成
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    # predict_probaメソッドで確率を予測、fitメソッドでモデルに適合させる
    probas = pipe_lr.fit(
        X_train2[train],
        y_train[train],
    ).predict_proba(X_train2[test])
    # roc_curve関数でROC曲線の性能を計算してプロット
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)  # FPR(X軸)とTPR（Y軸）を線形補間
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"RoC fold {i+1} (area= {roc_auc:2f})")
# 当て推量をプロット
plt.plot([0, 1], [0, 1], linestyle="--", color=(0.6, 0.6, 0.6), label="Random guessing")
# FPR, TPR, ROC AUCそれぞれの平均を計算してプロット
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, "k--", label=f"Mean ROC (area = {mean_auc:2f})", lw=2)
# 完全に予測が正解した時のROC曲線をプロット
plt.plot(
    [0, 0, 1], [0, 1, 1], linestyle=":", color="black", label="Perfect performance"
)
# グラフの各項目を指定ー
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
