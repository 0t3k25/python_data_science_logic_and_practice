# SVMを使用してirisデータを分類
from sklearn.svm import SVC

# 線形SVMのインスタンスを生成
svm = SVC(kernel="linear", C=1.0, random_state=1)
# 線形SVMのモデルに訓練データを適合させる
svm.fit(X_train_std, y_train)
plot_decision_regions(
    X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150)
)
plt.xlabel("petal length[standardized]")
plt.ylabel("petal width[standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
