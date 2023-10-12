# カーネル関数のγパラメータに関して
# 訓練データの値がより大きくなるため、決定境界がより狭くなる
# iris-dataで実験
# RBFカーネルによるSVMのインスタンスを生成(2つのパラメータを変更)
# 決定境界がかなり滑らかになっている。
svm = SVC(kernel="rbf", random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel("petal lengt[standardized]")
plt.ylabel("petal width[standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# RBFカーネルによるSVMのインスタンスを生成(γパラメータを変更)
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length[standardized]')
plt.ylabel('petal length[standardized]')h
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()