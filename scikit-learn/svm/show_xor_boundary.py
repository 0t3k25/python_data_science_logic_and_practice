from sklearn.svm import SVC

# RBFカーネルによるSVMのインスタンスを生成
svm = SVC(kernel="rbf", random_state=1, gamma=0.1, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
