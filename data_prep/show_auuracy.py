knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker="o")
plt.ylim([0.7, 1.02])
plt.ylabel("Accuracy")
plt.xlabel("Number of features")
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
plt.show()


k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])


knn.fit(X_train, y_train)
print("Training accuracy:", knn.score(X_train_std, y_train))
print("Test accuracy:", knn.score(X_test_std, y_test))
