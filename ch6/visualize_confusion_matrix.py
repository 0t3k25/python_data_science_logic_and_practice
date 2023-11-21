# confusion_matrix
from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
# テストと予測のデータから混合行列を生成
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

# 混合行列をpltを使って表示
fig, ax = plt.subplots(figsize=(25, 2.5))
# matshow関数で行列からヒートマップを描画
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")  # 件数を表示
plt.xlabel("Predict label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()
