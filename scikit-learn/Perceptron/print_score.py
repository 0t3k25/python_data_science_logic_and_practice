# 性能評価
from sklearn.metrics import accuracy_score

# 分類の正解率を表示
print("Accuracy: %3f" % accuracy_score(y_test, y_pred))
# 余談別メソッドを使用した性能評価
print(f"Accuracy{ppn.score(X_test_std,y_test)}")
