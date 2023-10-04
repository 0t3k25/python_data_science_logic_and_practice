# 入力データのクラスラベルの比率が、訓練サブセットとテストサブセットの比率と同じになっているか確認
# それぞれのラベルの個数が表示される
print("Label counts in y:", np.bincount(y))
print("Label counts in y_train:", np.bincount(y_train))
print("Label counts in y_test", np.bincount(y_test))
