# ロジスティック回帰において、L1正則化を使用していらない特徴量を削除する
from sklearn.linear_model import LogisticRegression

# L1正則化ロジスティック回帰のインスタンスを生成
LogisticRegression(penalty=1, solver="liblinear", multi_class="ovr")
# L1正則化ロジスティック回帰のインスタンスを生成：逆正則化パラメータC=1.0はデフォルト値であり、
# 値を大きくしたり小さくしたりすると、正則化の効果を強めたり弱めたりできる
lr = LogisticRegression(penalty="l1", solver="liblinear", multi_class="ovr")
# 訓練データに適合
lr.fit(X_train_std, y_train)
# 訓練データに対する正解率の表示
print("Training accuracy:", lr.score(X_train_std, y_train))

# テストデータに対する正解率の表示
print("Test accuracy:", lr.score(X_test_std, y_test))
# 1対他モデルアプローチ
# 3つの切片が表示される。それぞれのクラスに対する切片1つ目がクラス2,3、2つ目がクラス1,3...
print(lr.intercept_)

# 重み係数の表示
print(lr.coef_)
