# 予測モデルの作成
from sklearn.linear_model import Perceptron

# 一対他を使っているため3つの品種を分類可能
# エポック数40,学習率0.1でパーセプトロンのインスタンスを作成
ppn = Perceptron(eta0=0.01, random_state=1)
# 標準化された訓練データと訓練データのクラスラベルをモデルに適合させる
ppn.fit(X_train_std, y_train)
