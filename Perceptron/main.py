# パーセプトロンの分類機クラス作成
import numpy as np


class Perceptron(object):
    """パーセプトロンの分類機
    パラメータ
    --------------
    eta : float
      学習率(0.0 より大きく1.0以下の値)
    n_iter : int
      訓練データの訓練回数
    random_state : int
    重みを初期化するための乱数シード
    属性
    --------------
    w_ : 1次元配列
      適合後の重み
    errors_ : リスト
      各エポックでの誤分類(更新)の数

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """訓練データに適合させる

        パラメータ
        -------------
        X : {配列のようなデータ構造}, shape = [n_examples, n_features]
          訓練データ
          n_exampleは訓練データの個数、n_featuresは特徴量の個数
        y : 配列のようなデータ構造、 shape = [n_examples]
          目的変数

        戻り値
        ----------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):  # 訓練回数分まで訓練データを反復
            errors = 0
            for xi, target in zip(X, y):
                # 重みw1...,wmの更新
                # Δwj = 学習率*(yi-y(i))*xj(i)
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重みw0の更新
                self.w_[0] += update
                # 重みの更新が0でない場合は誤分類としてカウント
                errors += int(update != 0.0)
            # 反復後との誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """総入力を計算
        総入力関数
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
