import numpy as np
import sys


class NeuralNetMLP(object):
    """フィードフォワードニューラルネットワーク/多層パーセプトロン分類器
    パラメータ
    ----------
    n_hidden: int(デフォルト:30)
      隠れユニットの個数
    l2:float(デフォルト:0.)
      L2正則化のλパラメータ
      l2=0の場合は正則化なし(default)
    epochs:int(デフォルト:100)
      訓練の回数
    eta:float(デフォルト:0.001)
      学習率
    shuffle:bool(デフォルト:True)
      Trueの場合、循環を避けるためにエポックごとに訓練データをシャッフル
    mini_batch_size:int(デフォルト:1)
      ミニバッチあたりの訓練データの個数
    seed:int(デフォルト:None)
      重みとシャッフルを初期化するための乱数シード

    属性
    -----
    eval_: dict
      訓練のエポックごとに、コスト、訓練の正解率、検証の正解率を収集するディクショナリ
    """

    def __init__(
        self,
        n_hidden=30,
        l2=0.0,
        epochs=100,
        eta=0.001,
        shuffle=True,
        minibatch_size=1,
        seed=None,
    ):
        """NeuralNetMLPの初期化"""
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """ラベルをone-hot表現にエンコード
        パラメータ
        ----------
        y:array, shape = [n_examples]
          目的変数の値

        戻り値
        ----------
        onehot: array, shape = (n_examples, n_labels)
        """

        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.0
        return onehot.T

    def _sigmoid(self, z):
        """ロジスティック関数（シグモイド関数）を計算"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """フォワードプロパゲーションのステップを計算"""
        # ステップ1:隠れ層の総入力
        # [n_examples, n_features] dot [n_features,n_hidden]
        # -> [n_examples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h
        # ステップ2:隠れ層の総入力
        a_h = self._sigmoid(z_h)

        # ステップ3:出力層の総入力
        # [n_examples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_examples, n_classlabels]
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # ステップ4：出力層の活性化関数
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """コスト関数を計算
        パラメータ
        ----------
        y_enc: array,shape = (n_examples, n_labels)
          one-hot表現にエンコードされたクラスラベル
        output:array,shpe = [n_examples, n_output_units]
          出力層の活性化関数（フォワードプロパゲーション）

        戻り値
        -------
        cost:float
          正則化されたコスト

        """

        L2_term = self.l2 * (np.sum(self.w_h**2.0) + np.sum(self.w_out**2.0))

        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2) + L2_term

        #
        return cost

    def predict(self, X):
        """クラスラベルを予測

        parameter
        ----------
        X:array, shape = [n_examples,n_features]
          元の特徴量が設定された入力値

        戻り値
        --------
        y_pred: array,shape = [n_examples]
          予測されたクラスラベル

        """

        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)

        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """訓練データから重みを学習
        パラメータ
        ----------
        X_train:array,shape=[n_examples,n_features]
          元の特徴量が設定された入力層
        y_train:array,shape = [n_examples]
          目的変数（クラスラベル）
        X_valid:array,shape = [n_examples, n_features]
          訓練次の検証に使うサンプル特徴量
        y_valid:array,shape = [n_examples]
          訓練時の検証に使うサンプルラベル

        戻り値:
        -------
        self

        """
        # クラスラベルの個数
        n_output = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]

        ######
        # 重みの初期化
        #####

        # 入力層 ->隠れ層の重み
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(
            loc=0.0, scale=0.1, size=(n_features, self.n_hidden)
        )

        # 隠れ層 -> 出力層の重み
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(
            loc=0.0, scale=0.1, size=(self.n_hidden, n_output)
        )

        # 書式設定
        epoch_strlen = len(str(self.epochs))
        self.eval_ = {"cost": [], "train_acc": [], "valid_acc": []}

        y_train_enc = self._onehot(y_train, n_output)

        # エポック数だけ訓練を繰り返す
        for i in range(self.epochs):
            # ミニバッチの反復処理(イテレーション)
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(
                0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size
            ):
                batch_idx = indices[start_idx : start_idx + self.minibatch_size]

                # フォワードプロパゲーション
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                #
                # バックプロパゲーションアルゴリズム
                #

                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[batch_idx]

                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1.0 - a_h)

                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden]
                #  ->[n_examples, n_hidden]
                delta_h = np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h

                #  [n_feeatures,n_examples] dot [n_examples, n_classlabels]
                # ->[n_features,n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # [n_hidden, n_examples] dot [n_examples, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # 正則化と重みの更新
                delta_w_h = grad_w_h + self.l2 * self.w_h
                delta_b_h = grad_b_h  # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = grad_w_out + self.l2 * self.w_out
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #
            # 評価
            #

            # イテレーションごとに評価を行う
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = (np.sum(y_train == y_train_pred)).astype(float) / X_train.shape[
                0
            ]
            valid_acc = (np.sum(y_valid == y_valid_pred)).astype(float) / X_valid.shape[
                0
            ]

            sys.stderr.write(
                "\r%0*d/%d | Cost:%.2f"
                "|Train/Valid Acc.:%.2f%%/%.2f%%"
                % (
                    epoch_strlen,
                    i + 1,
                    self.epochs,
                    cost,
                    train_acc * 100,
                    valid_acc * 100,
                )
            )
            sys.stderr.flush()

            self.eval_["cost"].append(cost)
            self.eval_["train_acc"].append(train_acc)
            self.eval_["valid_acc"].append(valid_acc)
        return self


# インスタンス作成
nn = NeuralNetMLP(
    n_hidden=100,
    l2=0.01,
    epochs=200,
    eta=0.0005,
    minibatch_size=100,
    shuffle=True,
    seed=1,
)

# deeplearning訓練
nn.fit(
    X_train=X_train[:55000],
    y_train=y_train[:55000],
    X_valid=X_train[55000:],
    y_valid=y_train[55000:],
)

import matplotlib.pyplot as plt

plt.plot(range(nn.epochs), nn.eval_["cost"])
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.show()

plt.plot(range(nn.epochs), nn.eval_["train_acc"], label="Training")
plt.plot(range(nn.epochs), nn.eval_["valid_acc"], label="Validatiion", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(loc="lower right")
plt.show()

# print accuracy
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred).astype(float) / X_test.shape[0]
print(f"Test accuracy {acc*100: .2f}")

#
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap="Greys", interpolation="nearest")
    ax[i].set_title(f"{i+1}) t: {correct_lab[i]} p: {miscl_lab[i]}")
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
