# ADALINEの実装
class AdalineGD(object):
    """
    自分が考えるADALINEの流れ
    ①与えられたデータに対して、現在の重みでラベルの値を出してみる
    ②活性化関数を適用ADALINEの場合、特に何もしない(恒等式らしい...)
    ③真ラベルとの誤差を出す(Δと定義)
    ④重みの更新：特徴量の個数分、特徴量x11,x12,x13...と誤差(errors = [e1,e2,e3....])の内積,特徴量x21,x22,z23....と誤差(errors = [e1,e2,e3....])との内積を求める
    ⑤バイアスの重みの更新：バイアスの特徴量は1なのでΔの和
    """

    """ADAptive LInear NEuron分類機

    パラメータ
    ------------
    eta : float
      学習率(0.0より大きく1.0以下の値)
    n_iter : int
      訓練データの訓練回数
    random_state : int
      重みを初期化するための乱数シード
    属性
    ----------
    w_ : 1次元配列
      適合後の重み
    cost_ : リスト
      各エポックでの誤差平方和のコスト関数
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """訓練データに適合させる

        パラメータ
        ------------
        X : {配列のようなデータ構造}, shape= [n_examples, n_features]
          訓練データ
          n_exampleは訓練データの個数、n_featureは特徴量の個数
        y : 配列のようなデータ構造, shape = [n_examples]
          目的変数

        戻り値
        ---------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):  # 訓練回数分まで訓練データを反復
            # net_inputの例[3.5, 3.0, 4.5, 6.0]データ4つの場合
            net_input = self.net_input(X)
            # activationメソッドは単なる恒等関数であるため
            # このコードではなんの効果もないことに注意
            # 直接`output = self.net_input(X)と記述することもできた
            # activationメソッドの目的は、より概念的なものである。
            # つまり、(後ほど説明する)ロジスティック回帰の場合は、
            # ロジスティック回帰の分類機を実装するためにジグモイド関数に変更することもできる
            # activationは活性化関数
            output = self.activation(net_input)
            # ADALINEにおいてφz(i)は総入力
            # それぞれの真ラベルと、それぞれの計算された総入力との誤差 式はy - φ(z(i))
            # 例 errors = y - output = [-1.5, 0, -0.5, -1.0] データが4つの場合
            errors = y - output
            # w1,w2,w3...wmの更新
            # Δwj = ηΣi(y(i) - φ(z(i)))xj
            # X.T.dot(errors)の計算例
            """
            Xの特徴量が二個の場合
            X.T.dot(errors) = [[1, 2, 3, 4], [2, 3, 5, 7]].dot([-1.5, 0, -0.5, -1.0])
                = [-6.5, -11.5]
            """
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算j(w) = 1/2Σ(y(i) - φ(zi)**2
            cost = (errors**2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return X

    def predict(self, X):
        """1ステップごとのクラスラベルを返す"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
