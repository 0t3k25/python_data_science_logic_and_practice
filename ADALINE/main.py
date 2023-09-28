class AdalineGD(object):
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
