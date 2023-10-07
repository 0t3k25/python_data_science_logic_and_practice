# Implement Logistic Regression
class LogisticRegressionGD(object):
    """勾配降下法に基づくロジスティック回帰分類機

    パラメータ
    -------------
    eta : float
      学習率(0.0より大きく1.0以下の値)
    n_iter : int
      訓練データの訓練回数
    rando_state : int
      重みを初期化するための乱数シード

    属性
    ------------
    w_ : 1次元配列
    　適合後の重み
    cost_ : リスト
      各エポックでのロジスティックコスト関数
    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        # 学習率の初期化、訓練回数の初期化、乱数シードを固定にするrandom_state
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """訓練データに適合させる

        パラメータ
        -----------
        X : {配列のような構造}, shape = [n_examples, n_features]
          訓練データ
        y : 配列のようなデータ構造, shape = [n_examples]
          目的変数

        戻り値
        --------
        self : object
        """

        rgen = np.random.RandomState(self.random_state)
        # 特徴量の数分重みを乱数から生成
        # sizeのところで生成する乱数の数を指定、1+はバイアス項
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
