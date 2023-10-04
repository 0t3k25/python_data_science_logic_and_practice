from sklearn.model_selection import train_test_split

# 訓練データとテストデータに分割
# train訓練データ 全体の70% testがテストデータ 全体の30%
# stratifyはそうかサンプリングを行うための引数
# 入力データのクラスラベルの比率が、訓練サブセットとテストサブセットの比率と同じようにするために使用
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
