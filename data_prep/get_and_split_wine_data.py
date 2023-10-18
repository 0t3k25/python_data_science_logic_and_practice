# wineデータに関して特徴量が多いため、前処理で特徴量を減らす処理
import pandas as pd

# wineデータを読み込む
df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)
# 列名を設定
df_wine.columns = [
    "Class label",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]
# クラスラベルを設定
print("Class labels", np.unique(df_wine["Class label"]))
df_wine.head()

# 訓練データとテストデータに分割
from sklearn.model_selection import train_test_split

# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 30%をテストデータとして割り当てているtest_size=0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)
