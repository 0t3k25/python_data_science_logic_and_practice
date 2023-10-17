# クラスラベルのエンコーディング
import numpy as np

df = pd.DataFrame(
    [
        ["green", "M", "10.1", "class2"],
        ["red", "L", "13.5", "class1"],
        ["blue", "XL", "15.3", "class2"],
    ]
)
df.columns = ["color", "size", "price", "classlabel"]

print(np.unique(df["classlabel"]))
# クラスラベルと整数を対応させるディクショナリを生成
class_mapping = {label: idx for idx, label in enumerate(np.unique(df["classlabel"]))}
# クラスラベルを整数に変換
df["classlabel"] = df["classlabel"].map(class_mapping)
print(class_mapping.items())
# 整数とクラスラベルを対応させるディクショナリを生成
inv_class_mapping = {v: k for k, v in class_mapping.items()}
print(inv_class_mapping)
df["classlabel"] = df["classlabel"].map(inv_class_mapping)
print(df)

from sklearn.preprocessing import LabelEncoder

# ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df["classlabel"].values)
print(y)
class_le.inverse_transform(y)
