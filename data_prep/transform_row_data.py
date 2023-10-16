import pandas as pd

# サンプルデータを生成(Tシャツの色、サイズ、価格、クラスラベル)
df = pd.DataFrame(
    [
        ["green", "M", "10.1", "class2"],
        ["red", "L", "13.5", "class1"],
        ["blue", "XL", "15.3", "class2"],
    ]
)
# 列名を設定
df.columns = ["color", "size", "price", "classlabel"]
print(df)
# Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {"XL": 3, "L": 2, "M": 1}
# Tシャツのサイズを整数に変換
df["size"] = df["size"].map(size_mapping)
print(df["size"])
print(df)
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(inv_size_mapping)
