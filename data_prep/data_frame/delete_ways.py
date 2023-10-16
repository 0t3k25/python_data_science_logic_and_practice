# 欠測値を含む行を削除
print(df.dropna())

# 欠測値を含む列を削除
print(df.dropna(axis=1))

# 全てNaNの行だけを削除
print(df.dropna(how="all"))
# 非NaN値が4つ未満の行を削除
print(df.dropna(thresh=4))
# 特定の列(この場合'C')にNaNが含まれている行だけを削除
print(df.dropna(subset=["C"]))
