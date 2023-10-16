from sklearn.impute import SimpleImputer
import numpy as np

# 欠測値補完のインスタンスを生成(平均値補完)
imr = SimpleImputer(missing_values=np.nan, strategy="mean")
# データを適合
imr = imr.fit(df.values)
# 補完を実行
imputed_data = imr.transform(df.values)
print(imputed_data)
