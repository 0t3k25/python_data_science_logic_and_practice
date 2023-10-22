# ランダムフォレストを用いて特徴量の重要性を可視化
from sklearn.ensemble import RandomForestClassifier

# Wineデータセットの特徴量の名称取得
feat_labels = df_wine.columns[1:]
# ランダムフォレスオブジェクトの生成（決定木の個数=500）
forest = RandomForestClassifier(n_estimators=500, random_state=1)
# モデルを適合
forest.fit(X_train, y_train)
# 特徴量の重要度を抽出
importances = forest.feature_importances_

# 重要度の降順で特徴量のインデックスを抽出
# スライス::-1は降順を作るために使用
indices = np.argsort(importances)[::-1]
# print(importances)
# print(importances[indices])
# 重要度の降順で特徴量の名称、重要度を表示
for f in range(X_train.shape[1]):
    print(
        "%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
    )
plt.title("Feature importances")
# 棒グラフの設定
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
#
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# 推測後しきい値の閾値をきめることにより特徴量をさらに軽減
# 中間ステップとして利用
from sklearn.feature_selection import SelectFromModel

# 特徴量選択オブジェクトを生成（重要度のしきい値を0.1に設定）
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
# 特徴量を抽出
X_selected = sfm.transform(X_train)
print("Number of features that meet this thershold criterion:", X_selected.shape[1])

for f in range(X_selected.shape[1]):
    print(
        "%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
    )
