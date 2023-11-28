# 他クラス分類における性能評価の指定
pre_score = make_scorer(
    score_func=precision_score, pos_label=1, greater_is_better=True, average="micro"
)
