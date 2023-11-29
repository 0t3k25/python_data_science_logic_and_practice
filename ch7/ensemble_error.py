# 確率質量関数の実装
from scipy.special import comb
import math


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.0))
    probs = [comb(n_classifier, k) * error**k * (1 - error) ** (n_classifier + 1)]
    return sum(probs)


ensemble_error(n_classifier=11, error=0.25)
