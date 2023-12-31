# evaluate word importance
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array(
    [
        "The sun is shining",
        "The weather is sweet",
        "The sun is shining, the weather is sweet, and one and one is two",
    ]
)
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

df.loc[0, "review"][-50:]

import re


def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + "".join(emoticons).replace("-", "")
    return text


# クレンジング適応
df["review"] = df["review"].apply(preprocessor)


# テキストをトークン化
def tokenizer(text):
    return text.split()


tokenizer("runners like running and thus they run")

# porterステミングアルゴリズム
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tokenizer_porter("runners like running and thus they run")

# remove stop word
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

stop = stopwords.words("english")
[
    w
    for w in tokenizer_porter("a runner likes running and runs a lot")[-10:]
    if w not in stop
]
# train text classfier logisitc regression
X_train = df.loc[:25000, "review"].values
y_train = df.loc[:25000, "sentiment"].values
X_test = df.loc[25000:, "review"].values
y_test = df.loc[25000:, "sentiment"].values

# seek logistic regression parameter
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [(stop, None)],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [1.0, 10.0, 100.0],
    },
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [stop, None],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "vect__use_idf": [False],
        "vect__norm": [None],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [1.0, 10.0, 100.0],
    },
]
lr_tfidf = Pipeline(
    [("vect", tfidf), ("clf", LogisticRegression(random_state=0, solver="liblinear"))]
)
gs_lr_tfidf = GridSearchCV(
    lr_tfidf, param_grid, scoring="accuracy", cv=5, verbose=2, n_jobs=-1
)
gs_lr_tfidf.fit(X_train, y_train)

print(f"Best parameter set :{gs_lr_tfidf.best_params}")

print(f"CV Accuracy : {gs_lr_tfidf.best_score_:.3f}")

clf = gs_lr_tfidf.best_estimator_
print(f"Test Accuracy : f{clf.score(X_test,y_test):.3f}")
