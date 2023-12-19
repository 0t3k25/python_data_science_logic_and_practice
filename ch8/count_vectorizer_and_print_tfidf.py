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
