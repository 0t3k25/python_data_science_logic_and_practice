import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words("english")


def tokenizer(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(D|P)", text.lower())
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, "r", encoding="utf-8") as csv:
        next(csv)  # ヘッダーを読み飛ばす
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


next(stream_docs(path="/content/movie_data.csv"))


# get doc_stream and return docs
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(
    decode_error="ignore", n_features=2**21, preprocessor=None, tokenizer=tokenizer
)
clf = SGDClassifier(loss="log", random_state=1)
doc_stream = stream_docs(path="/content/movie_data.csv")
