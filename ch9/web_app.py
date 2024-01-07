# serialize text
import pickle
import os

dest = os.path.join("movieclassifier", "pkl_objects")
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop, open(os.path.join(dest, "stopwords.pkl"), "wb"), protocol=4)
pickle.dump(clf, open(os.path.join(dest, "classifier.pkl"), "wb"), protocol=4)

# deserialize
import pickle
import re
import os
from vectorizer import vect

clf = pickle.load(open(os.path.join("pkl_objects", "classifier.pkl"), "rb"))

import numpy as np

label = {0: "negative", 1: "positive"}
example = ["I love this movie. It's amazing."]
X = vect.transform(example)
print(
    f"Prediction{label[clf.predict(X)[0]]} \nProbability: {np.max(clf.predict_proba(X))*100}"
)
