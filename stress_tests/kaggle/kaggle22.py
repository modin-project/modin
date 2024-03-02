import matplotlib

matplotlib.use("PS")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # CountVectorizer
from sklearn.linear_model import LogisticRegression

import modin.pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
subm = pd.read_csv("sample_submission.csv")
train.head()
train["comment_text"][0]
train["comment_text"][2]
lens = train.comment_text.str.len()
lens.mean(), lens.std(), lens.max()
lens.hist()
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train["none"] = 1 - train[label_cols].max(axis=1)
train.describe()
len(train), len(test)
COMMENT = "comment_text"
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
import re
import string

re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")


def tokenize(s):
    return re_tok.sub(r" \1 ", s).split()


n = train.shape[0]
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    tokenizer=tokenize,
    min_df=3,
    max_df=0.9,
    strip_accents="unicode",
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1,
)
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])
trn_term_doc, test_term_doc


def pr(y_i, y):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


x = trn_term_doc
test_x = test_term_doc


def get_mdl(y):
    y = y.values
    r = np.log(pr(1, y) / pr(0, y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


preds = np.zeros((len(test), len(label_cols)))
for i, j in enumerate(label_cols):
    print("fit", j)
    m, r = get_mdl(train[j])
    preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]
submid = pd.DataFrame({"id": subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv("submission.csv", index=False)
