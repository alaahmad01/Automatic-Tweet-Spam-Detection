import string
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv("train.csv")
df = df.dropna(axis=0, subset=['following', 'followers', 'is_retweet', 'Tweet', 'actions', 'Type'])
df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(
    r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
    " ", x))
df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
df['Tweet'] = df['Tweet'].apply(lambda x: x.lower())
df = df.drop_duplicates(keep='first')

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(df['Tweet'])

pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", SGDClassifier()),
    ]
)

parameters = {
    "vect__max_df": (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    "clf__max_iter": (200,),
    "clf__alpha": (0.00001, 0.000001),
    "clf__penalty": ("l2", "elasticnet"),
    # 'clf__max_iter': (10, 50, 80),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

grid_search.fit([_ for _ in df.Tweet], [1 if x == 'Spam' else 0 for x in df.Type])

grid_search.predict(['The Sanders campaign appears to say that this tension/chaos/unrest out of NV isn\'t its problempic.twitter.com/k1L8CmYbq5'])

