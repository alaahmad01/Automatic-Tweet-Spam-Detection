import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from learn_pipe import LearnPipe


def text_learning(Tweet, Type):
    Tweet = Tweet.apply(lambda x: re.sub(
        r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
        '', x))
    Tweet = Tweet.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    Tweet = Tweet.apply(lambda x: x.lower())

    X = [_ for _ in Tweet]
    Y = [1 if x == 'Spam' else 0 for x in Type]

    # split the data so it trains on the first third
    # and predicts the other two thirds
    X_train = X[:int(len(X) * (1 / 3))]
    Y_train = Y[:int(len(X) * (1 / 3))]
    X_test = X[int(len(X) * (1 / 3)):]
    Y_test = Y[int(len(X) * (1 / 3)):]

    pipe = LearnPipe(
        "NLP Neural Network",
        Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1,
                                      max_iter=5000)),
            ]
        ),
        [X_train, X_test, Y_train, Y_test]
    )

    pipe.learn_and_test()

    return pipe.predict()
