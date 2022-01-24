import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler


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

    # NLP Neural Network
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1,
                                  max_iter=5000)),
        ]
    )

    pipeline.fit(X_train, Y_train)
    pred_NNN = pipeline.predict(X_test)

    accuracy_NNN = accuracy_score(Y_test, pred_NNN)
    cn_NNN = confusion_matrix(Y_test, pred_NNN)
    precision_NNN = average_precision_score(Y_test, pred_NNN)
    f1_NNN = f1_score(Y_test, pred_NNN)
    recall_NNN = recall_score(Y_test, pred_NNN)
    print("\nNLP Neural Network: ",
          "\nconfusion matrix: \n", cn_NNN,
          "\naccuracy: ", accuracy_NNN,
          "\nprecision: ", precision_NNN,
          "\nrecall: ", recall_NNN,
          "\nf1: ", f1_NNN,
          "\n----------------------------------------")

    return pred_NNN
