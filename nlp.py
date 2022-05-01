import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from learn_pipe import LearnPipe


def text_learning(tweet_in, type_in):
    """
    function to apply text learning methods to the text of the tweet in a neural network pipe
    """

    tweet_in = tweet_in.apply(lambda x: re.sub(
        r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
        '', x))  # remove URLs from tweet
    tweet_in = tweet_in.apply(lambda x: re.sub(r'[^\w\s]', '', x))  # remove redundant spaces
    tweet_in = tweet_in.apply(lambda x: x.lower())

    x_vec = [_ for _ in tweet_in]
    y_vec = [1 if x == 'Spam' else 0 for x in type_in]

    # split the data so it trains on the first third
    # and predicts the other two thirds
    x_train = x_vec[:int(len(x_vec) * (1 / 3))]
    y_train = y_vec[:int(len(x_vec) * (1 / 3))]
    x_test = x_vec[int(len(x_vec) * (1 / 3)):]
    y_test = y_vec[int(len(x_vec) * (1 / 3)):]

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
        [x_train, x_test, y_train, y_test]
    )

    pipe.learn_and_test()

    return pipe.predict()
