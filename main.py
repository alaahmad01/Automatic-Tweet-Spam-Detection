from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from plot import calibration_plot
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tweet import extract_features
from learn_pipe import LearnPipe

if __name__ == '__main__':
    feature = extract_features("train.csv")

    X = [x[0] for x in feature]
    Y = [x[1] for x in feature]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    data = [X_train, X_test, Y_train, Y_test]

    pipes = [
        LearnPipe(
            "Naive Bayes",
            Pipeline(
                [
                    ("var", VarianceThreshold(threshold=(.8 * (1 - .8)))),
                    ("chi2", SelectKBest(chi2, k=8)),
                    ("scaler", StandardScaler()),
                    ("clf", GaussianNB()),
                ]
            ), data),
        LearnPipe(
            "Random Forrest",
            Pipeline(
                [
                    ("var", VarianceThreshold(threshold=(.8 * (1 - .8)))),
                    ("chi2", SelectKBest(chi2, k=8)),
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(n_estimators=10, min_samples_split=1000, random_state=1)),
                ]
            ), data),
        LearnPipe(
            "Neural Network",
            Pipeline(
                [
                    ("var", VarianceThreshold(threshold=(.8 * (1 - .8)))),
                    ("chi2", SelectKBest(chi2, k=8)),
                    ("scaler", StandardScaler()),
                    ("clf",
                     MLPClassifier(solver='lbfgs', hidden_layer_sizes=(7, 6, 5, 4, 3), alpha=15e-6, random_state=1,
                                   max_iter=5000)),
                ]
            ), data)
    ]

    for pipe in pipes:
        pipe.learn_and_test()

    calibration_plot(X_test, Y_test, [x.learning_pipeline for x in pipes])
