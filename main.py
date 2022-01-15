from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from plot import calibration_plot
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, f1_score, recall_score
from tweet import extract_features

test_per = 0.1
feature = extract_features("train.csv", test_per)

X = [x[0] for x in feature]
Y = [x[1] for x in feature]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# Naive Bayes
pipeline_NB = Pipeline(
    [
        ("var", VarianceThreshold(threshold=(.8 * (1 - .8)))),
        ("chi2", SelectKBest(chi2, k=8)),
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]
)

pipeline_NB.fit(X_train, Y_train)

pred_NB = pipeline_NB.predict(X_test)

accuracy_NB = accuracy_score(Y_test, pred_NB)
cn_NB = confusion_matrix(Y_test, pred_NB)
precision_NB = average_precision_score(Y_test, pred_NB)
f1_NB = f1_score(Y_test, pred_NB)
recall_NB = recall_score(Y_test, pred_NB)
print("\nNaive Bayes: ",
      "\nconfusion matrix: \n", cn_NB,
      "\naccuracy: ", accuracy_NB,
      "\nprecision: ", precision_NB,
      "\nrecall: ", recall_NB,
      "\nf1: ", f1_NB)

# Random Forrest Method
pipeline_RF = Pipeline(
    [
        ("var", VarianceThreshold(threshold=(.8 * (1 - .8)))),
        ("chi2", SelectKBest(chi2, k=8)),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(max_depth=1, random_state=1)),
    ]
)

pipeline_RF.fit(X_train, Y_train)

pred_RF = pipeline_RF.predict(X_test)

accuracy_RF = accuracy_score(Y_test, pred_RF)
cn_RF = confusion_matrix(Y_test, pred_RF)
precision_RF = average_precision_score(Y_test, pred_RF)
f1_RF = f1_score(Y_test, pred_RF)
recall_RF = recall_score(Y_test, pred_RF)
print("\nRandom Forrest: ",
      "\nconfusion matrix: \n", cn_RF,
      "\naccuracy: ", accuracy_RF,
      "\nprecision: ", precision_RF,
      "\nrecall: ", recall_RF,
      "\nf1: ", f1_RF)

# Neural Network
pipeline_NN = Pipeline(
    [
        ("var", VarianceThreshold(threshold=(.8 * (1 - .8)))),
        ("chi2", SelectKBest(chi2, k=8)),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(solver='lbfgs', alpha=5e-5, hidden_layer_sizes=(8, 2), random_state=1,
                              max_iter=400)),
    ]
)

pipeline_NN.fit(X_train, Y_train)

pred_NN = pipeline_NN.predict(X_test)

accuracy_NN = accuracy_score(Y_test, pred_NN)
cn_NN = confusion_matrix(Y_test, pred_NN)
precision_NN = average_precision_score(Y_test, pred_NN)
f1_NN = f1_score(Y_test, pred_NN)
recall_NN = recall_score(Y_test, pred_NN)
print("\nNeural Network: ",
      "\nconfusion matrix: \n", cn_NN,
      "\naccuracy: ", accuracy_NN,
      "\nprecision: ", precision_NN,
      "\nrecall: ", recall_NN,
      "\nf1: ", f1_NN)

calibration_plot(X_test, Y_test, pipeline_NB, pipeline_RF, pipeline_NN)