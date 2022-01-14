import pandas as pd
from tweet import *
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

train_features, test_features = extract_features("train.csv", 0.7)

X = [x[0] for x in train_features]
Y = [x[1] for x in train_features]

t = Features(Tweet(1, "hi my name is Ala, I'm new to twitter #Add_me @arajeh", 10, 2, 3, 0, "Nablus",
                   'Quality')).assemble_vector()

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(11, 2), random_state=1)
# clf.fit(X, Y)

# gnb = GaussianNB()
# clf = gnb.fit(X, Y)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, Y)

f = open('out.txt', 'w')

m = 0
n = 0
for i in range(len(test_features)):
    p = clf.predict([test_features[i][0]])
    if p == test_features[i][1]:
        s = "Match at "
        s += str(i) + '\n'
        f.write(s)
        m += 1
    else:
        s = "Not match at "
        s += str(i) + '\n'
        f.write(s)
        n += 1

print("Match: ", m)
print("Not Match: ", n)
