import pandas as pd
from tweet import Tweet, Features
from sklearn.neural_network import MLPClassifier
from sklearn import tree

tweets = []
features = []
df = pd.read_csv("train.csv")

for row_index in range(df.shape[0]):
    row = df.loc[row_index]

    tweets.append(Tweet(0 if pd.isnull(df['Id'][row_index]) else int(df['Id'][row_index]),
                        "" if pd.isnull(df['Tweet'][row_index]) else str(df['Tweet'][row_index]),
                        0 if pd.isnull(df['following'][row_index]) else int(df['following'][row_index]),
                        0 if pd.isnull(df['followers'][row_index]) else int(df['followers'][row_index]),
                        0 if pd.isnull(df['actions'][row_index]) else int(df['actions'][row_index]),
                        0 if pd.isnull(df['is_retweet'][row_index]) else int(df['is_retweet'][row_index]),
                        "" if pd.isnull(df['location'][row_index]) else str(df['location'][row_index]),
                        "" if pd.isnull(df['Type'][row_index]) else str(df['Type'][row_index])))

for tweet in tweets:
    features.append(Features(tweet).assemble_vector())

X = [x[0] for x in features]
Y = [x[1] for x in features]

t = Features(Tweet(1, "hi my name is Ala, I'm new to twitter #Add_me @arajeh", 10, 2, 3, 0, "Nablus",
                   'Quality')).assemble_vector()

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, Y)
"""MLPClassifier(alpha=1e-05, hidden_layer_sizes=(11, 2), random_state=1,
              solver='lbfgs')"""

"""clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

tree.plot_tree(clf)

import graphviz"""
