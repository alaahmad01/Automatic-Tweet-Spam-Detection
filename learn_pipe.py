from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, average_precision_score, f1_score


class LearnPipe:

    def __init__(self, label, learning_pipeline, data):
        self.label = label
        self.learning_pipeline = learning_pipeline
        self.learn_set = data[0]
        self.learn_label = data[2]
        self.test_set = data[1]
        self.test_label = data[3]
        pass

    def learn(self):
        self.learning_pipeline.fit(self.learn_set, self.learn_label)
        pass

    def predict(self):
        return self.learning_pipeline.predict(self.test_set)

    def test(self, prediction):
        print(
            "\n", self.label, ": \n",
            "\nconfusion matrix: \n", confusion_matrix(self.test_label, prediction),
              "\naccuracy: ", accuracy_score(self.test_label, prediction),
              "\nprecision: ", average_precision_score(self.test_label, prediction),
              "\nrecall: ", recall_score(self.test_label, prediction),
              "\nf1: ", f1_score(self.test_label, prediction))
        pass

    def learn_and_test(self):
        self.learn()
        self.test(prediction=self.predict())
        pass
