from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def evaluateModelWithCV(classifier, data_x, data_y, cv):
    scores = cross_val_score(classifier, data_x, data_y, cv=10, scoring="accuracy")
    # print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(classifier, data_x, data_y, cv=10, scoring="recall_macro")
    print(scores)
    print("Recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def evaluateModel(class_labels, prediction):
    accuracy = accuracy_score(class_labels, prediction)
    precision = precision_score(class_labels, prediction, average="macro")
    recall = recall_score(class_labels, prediction, average="macro")

    print("accuracy: {0}%".format(round(accuracy * 100, 2)))
    print("precision: {0}%".format(round(precision * 100, 2)))
    print("recall: {0}%".format(round(recall * 100, 2)))
    # print(classification_report(self.test_y, prediction))

