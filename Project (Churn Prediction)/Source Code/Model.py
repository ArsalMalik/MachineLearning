from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
import Train
import Test
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


"""
Check the accuracy, precision, recall of different classifiers with best parameters found in SearchParameters step
"""

def DT(train_x, train_y, test_x, test_y, msno_df):
    print ("DT")
    clf = tree.DecisionTreeClassifier(splitter="best", criterion="entropy", max_depth=10, min_samples_split=10, min_samples_leaf=1)
    checkResult(clf, "Decision Tree", train_x, train_y, test_x, test_y, msno_df)

def svm(train_x, train_y, test_x, test_y, msno_df):
    print ("SVM")
    clf = SVC(kernel='rbf', gamma=.1, C=10)
    checkResult(clf, "SVM", train_x, train_y, test_x, test_y, msno_df)

def adaboost(train_x, train_y, test_x, test_y, msno_df):
    print("Adaboost")
    clf = AdaBoostClassifier(base_estimator=LogisticRegression(), learning_rate=1.0, n_estimators=200, algorithm='SAMME.R')
    checkResult(clf, "Adaboost", train_x, train_y, test_x, test_y, msno_df)

def bagging(train_x, train_y, test_x, test_y, msno_df):
    print ("Bagging")
    clf = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(), max_samples=0.9, n_estimators=30, bootstrap=False)
    checkResult(clf, "Bagging", train_x, train_y, test_x, test_y, msno_df)

def ann(train_x, train_y, test_x, test_y, msno_df):
    print ("ANN")
    clf = MLPClassifier(hidden_layer_sizes=(100,150,100,50), activation="relu", solver="lbfgs", alpha=1.0, max_iter=500)
    checkResult(clf, "ANN", train_x, train_y, test_x, test_y, msno_df)

def checkResult(clf, clf_name, train_x, train_y, test_x, test_y, msno_df):
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    accuracy = accuracy_score(test_y, y_pred)
    recall_macro = recall_score(test_y, y_pred, average="macro")
    recall_micro = recall_score(test_y, y_pred, average="micro")
    precision_macro = precision_score(test_y, y_pred, average="macro")
    precision_micro = precision_score(test_y, y_pred, average="micro")
    scores = [accuracy, recall_macro, recall_micro, precision_macro, precision_micro]
    print (accuracy, recall_macro, recall_micro, precision_macro, precision_micro)
    writeResult(scores, clf_name)

def writeResult(results, classifier):
    with open("Results.txt", 'a+') as f:
        f.write("-------------------------------------------")
        f.write(classifier)
        f.write(str(results))
    f.close()


def main():
    df = Train.getTrainData()
    (train_x, train_y) = Train.splitDataset(df)

    test_df = Test.getTestData()
    (test_x, test_y, msno_df) = Test.splitDataset(test_df)

    DT(train_x, train_y, test_x, test_y, msno_df)
    svm(train_x, train_y, test_x, test_y, msno_df)
    adaboost(train_x, train_y, test_x, test_y, msno_df)
    ann(train_x, train_y, test_x, test_y, msno_df)
    bagging(train_x, train_y, test_x, test_y, msno_df)

if __name__ == '__main__':
    main()