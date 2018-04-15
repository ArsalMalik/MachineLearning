from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import ModelEvaluation
import SearchParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


class Bagging(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

        self.clf = BaggingClassifier()
        self.best_parameter = {}

    def startBagging(self):
        print("------------------ Bagging Classifier -------------------")
        # self.findBestParameters()
        # self.gridSearch()
        self.randomSearch()


    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = BaggingClassifier()
        scores = cross_val_score(self.clf, self.dataset_x, self.dataset_y, cv=10, scoring="accuracy")
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    def test(self):
        """
        Test the model with best parameters found in randomSearch() or gridSearch()
        :return:
        """
        # self.clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=30, bootstrap=False, max_samples=0.9)
        self.clf = BaggingClassifier()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for Bagging ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)

    def randomSearch(self):
        tuned_parameters = {'base_estimator': [DecisionTreeClassifier(), SVC(), LogisticRegression(), KNeighborsClassifier(), MultinomialNB()],
                            'n_estimators': [5, 10, 15, 20, 30, 50],
                            'max_samples': [0.5, 0.7, 0.9],
                            'bootstrap': [True, False]
                            }
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=30, train_x=self.dataset_x, train_y=self.dataset_y)


    def gridSearch(self):
        tuned_parameters = {
            'base_estimator': [DecisionTreeClassifier(), SVC(), LogisticRegression(), KNeighborsClassifier(),
                               MultinomialNB()],
            'n_estimators': [5, 10, 15, 20],
            'max_samples': [0.3, 0.5, 0.7, 0.9],
            'max_features': [0.5, 1.0],
            'bootstrap': [True, False]
            }
        self.best_parameter = SearchParameters.gridSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, train_x=self.dataset_x, train_y=self.dataset_y)
