from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import ModelEvaluation
import SearchParameters

class LR(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

        self.clf = LogisticRegression()
        self.best_parameter = {}

    def startLR(self):
        print("------------------ Logistic Regression -------------------")
        # self.findBestParameters()
        # self.gridSearch()
        self.randomSearch()


    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial', C=10, max_iter=300, fit_intercept=True)
        scores = cross_val_score(self.clf, self.dataset_x, self.dataset_y, cv=10, scoring="accuracy")
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    def test(self):
        """
        Test the model with best parameters found in randomSearch() or gridSearch()
        :return:
        """
        # self.clf = LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial', C=10, max_iter=300, fit_intercept=True)
        self.clf = LogisticRegression()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for Logistic Regression ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)


    def randomSearch(self):
        tuned_parameters = {'penalty': ['l2'],
                            'C': [0.5, 1, 5, 10, 100],
                            'solver': ['lbfgs', 'newton-cg', 'sag'],
                            'multi_class': ['ovr', 'multinomial'],
                            'max_iter': [100, 200, 300, 400, 500],
                            'fit_intercept': [True, False],
                            'warm_start': [True, False]}
        # tuned_parameters = {'penalty': ['l1'],
        #                     'C': [0.5, 1, 5, 10, 100],
        #                     'solver': ['liblinear', 'saga'],
        #                     'multi_class': ['ovr'],
        #                     'max_iter': [100, 200, 300, 400],
        #                     'fit_intercept': [True, False]}
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=60, train_x=self.dataset_x, train_y=self.dataset_y)


    def gridSearch(self):
        tuned_parameters = {'penalty': ['l2'],
                            'C': [0.5, 1, 10, 100],
                            'solver': ['lbfgs', 'newton-cg', 'sag'],
                            'multi_class': ['ovr'],
                            'max_iter': [100, 200, 300],
                            'fit_intercept': [True, False]}
        self.best_parameter = SearchParameters.gridSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, train_x=self.dataset_x, train_y=self.dataset_y)

