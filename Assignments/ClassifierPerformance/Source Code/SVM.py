from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import ModelEvaluation
import SearchParameters

class SVM(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

        self.clf = SVC()
        self.best_parameter = {}

    def startSVM(self):
        print("------------------ SVM -------------------")
        # self.findBestParameters()
        # self.gridSearch()
        self.randomSearch()


    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = SVC(kernel='poly', C=1000, gamma=0.5, degree=3, coef0=0.1)
        scores = cross_val_score(self.clf, self.dataset_x, self.dataset_y, cv=10, scoring="accuracy")
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



    def test(self):
        """
        Test the model with best parameters found in randomSearch() or gridSearch()
        :return:
        """
        # self.clf = SVC(kernel='poly', C=1000, gamma=0.5, degree=3, coef0=0.1)
        self.clf = SVC()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for SVM ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)


    def randomSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = {'kernel': ['rbf','linear','sigmoid','poly'],
                            'gamma': [1e-3, 1e-4, 0.1, 0.5, 1.0],
                            'C': [1, 10, 100, 1000],
                            'degree': [1, 3, 5],
                            'coef0': [1e-3, 0.1, 0.5]}
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=50, train_x=self.dataset_x, train_y=self.dataset_y)


    def gridSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0.1, 0.5], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['sigmoid'], 'gamma': [1e-2, 0.1, 0.5, 0.01], 'C': [1, 10, 100, 1000], 'coef0':[1e-3, 0.1, 0.5]},
                            {'kernel': ['poly'], 'gamma': [1e-2, 0.1, 0.5, 0.01], 'C': [1, 10, 100, 1000],
                             'coef0': [1e-3, 0.1, 0.5], 'degree': [1, 3, 5]}]
        # tuned_parameters = {'kernel': ['rbf', 'linear', 'sigmoid', 'poly'], 'gamma': [1e-3, 1e-4, 0.1, 0.5, 1.0],
        #                     'C': [1, 10, 100, 1000], 'degree': [1, 3, 5], 'coef0': [1e-3, 0.1, 0.5]}
        self.best_parameter = SearchParameters.gridSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, train_x=self.dataset_x, train_y=self.dataset_y)
