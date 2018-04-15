from sklearn import tree
from sklearn.model_selection import cross_val_score
import ModelEvaluation
import SearchParameters

class DT(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.clf = tree.DecisionTreeClassifier()
        self.best_parameter = {}

    def startDT(self):
        print("------------------ Decision Tree -------------------")
        # self.findBestParameters()
        # self.gridSearch()
        self.randomSearch()



    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = tree.DecisionTreeClassifier(criterion="entropy")
        scores = cross_val_score(self.clf, self.dataset_x, self.dataset_y, cv=10)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    def test(self):
        """
        Test the model with best parameters found in randomSearch() or gridSearch()
        :return:
        """
        # self.clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=30, min_samples_split=3, max_features=None)
        self.clf = tree.DecisionTreeClassifier()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for Decision Tree ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)



    def randomSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = {'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [10, 20, 30, 40, 50, 60, 100],
                            'max_features': ['auto', 'log2', None],
                            'min_samples_split': [2, 5, 10, 15],
                            'min_samples_leaf': [1, 2, 5, 10]
                            }
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=100, train_x=self.dataset_x, train_y=self.dataset_y)


    def gridSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = {'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [10, 20, 25, 35, 30, 40],
                            'max_features': ['auto', 'log2', None],
                            'min_samples_split': [2, 3, 5],
                            'min_samples_leaf': [1, 2, 5]
                            }

        self.best_parameter = SearchParameters.gridSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, train_x=self.dataset_x, train_y=self.dataset_y)
