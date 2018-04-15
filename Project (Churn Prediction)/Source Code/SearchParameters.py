from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import Train


"""
Use Random search or Grid search to search best parameters for different classifiers
"""

def randomSearch(classifier, parameters, cv, n_iter, train_x, train_y):
    print("***** Random Search *****")
    print("Cross-Validation:{0} and number of iterations:{1}".format(cv, n_iter))

    scores = ['accuracy', 'recall', 'precision']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        if (score == 'accuracy'):
            scoring_method = score
        else:
            scoring_method = score + '_macro'

        clf = RandomizedSearchCV(classifier, param_distributions=parameters, cv=cv, scoring=scoring_method, n_iter=n_iter)
        clf.fit(train_x, train_y)

        print("Best parameters and scores set found on development set:")
        # print(self.clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)
        print()
        # print("Random scores on development set:")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # print()

        return clf.best_params_


def gridSearch(classifier, parameters, cv, train_x, train_y):
    print("***** Grid Search *****")
    print("Cross-Validation: ",cv)

    scores = ['accuracy', 'recall', 'precision']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        if (score == 'accuracy'):
            scoring_method = score
        else:
            scoring_method = score + '_macro'

        clf = GridSearchCV(classifier, param_grid=parameters, cv=cv, scoring=scoring_method)
        clf.fit(train_x, train_y)

        print("Best parameters and scores set found on development set:")
        # print(self.clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)
        print()
        # print("Grid scores on development set:")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # print()

        return clf.best_params_

def writeResult(results, classifier):
    with open("Best_Parameters.txt", 'a+') as f:
        f.write("-------------------------------------------")
        f.write(classifier)
        f.write(str(results))
    f.close()


def DT(train_x, train_y):
    # Set the parameters by cross-validation
    clf = tree.DecisionTreeClassifier()
    tuned_parameters = {'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [10, 20, 30, 50, 100],
                        'max_features': ['auto', 'log2', None],
                        'min_samples_split': [2, 5, 10, 15],
                        'min_samples_leaf': [1, 2, 5, 10]
                        }
    best_parameter = randomSearch(classifier=clf, parameters=tuned_parameters, cv=5,
                                                        n_iter=50, train_x=train_x, train_y=train_y)
    writeResult(best_parameter, "DT")


def SVM(train_x, train_y):
    clf = SVC()
    tuned_parameters = {'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
                        'gamma': [0.1, 0.5, 1.0],
                        'C': [1, 10, 100],
                        'degree': [1, 3, 5],
                        'coef0': [1e-3, 0.1, 0.5]}
    best_parameter = randomSearch(classifier=clf, parameters=tuned_parameters, cv=5,
                                                        n_iter=50, train_x=train_x, train_y=train_y)
    writeResult(best_parameter, "SVM")

def adaboost(train_x, train_y):
    clf = AdaBoostClassifier()
    tuned_parameters = {'base_estimator': [tree.DecisionTreeClassifier(), LogisticRegression()],
                        'n_estimators': [100, 150, 200],
                        'learning_rate': [0.5, 1.0, 1.5],
                        'algorithm': ['SAMME', "SAMME.R"]
                        }
    best_parameter = randomSearch(classifier=clf, parameters=tuned_parameters, cv=5,
                                                        n_iter=50, train_x=train_x, train_y=train_y)
    writeResult(best_parameter, "Adaboost")


def bagging(train_x, train_y):
    clf = BaggingClassifier()
    tuned_parameters = {
        'base_estimator': [tree.DecisionTreeClassifier(), LogisticRegression(), KNeighborsClassifier()],
        'n_estimators': [5, 15, 20, 30],
        'max_samples': [0.5, 0.7, 0.9],
        'max_features': [0.5, 1.0],
        'bootstrap': [True, False]
    }
    best_parameter = randomSearch(classifier=clf, parameters=tuned_parameters, cv=5,
                                                        n_iter=50, train_x=train_x, train_y=train_y)
    writeResult(best_parameter, "Bagging")


def ann(train_x, train_y):
    clf = MLPClassifier()
    tuned_parameters = {'hidden_layer_sizes': [(100, 50, 20), (50, 25), (100, 150, 100, 50)],
                        'activation': ['logistic', 'tanh', 'relu'],
                        'solver': ['lbfgs', 'sgd'],
                        'alpha': [0.01, 0.1, 0.05, 0.5, 1.0],
                        'learning_rate': ['constant', 'adaptive'],
                        'max_iter': [200, 300, 400, 500],
                        'early_stopping': [True]}
    best_parameter = randomSearch(classifier=clf, parameters=tuned_parameters, cv=5,
                                                        n_iter=50, train_x=train_x, train_y=train_y)
    writeResult(best_parameter, "ANN")



if __name__ == '__main__':

    df = Train.getTrainData()
    (train_x, train_y) = Train.splitDataset(df)

    DT(train_x, train_y)
    SVM(train_x, train_y)
    adaboost(train_x, train_y)
    bagging(train_x, train_y)
    ann(train_x, train_y)
