# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:36:34 2017

@author: ArsalanMalik
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import ModelEvaluation
import SearchParameters

class DeepLearning(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

        self.clf = MLPClassifier()
        self.best_parameter = {}

    def startDL(self):
        print("------------------ Deep Learning -------------------")
        # self.findBestParameters()
        # self.gridSearch()
        self.randomSearch()
        

    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = MLPClassifier()
        scores = cross_val_score(self.clf, self.dataset_x, self.dataset_y, cv=10, scoring="accuracy")
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        # score= cross_val_score(self.clf, self.train_x, self.train_y, cv=10, scoring="recall")
        # print(score)
        # print("Roc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def test(self):
        """
        Test the model with best parameters found in randomSearch() or gridSearch()
        :return:
        """
        #self.clf = MLPClassifier()
        self.clf = MLPClassifier()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for Deep Learner ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)


    def randomSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = {'hidden_layer_sizes': [(400, 360, 340, 300, 260, 220, 180, 140,100,60,20)],
                            'activation': ['identity', 'logistic', 'tanh', 'relu'],
                            'solver': ['lbfgs', 'sgd', 'adam'],
                            'alpha': [0.01, 0.1, 0.05, 0.5, 1.0, ],
                            'learning_rate': ['constant', 'adaptive'],
                            'max_iter': [100, 200, 300, 400, 500],
                            'early_stopping': [True]}
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=100, train_x=self.dataset_x, train_y=self.dataset_y)


    def gridSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = {'hidden_layer_sizes': [(50, 100, 50)],
                            'activation': ['relu'],
                            'solver': ['lbfgs'],
                            'alpha': [0.01, 0.1, 0.05, 0.5, 1.0],
                            'learning_rate': ['adaptive'],
                            'max_iter': [100, 200, 300],
                            'early_stopping': [True]}
        self.best_parameter = SearchParameters.gridSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, train_x=self.dataset_x, train_y=self.dataset_y)