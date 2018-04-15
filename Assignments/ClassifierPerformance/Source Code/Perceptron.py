# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:27:05 2017

@author: ArsalanMalik
"""
from sklearn.linear_model import Perceptron
import SearchParameters
import ModelEvaluation

class PC(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        
        self.clf = Perceptron()
        self.best_parameter = {}
        
    def startPC(self):
        print("-----------Perceptron---------")
        #self.findBestParameters()
        #self.gridSearch()
        self.randomSearch()

    '''     
    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = PC(shuffle=True, n_jobs=1, eta0=1)
        scores = cross_val_score(self.clf, self.train_x, self.train_y, cv=10, scoring="accuracy")
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        # score= cross_val_score(self.clf, self.train_x, self.train_y, cv=10, scoring="recall")
        # print(score)
        # print("Roc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
     '''  
     
    def test(self):
        """
        Test the model with best parameters found in randomSearch() or gridSearch()
        :return:
        """
        # self.clf = Perceptron(shuffle=False, random_state=200, penalty='None', eta0=16, alpha = 0.00001)
        self.clf = Perceptron()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for Perceptron ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)

    
    def randomSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = {'penalty':[None, 'l2', 'l1', 'elasticnet'],
                            'random_state': [10,20,40, 60, 80, 140],
                            'alpha':[0.0001, 0.00001, 0.001, 0.01],
                            'n_jobs': [1, 2, 3, 4],
                            'eta0': [1, 2, 3, 4],
                            'warm_start':[True, False],
                            'shuffle':[True, False],
                            'fit_intercept':[True, False]
                            }
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=50, train_x=self.dataset_x, train_y=self.dataset_y)