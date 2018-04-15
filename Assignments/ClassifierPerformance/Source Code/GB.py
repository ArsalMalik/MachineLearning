# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 00:29:04 2017

@author: ArsalanMalik
"""
from sklearn.ensemble import GradientBoostingClassifier as GB
import SearchParameters
import ModelEvaluation

class GBC(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        
        self.clf = GB()
        self.best_parameter = {}
        
    def startGBC(self):
        print("-----------Gradient Boosting---------")
        #self.findBestParameters()
        #self.gridSearch()
        self.randomSearch()

    '''     
    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = GB()
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
        # self.clf = GB(random_state=40, n_estimators=40, max_features='sqrt', learning_rate=0.8, criterion='friedman_mse')
        self.clf = GB()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for Gradient Boosting ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)

    
    def randomSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = { 'learning_rate':[0.1,0.5,1.0],
                             'n_estimators':[50,100,150],
                             'criterion':['friedman_mse'],
                             'max_features':['sqrt', 'log2'],
                             'random_state':[None, 20,50]
                            }
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=30, train_x=self.dataset_x, train_y=self.dataset_y)