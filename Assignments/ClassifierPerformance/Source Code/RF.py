# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 00:17:30 2017

@author: ArsalanMalik
"""
from sklearn.ensemble import RandomForestClassifier as RF
import SearchParameters
import ModelEvaluation

class RFC(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        
        self.clf = RF()
        self.best_parameter = {}
        
    def startRFC(self):
        print("-----------Random Forest---------")
        #self.findBestParameters()
        #self.gridSearch()
        self.randomSearch()

    '''     
    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = KNN()
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
        # self.clf = RF(random_state=20, n_estimators=10, max_features='sqrt', criterion='entropy', bootstrap=False)
        self.clf = RF()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for Random Forest ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)

    
    def randomSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = { 'n_estimators':[10,20,30,40],
                             'criterion':['gini', 'entropy'],
                             'max_features':['sqrt', 'log2'],
                             'random_state':[None, 10, 30],
                             'bootstrap':[True, False]
                            }
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=30, train_x=self.dataset_x, train_y=self.dataset_y)