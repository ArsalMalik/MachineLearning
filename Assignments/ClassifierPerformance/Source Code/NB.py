# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:27:05 2017

@author: ArsalanMalik
"""
from sklearn.naive_bayes import BernoulliNB as NaiveBayes
import SearchParameters
import ModelEvaluation

class NB(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        
        self.clf = NaiveBayes()
        self.best_parameter = {}
        
    def startNB(self):
        print("-----------Naive Bayes---------")
        #self.findBestParameters()
        #self.gridSearch()
        self.randomSearch()

    '''     
    def findBestParameters(self):
        """
        Try different parameters for finding the best score
        :return:
        """
        self.clf = GNB()
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
        # self.clf = NaiveBayes(fit_prior=True, binarize=0.0, alpha=0.5)
        self.clf = NaiveBayes()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for Naive Bayes ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)
    
    def randomSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = { 'alpha':[0.5, 1.0, 3.0, 5.0, 10.0],
                             'binarize':[0.0, 1.0, 2.0, 5.0,10.0,20.0],
                             'fit_prior':[True, False]
                            }
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=40, train_x=self.dataset_x, train_y=self.dataset_y)