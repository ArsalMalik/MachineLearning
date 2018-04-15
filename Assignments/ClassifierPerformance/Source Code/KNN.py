# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:27:05 2017

@author: ArsalanMalik
"""
from sklearn.neighbors import KNeighborsClassifier as KNN
import SearchParameters
import ModelEvaluation

class KNNC(object):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        
        self.clf = KNN()
        self.best_parameter = {}
        
    def startKNN(self):
        print("-----------K-Nearest Neighbors---------")
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
        # self.clf = KNN(n_neighbors=20, weights='uniform',algorithm='ball_tree', leaf_size=10,p=1)
        self.clf = KNN()
        self.clf.set_params(**self.best_parameter)
        print("*** Test Result for K-Nearest Neighbors ***")
        ModelEvaluation.evaluateModelWithCV(self.clf, self.dataset_x, self.dataset_y, cv=10)

    
    def randomSearch(self):
        # Set the parameters by cross-validation
        tuned_parameters = { 'n_neighbors':[5,10,20,40,50],
                             'weights':['uniform', 'distance'],
                             'algorithm':['ball_tree', 'kd_tree'],
                             'leaf_size':[15,30,45,50,60],
                             'p':[1, 2],
                             'n_jobs':[1,4]
                            }
        self.best_parameter = SearchParameters.randomSearch(classifier=self.clf, parameters=tuned_parameters, cv=10, n_iter=50, train_x=self.dataset_x, train_y=self.dataset_y)