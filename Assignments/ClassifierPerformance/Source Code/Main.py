import sys
from os import path
import PreProcess
from DecisionTree import DT
from NeuralNet import NeuralNet
from DeepLearning import DeepLearning
from SVM import SVM
from LogisticRegression import LR
from Bagging import Bagging
from AdaBoost import AdaBoost
from GB import GBC
from KNN import KNNC
from NB import NB
from Perceptron import PC
from RF import RFC
import pandas as pd


def main():

    if len(sys.argv) != 2:
        print('Incorrect number of arguments.')
        print('Arg: Complete Dataset path')
        return

    datasetPath = sys.argv[1]
    if ( path.exists(datasetPath) == False ):
        print("Error! Wrong path to file!")
        return
    
    
    #(dataset_x, dataset_y) = PreProcess.preprocessData(datasetPath)
    '''
    #1
    dtModel = DT(dataset_x, dataset_y)
    dtModel.startDT()

    #2
    perceptronModel = PC(dataset_x, dataset_y)
    perceptronModel.startPC()
    '''
    dataset_x = pd.read_csv('')
    dataset_y = pd.read_csv('')
    #3
    nnModel = NeuralNet(dataset_x, dataset_y)
    nnModel.startNN()
    '''
    #4
    dlModel = DeepLearning(dataset_x, dataset_y)
    dlModel.startDL()
    
    
    #5
    svmModel = SVM(dataset_x, dataset_y)
    svmModel.startSVM()

    #6
    nbModel = NB(dataset_x, dataset_y)
    nbModel.startNB()

    #7
    lrModel = LR(dataset_x, dataset_y)
    lrModel.startLR()

    #8
    knnModel = KNNC(dataset_x, dataset_y)
    knnModel.startKNN()

    #9
    baggingModel = Bagging(dataset_x, dataset_y)
    baggingModel.startBagging()

    #10
    rfcModel = RFC(dataset_x, dataset_y)
    rfcModel.startRFC()

    #11
    adaBoostModel = AdaBoost(dataset_x, dataset_y)
    adaBoostModel.startAdaBoost()

    # 12
    gbcModel = GBC(dataset_x, dataset_y)
    gbcModel.startGBC()


    ### Test Classifiers
    
    dtModel.test()
    perceptronModel.test()
    
    svmModel.test()
    nbModel.test()
    lrModel.test()
    knnModel.test()
    baggingModel.test()
    rfcModel.test()
    adaBoostModel.test()
    gbcModel.test()
    dlModel.test()
    '''
    nnModel.test()
    
if __name__ == '__main__':
    main()