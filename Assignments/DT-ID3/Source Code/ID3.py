from os import path
import pandas as pd
import math, sys
from random import randint
from Constants import NegativeClass, PositiveClass, EmptySpace, TargetLabel
import ReducedErrorPruning
from printDT import printTree

Tree = []

'''
Read from file
returns a DataFrame
'''
def getData(filename):
    data = pd.read_csv(filename)
    #print("Shape: {0}".format(data.shape))
    return data

def log2( x ):
    if x == 0:
        return 0
    return math.log( x ) / math.log( 2 )


'''
Calculate Entropy
'''
def entropy(training_data):

    negative_class_count = training_data.where(training_data[TargetLabel] == NegativeClass).dropna().shape[0]
    positive_class_count = training_data.shape[0] - negative_class_count
    #print("Total - instance: {0}".format(negative_class_count))
    #print("Total + instance: {0}".format(positive_class_count))

    if (positive_class_count == negative_class_count) :
        return 1.0
    elif (positive_class_count==0 or negative_class_count==0):
        return 0.0
    else:
        total_class_count = positive_class_count + negative_class_count

        s1 = -(positive_class_count/total_class_count)*log2(positive_class_count/total_class_count)
        s0 = -(negative_class_count/total_class_count)*log2(negative_class_count/total_class_count)
        #print(" +: {0} and -: {1}".format(s1,s0))
        e = s1 + s0
        return e



'''
Calculate Information Gain
'''
def informationGain(training_data):
    s = entropy(training_data)

    ###calculate Gain for each attribute
    column_names = list(training_data.columns.values)
    column_names.pop()
    #print(column_names)

    info_gain_dict = {}
    for col in column_names:
        negative_set = training_data.where(training_data[col] == NegativeClass).dropna()
        negative_set_count = negative_set.shape[0]
        entropy_for_negative_set = entropy(negative_set)
        #print("S0: {0} with total negative instance: {1}".format(entropy_for_negative_set, negative_set_count))

        positive_set = training_data.where(training_data[col] == PositiveClass).dropna()
        positive_set_count = positive_set.shape[0]
        entropy_for_positive_set = entropy(positive_set)
        #print("S1: {0} with total positive instance: {1}".format(entropy_for_positive_set, positive_set_count))

        total_count = positive_set_count + negative_set_count

        ig = s - (positive_set_count/total_count * entropy_for_positive_set) - (negative_set_count/total_count * entropy_for_negative_set)
        info_gain_dict[col] = ig

    return info_gain_dict


'''
ID3 Algorithm
'''
def ID3Algo(data, current_node_index):
    ig_dict = informationGain(data)
    root = max(ig_dict, key=lambda k: ig_dict[k])
    if ig_dict[root] == 0.0:
        # toss a coin
        Tree[2 * current_node_index - 1] = NegativeClass
        Tree[2 * current_node_index] = PositiveClass
        return
    #print("***Node inserted: {0} at index: {1}".format(root,current_node_index-1))
    Tree[current_node_index-1] = root

    total_records = data.shape[0]
    #print("total records {0}".format(total_records))

    negative_class_count = data.where(data[TargetLabel] == NegativeClass).dropna().shape[0]
    positive_class_count = data.shape[0] - negative_class_count
    ### check if all examples are negative
    if (negative_class_count == total_records):
        #print("***putting - at index: {0} and {1}".format(2*current_node_index-1, 2*current_node_index))
        Tree[2 * current_node_index - 1] = NegativeClass
        Tree[2 * current_node_index] = NegativeClass
        return 0
    ### check if all examples are positive
    if (positive_class_count == total_records):
        #print("***putting + at index: {0} and {1}".format(2 * current_node_index - 1, 2 * current_node_index))
        Tree[2 * current_node_index - 1] = PositiveClass
        Tree[2 * current_node_index] = PositiveClass
        return 0


    ### Left node
    data_negative = data.where(data[root] == NegativeClass).dropna()
    del data_negative[root]
    ## if all Class values are negative/positive
    n = data_negative.where(data_negative[TargetLabel] == NegativeClass).dropna().shape[0]
    p = data_negative.shape[0] - n

    if (n == data_negative.shape[0] or p == data_negative.shape[0]):
        if(n == data_negative.shape[0]):
            #print("***Inserting - at index: {0}".format(2*current_node_index-1))
            Tree[2*current_node_index-1] = NegativeClass

        if (p == data_negative.shape[0]):
            #print("***Inserting + at index: {0}".format(2*current_node_index-1))
            Tree[2*current_node_index-1] = PositiveClass

    else:
        ID3Algo(data_negative, current_node_index*2)


    ### Right node
    data_positive = data.where(data[root] == PositiveClass).dropna()
    del data_positive[root]
    ## if all Class values are negative/positive
    n = data_positive.where(data_positive[TargetLabel] == NegativeClass).dropna().shape[0]
    p = data_positive.shape[0] - n

    if (n == data_positive.shape[0] or p == data_positive.shape[0]):
        if (n == data_positive.shape[0]):
            #print("***Inserting - at index: {0}".format(2*current_node_index))
            Tree[2*current_node_index] = NegativeClass

        if (p == data_positive.shape[0]):
            #print("***Inserting + at index: {0}".format(2*current_node_index))
            Tree[2 * current_node_index] = PositiveClass

    else:
        ID3Algo(data_positive, current_node_index*2+1)

    return #return to method train_classifier



'''
Build Decision Tree with training data
'''
def train_classifier(training_data):
    print("Building Decision Tree...")
    total_attributes = training_data.shape[1] - 1
    current_node_index = 1

    ### initialize Tree array
    for i in range(2 ** (total_attributes + 1)):
        Tree.insert(i, EmptySpace)

    ### Begin ID3
    ID3Algo(training_data, current_node_index)

    remove_empty_spaces_from_tree()
    # print("Final Tree...")
    # print(Tree)


'''
Test Classifier with data
'''
def test_classifier(test_data, t):
    predictions = []
    correct_prediction = 0
    incorrect_prediction = 0

    ### iterate through Test data set
    for index,row in test_data.iterrows():
        i = 1 #start from first node of Tree

        while True:
            node = t[i-1]
            ### check if leaf node
            if (node == NegativeClass):
                predictions.append(NegativeClass)
                if (row[TargetLabel] == node):
                    correct_prediction += 1
                else:
                    incorrect_prediction += 1
                break
            elif (node == PositiveClass):
                predictions.append(PositiveClass)
                if (row[TargetLabel] == node):
                    correct_prediction += 1
                else:
                    incorrect_prediction += 1
                break
            elif (node == EmptySpace):
                print("Error! Got Empty Space in Tree node")
                predictions.append(randint(NegativeClass,PositiveClass))
                break

            ### Go to next internel node
            if (row[node] == NegativeClass):
                i = i*2 # left branch
            else:
                i = i*2+1 # right branch

    #print("Correct Predictions:{0}, Incorrect Predictions:{1}".format(correct_prediction,incorrect_prediction))
    return (correct_prediction, incorrect_prediction)


def find_accuracy(data, t):
    predictions = test_classifier(data,t)
    accuracy = predictions[0]/(predictions[0]+predictions[1])*100
    return accuracy


def check_tree_nodes(t):
    total_nodes = len(t) - t.count(EmptySpace)
    leafs = t.count(PositiveClass) + t.count(NegativeClass)
    nodes = total_nodes - leafs
    return (nodes,leafs)


def remove_empty_spaces_from_tree():
    ### Delete empty spaces from the end of the tree
    for element in reversed(Tree):
        if element == EmptySpace:
            Tree.pop()
        else:
            break

def print_general_summary(trainData, testData, t):
    print("Number of training instances = {0}".format(trainData.shape[0]))
    print("Number of training attributes = {0}".format(trainData.shape[1] - 1))
    nodes = check_tree_nodes(t)
    print("Total number of nodes in the tree = {0}".format(nodes[0]+nodes[1]))
    print("Number of leaf nodes in the tree = {0}".format(nodes[1]))
    accuracy = find_accuracy(trainData, t)
    print("Accuracy of the model on training dataset = {0}%\n".format(accuracy))

    print("Number of testing instances = {0}".format(testData.shape[0]))
    print("Number of testing attributes = {0}".format(testData.shape[1] - 1))
    accuracy = find_accuracy(testData, t)
    print("Accuracy of the model on testing dataset = {0}%\n".format(accuracy))


def pre_pruned_summary(trainData, testData, validationData, t):
    print_general_summary(trainData, testData, t)
    print("Number of validation instances = {0}".format(validationData.shape[0]))
    print("Number of validation attributes = {0}".format(validationData.shape[1] - 1))
    accuracy = find_accuracy(validationData, t)
    print("Accuracy of the model on validation dataset before pruning = {0}%\n".format(accuracy))


def post_pruned_summary(trainData, testData, validationData, t):
    print_general_summary(trainData, testData, t)
    print("Number of validation instances = {0}".format(validationData.shape[0]))
    print("Number of validation attributes = {0}".format(validationData.shape[1] - 1))
    accuracy = find_accuracy(validationData, t)
    print("Accuracy of the model on validation dataset after pruning = {0}%\n".format(accuracy))




'''
Main Method
'''
def main():
    if len(sys.argv) != 5:
        print('Incorrect number of arguments.')
        print('Arg: training_file, test_file, validation_file and pruning factor.')
        return

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    validation_file = sys.argv[3]
    pruning_factor = float(sys.argv[4])

    if ( path.exists(training_file) == False or path.exists(test_file) == False or path.exists(validation_file)==False ):
        print("Error! Wrong path to files!")
        return

    ### Read Data from file
    training_data = getData(training_file)
    test_data = getData(test_file)
    validation_data = getData(validation_file)

    print("Got all the input!")

    ### Train the classifier with training data
    train_classifier(training_data)

    ### Prune Tree
    pruned_tree = ReducedErrorPruning.pruneTree(training_data, list(Tree), pruning_factor)


    print("\n-------------Pre-Pruned Accuracy------------")
    pre_pruned_summary(training_data, test_data, validation_data, Tree)
    print("Final Tree:")
    printTree(Tree)

    print("\n-------------Post-Pruned Accuracy------------")
    post_pruned_summary(training_data, test_data, validation_data, pruned_tree)
    print("Pruned Tree:")
    printTree(pruned_tree)




if __name__ == '__main__':
	main()