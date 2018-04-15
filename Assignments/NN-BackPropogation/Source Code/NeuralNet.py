import pandas as pd
import random
import math, sys
from os import path

def getData(filename):
    data = pd.read_csv(filename, sep=",", header=None)
    print("Shape: {0}".format(data.shape))
    # print(data)
    return data


def findOutputNeuronCount(data):
    total_col = data.shape[1]
    max_value = data[total_col-1].max()
    return max_value


def divideDataset(dataset, training_percentage):
    train_dataset = dataset.sample(frac = training_percentage/100)
    test_dataset = dataset.loc[~dataset.index.isin(train_dataset.index)]
    print("Total Data:{0}, Training Data:{1}, Test Data:{2}".format(dataset.shape, train_dataset.shape,
                                                                    test_dataset.shape))
    return (train_dataset, test_dataset)


def getMaxIndex(output):
    max = 0
    index = 0
    for j in range(len(output)):
        if output[j] > max:
            max = output[j]
            index = j
    return index


def meanSquareError(output_matrix, instance):
    output_class = instance[instance.shape[0] - 1]
    predicted_output_list = output_matrix[len(output_matrix) - 1]
    e = 0
    for i in range(len(predicted_output_list)):
        ## only the correct output node will be 1, rest will be 0
        if (i == output_class - 1):
            t = 1
        else:
            t = 0
        e = e + (t - predicted_output_list[i])**2
    mse = e/len(predicted_output_list)
    return mse


def assignRandomWeights(nodes_per_layer_list):
    weight_matrix = []
    for i in range(len(nodes_per_layer_list)-1):
        col = nodes_per_layer_list[i] + 1  # 1 added for bias node
        row = nodes_per_layer_list[i+1]
        for j in range(row):
            m = []
            for w in range(col):
                m.append(random.uniform(-1, 1))
            weight_matrix.append(m)

    wdf = pd.DataFrame(weight_matrix)
    # print(wdf)
    return wdf


def activationFunction(value):
    result = round(1/(1+math.exp(-value)),2)
    return result


def forwardPass(data, nodes_per_layer_list, wdf):
    start = 0
    net_matrix = []
    for i in range(len(nodes_per_layer_list)-1):
        end = start + nodes_per_layer_list[i+1]
        temp = wdf.iloc[start:end]
        net_list = []
        for index, row in temp.iterrows():
            net = 0
            # for input nodes i=0
            if i == 0:
                for j in range(data.shape[0]):
                    if j==data.shape[0]: #Bias weight
                        net = net + row[j]
                    else:
                        net = net + activationFunction(data[j]) * row[j]
            else:
                m = net_matrix[len(net_matrix)-1]
                for j in range(len(m)+1):
                    if j==len(m): #Bias weight
                        net = net + row[j]
                    else:
                        net = net + m[j] * row[j]

            net = activationFunction(net)
            net_list.append(net)
        net_matrix.append(net_list)
        start = start + nodes_per_layer_list[i+1]

    # print("Net Matrix: ", net_matrix)
    return net_matrix



def backwardPass(data, hidden_layers, wdf, output_matrix, learning_rate):
    # print("***Backward pass***")

    ## Calculate Delta for output nodes
    output_class = data[data.shape[0]-1]
    predicted_output = output_matrix[len(output_matrix)-1]
    delta_list = []
    for i in range(len(predicted_output)):
        ## only the correct output node will be 1, rest will be 0
        if(i==output_class-1):
            t=1
        else:
            t=0
        delta = predicted_output[i] * (1 - predicted_output[i]) * (t - predicted_output[i])
        delta_list.append(delta)

    ## Calculate Delta and Weights for hidden nodes
    end = wdf.shape[0]
    start = end - len(delta_list)
    for i in reversed(range(hidden_layers)):
        temp = wdf.iloc[start:end]
        new_delta_list = []
        x = output_matrix[i]
        for j in range(len(x)+1):
            if j==len(x): #bias weight column
                count = 0
                for index, row in temp.iterrows():
                    w = learning_rate * delta_list[count]
                    row[j] = row[j] + w
                    count += 1
            else:
                delta = x[j] * (1-x[j])
                sum = 0
                count = 0
                for index, row in temp.iterrows():
                    sum = sum + row[j] * delta_list[count]
                    w =  learning_rate * x[j] * delta_list[count]
                    row[j] = row[j] + w
                    count += 1
                delta = delta * sum
                new_delta_list.append(delta)

        delta_list = new_delta_list
        end = start
        start = end - len(delta_list)


    ## Calculate Weights of input nodes - no need to calculate Delta
    temp = wdf.iloc[start:end]
    for i in range(data.shape[0]):
        count = 0
        for index, row in temp.iterrows():
            if i == data.shape[0] - 1: # bias weight column
                w = learning_rate * delta_list[count]
            else:
                w = learning_rate * data[i] * delta_list[count]
            row[i] = row[i] + w
            count += 1


def printWeights(wdf,nodes_per_layer_list):
    start = 0
    for layer in range(len(nodes_per_layer_list)-1):
        if layer==0:
            print("Layer {0} (Input Layer):".format(layer))
        elif layer == len(nodes_per_layer_list)-1-1:
            print("Layer {0} (Last hidden Layer):".format(layer))
        else:
            print("Layer {0} (hidden Layer number {1}):".format(layer,layer))
        end = start + nodes_per_layer_list[layer + 1]
        temp = wdf.iloc[start:end]
        count = 0
        for index, row in temp.iterrows():
            msg = "Neuron" + str(count+1) + " weights: "
            for node in range(nodes_per_layer_list[layer]):
                msg = msg + " " + str(row[node])
            print(msg)
            count+=1

        start = end



def main():

    if len(sys.argv) < 6:
        print('Incorrect number of arguments.')
        print('Arg: input dataset, training percent, maximum iterations, number of hidden layers, number of neurons in each layer')
        return

    filename = sys.argv[1]
    training_percentage = int(sys.argv[2])
    max_iteration = int(sys.argv[3])
    hidden_layers = int(sys.argv[4])

    j = 5
    nodes_per_layer_list = []
    for i in range(hidden_layers):
        nodes_per_layer_list.append(int(sys.argv[j]))
        j+=1

    if ( path.exists(filename) == False ):
        print("Error! Wrong path to file!")
        return

    processed_data = getData(filename)
    output_neurons = findOutputNeuronCount(processed_data)
    nodes_per_layer_list.insert(0,processed_data.shape[1]-1)
    nodes_per_layer_list.append(output_neurons)
    wdf = assignRandomWeights(nodes_per_layer_list)

    dataset = divideDataset(processed_data,training_percentage)
    train_data = dataset[0]
    test_data = dataset[1]

    learning_rate = 0.5

    ### Train Learner
    print("*** Training Learner ***")
    total_mse = 0
    count = 0
    for i in range(train_data.shape[0]):
        instance = train_data.iloc[i]
        #Forward pass
        output_matrix = forwardPass(instance, nodes_per_layer_list, wdf)
        mse = meanSquareError(output_matrix, instance)
        if mse == 0.0:
            break
        total_mse = total_mse + mse
        count+=1
        #Backward pass
        backwardPass(instance, hidden_layers, wdf, output_matrix, learning_rate)

        if(count>=max_iteration or count>=train_data.shape[0]):
            break;

    printWeights(wdf, nodes_per_layer_list)
    print("Total training error (MSE):{0}%".format(round(total_mse*100/count, 2)))
    print("Total iterations:",count)

    ### Test Learner
    print("*** Testing Learner ***")
    pos = 0
    neg = 0
    total_mse = 0
    count = 0
    for i in range(test_data.shape[0]):
        # print("\n*** ", i + 1)
        instance = test_data.iloc[i]
        output = forwardPass(instance, nodes_per_layer_list, wdf)
        mse = meanSquareError(output, instance)
        total_mse = total_mse + mse
        count += 1
        output_list = output.pop()
        index = getMaxIndex(output_list)
        if(instance[instance.shape[0]-1] == index+1):
            pos+=1
        else:
            neg+=1

    print("Total test error (MSE):{0}%".format(round(total_mse * 100 / count, 2)))
    # print(" Pos:{0}\n Neg:{1}".format(pos,neg))
    print("Accuracy: {0}%".format(round(pos*100/(pos+neg), 2)))


if __name__ == '__main__':
	main()