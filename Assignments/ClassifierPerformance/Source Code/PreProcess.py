from sklearn import preprocessing
from  sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np


def readData(filename):
    """
    Read data from a file
    @param filename: path to the file
    @return: All the records from the file
    """
    with open(filename) as f:
        dataset = f.readlines()

    data = []
    for line in dataset:
        attr = line.strip().split(",")
        data.append(attr)
    datas = np.array(data)
    print("Shape of Data: {0}".format(datas.shape))
    return datas


def isNumeric(data, col):
    """
    Check if all the values of a column in 2D array are numeric
    :param data: a 2D array of instances
    :param col: column index
    :return: True if all the values are numeric, false otherwise
    """
    for row in range(len(data)):
        number = data[row][col].strip()
        if isinstance(number, np.float) or isinstance(number, np.int):
            continue
        else:
            return False
    return True


def standardizeData(dataset):
    """
    Standardize numeric values
    :param dataset: a 2D array of instances
    :return: standardized data
    """
    print("Standardizing numeric values...")
    no_of_attributes = dataset.shape[1]
    for col in range(no_of_attributes):
        if isNumeric(dataset, col):
            column_values = dataset[:, col]
            scaler = preprocessing.StandardScaler()
            column_values_t = scaler.fit_transform(column_values)
            dataset[:, col] = column_values_t
            print(len(column_values_t), column_values_t)
    return dataset


def transformCategoricalData(dataset_x, dataset_y):
    """
    Transform categorical/nominal values to numeric
    :param dataset_x: 2D array of instances without class labels
    :param dataset_y: 1D array of class labels
    :return: transformed data
    """
    print("Transforming categorical values to numeric...")
    no_of_attributes = dataset_x.shape[1]
    le = preprocessing.LabelEncoder()

    for col in range(no_of_attributes):
        column_values = dataset_x[:,col]
        column_values_t = le.fit_transform(column_values)
        # print("###col {0} unique values:{1} ".format(col, np.unique(column_values_t)))
        dataset_x[:,col] = column_values_t
    # print(dataset_x)
    dataset_x = encodeCategoricalFeatures(dataset_x)

    dataset_y = le.fit_transform(dataset_y)
    # print(dataset_y)

    return (dataset_x, dataset_y)


def encodeCategoricalFeatures(dataset_x):
    """
    Encode categorical features to fit scikit-learn models
    :param dataset_x: 2D array of instances without class labels
    :return: encoded values for categorical features
    """
    print("Encoding numeric values...")
    enc = OneHotEncoder(sparse=False)
    output = enc.fit_transform(dataset_x)
    # print(enc.n_values_)
    # print(enc.feature_indices_)
    # print("Total records:{0}\nSample record{1}".format(len(output),output[0]))
    return output


def splitDataset(dataset_x, dataset_y):
    """
    Split the dataset into train and test set
    :param dataset_x: 2D array of instances without class labels
    :param dataset_y: 1D array of class labels
    :return: 2D array train set without class labels, 2D array test set without class labels, 1D array class labels for train set, 1D array class labels for test set
    """
    (train_x, test_x, train_y, test_y) = train_test_split(dataset_x, dataset_y, test_size = 0.25)
    print("X_train:{0}, X_test:{1}, y_train:{2}, y_test:{3}".format(len(train_x), len(test_x), len(train_y), len(test_y)))
    return (train_x, test_x, train_y, test_y)


def preprocessData(filename):
    """
    Pre process data
    :param filename: data file path
    :return: pre-processed data split into training and test set
    """
    dataset = readData(filename)
    dataset_x = dataset[:, :dataset.shape[1] - 1]
    dataset_y = dataset[:, dataset.shape[1] - 1]

    dataset_x = standardizeData(dataset_x)
    (dataset_x, dataset_y) = transformCategoricalData(dataset_x, dataset_y)

    # (train_x, test_x, train_y, test_y) = splitDataset(dataset_x, dataset_y)

    return (dataset_x, dataset_y)