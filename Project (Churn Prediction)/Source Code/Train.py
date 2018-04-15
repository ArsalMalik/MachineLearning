import pandas as pd
import Members, Transactions, Preprocessing_user_logs

"""
Make the training dataset by merging mambers, transactions, user_logs and train dataset all together
"""

filepath = "data\\train.csv"
new_filename = "processed_data\processed_train_new.csv"

def getTrainData():
    df = readData(new_filename)
    return df

def readData(filepath):
    print("Reading data from Train dataset...")
    train = pd.read_csv(filepath)
    # print(train.info())
    # print(train.head())
    return train

def writeData(train):
    train.to_csv(new_filename, index=False)

def splitDataset(dataset):
    dataset_y = dataset['is_churn'].values.tolist()
    del dataset['msno']
    del dataset['is_churn']
    dataset_x = dataset.values
    # print("dataset: ", dataset.shape)
    # print("x: ",dataset_x.shape)
    # print("y: ",len(dataset_y))
    return (dataset_x, dataset_y)


def processTrainData():
    train = readData(filepath)

    # Merge all datasets
    members = Members.getProcessedData()
    training = pd.merge(left=train, right=members, how='left', on=['msno'])
    training = Members.imputeMissingValues(training)
    del members

    transactions = Transactions.getProcessedData()
    training = pd.merge(left=training, right=transactions, how='left', on=['msno'])
    training = Transactions.imputeMissingValues(training)
    del transactions

    user_logs = Preprocessing_user_logs.getProcessedDataForTrain()
    training = pd.merge(left=training, right=user_logs, how='left', on=['msno'])
    del user_logs

    del training['Unnamed: 0']
    training = training.drop_duplicates(keep="first")
    training = training.dropna()

    writeData(training)


if __name__ == '__main__':
    processTrainData()
