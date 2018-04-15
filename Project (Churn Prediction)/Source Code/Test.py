import pandas as pd
import Members, Transactions, Preprocessing_user_logs

"""
Make the Test dataset by merging mambers, transactions, user_logs and sample_submission_zero dataset all together
"""

filepath = "data\\sample_submission_zero.csv"
new_filename = "processed_data\\processed_test_new.csv"

def getTestData():
    df = readData(new_filename)
    return df

def readData(filepath):
    print("Reading data from Test dataset...")
    test = pd.read_csv(filepath)
    #print(test.info())
    #print(test.head())
    return test

def writeData(test):
    test.to_csv(new_filename, index=False)

def splitDataset(dataset):
    dataset_y = dataset['is_churn'].values.tolist()
    msno_df = dataset[['msno']]
    del dataset['msno']
    del dataset['is_churn']
    dataset_x = dataset.values
    # print("msno_df: ", msno_df.shape)
    # print (msno_df.head())
    return (dataset_x, dataset_y, msno_df)


def processTestData():
    test = readData(filepath)

    # Merge all datasets
    members = Members.getProcessedData()
    test_dataset = pd.merge(left=test, right=members, how='left', on=['msno'])
    test_dataset = Members.imputeMissingValues(test_dataset)
    del members

    transactions = Transactions.getProcessedData()
    test_dataset = pd.merge(left=test_dataset, right=transactions, how='left', on=['msno'])
    test_dataset = Transactions.imputeMissingValues(test_dataset)
    del transactions

    user_logs = Preprocessing_user_logs.getProcessedDataForTest()
    test_dataset = pd.merge(left=test_dataset, right=user_logs, how='left', on=['msno'])
    del user_logs

    del test_dataset['Unnamed: 0']
    test_dataset = test_dataset.drop_duplicates(keep="first")
    test_dataset = test_dataset.dropna()

    writeData(test_dataset)

if __name__ == '__main__':
    processTestData()