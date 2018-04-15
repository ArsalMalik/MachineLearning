import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import StandardScaler

"""
Pre-process the user_logs dataset
"""

filename = 'data\\user_logs.csv'
new_filename = "processed_data\processed_user_logs_new.csv"
new_filename_for_test = "processed_data\processed_user_logs_test.csv"
trainfile = 'data\\train.csv'
testfile = 'data\\sample_submission_zero.csv'

def readData():
    print("Reading User_log...")
    df_train = pd.read_csv(trainfile)
    train = pd.DataFrame(df_train['msno'])

    user_data = pd.DataFrame()
    for chunk in pd.read_csv(filename, chunksize=500000):
        merged = train.merge(chunk, on='msno', how='inner')
        user_data = pd.concat([user_data, merged])

    print(str(len(train['msno'].unique())) + 'unique members in Train')
    print(str(len(user_data['msno'].unique())) + "users have additional information")

    del train
    return user_data

def readDataForTest():
    print("Reading User_log...")
    df_test = pd.read_csv(testfile)
    test = pd.DataFrame(df_test['msno'])

    user_data = pd.DataFrame()
    for chunk in pd.read_csv(filename, chunksize=500000):
        merged = test.merge(chunk, on='msno', how='inner')
        user_data = pd.concat([user_data, merged])

    print(str(len(test['msno'].unique())) + 'unique members in Test')
    print(str(len(user_data['msno'].unique())) + "users have additional information")

    del test
    return user_data

def preprocessUserLogs(user_data):
    print("Started Preprocessing...")
    print(user_data.head())

    # checking number of outliers
    for col in user_data.columns[1:]:
        outliers = user_data['msno'][user_data[col]<0].count()
        print(str(outliers)+" outliers in " + col)


    # deleting column date
    del user_data['date']

    # checking correlation heatmap
    corrmat = user_data[user_data.columns[1:]].corr()
    f, ax = plt.subplots(figsize=(12,9))
    sbn.heatmap(corrmat, vmax=1, cbar=True, annot=True, square=True)
    plt.show()

    # Heavily correlated columns num_75, num_50 = 0.95
    # num_unq, num_100 = 0.92
    # Removing one column from each pair
    del user_data['num_75']
    del user_data['num_unq']


    # removing outliers from 'total_secs'
    user_data = user_data[user_data['total_secs'] >= 0]
    # print (user_data['msno'][user_data['total_secs'] < 0].count())

    # Grouping data by member id and summing up the columns corresponding to each member
    counts = user_data.groupby('msno')['total_secs'].count().reset_index()
    counts.columns = ['msno', 'days_listened']
    sums = user_data.groupby('msno').sum().reset_index()
    user_data = sums.merge(counts, how='inner', on='msno')

    # Updated shape of data
    # print (str(np.shape(user_data)) + " -- New size of data matches unique member count")

    # print(user_data.head())

    cols = user_data.columns[1:]

    # getting natural log of the data to make its distribution normal
    log_user_data = user_data.copy()
    log_user_data[cols] = np.log1p(user_data[cols])

    ss = StandardScaler()
    # Normalize (mean=0, std = 1)
    log_user_data[cols] = ss.fit_transform(log_user_data[cols])

    # check distribution of data, before and after normalization
    '''
    for col in cols:
        plt.figure(figsize=(15,7))
        plt.subplot(1,2,1)
        sbn.distplot(user_data[col].dropna())
        plt.subplot(1,2,2)
        sbn.distplot(log_user_data[col].dropna())
        plt.figure()
    '''
    return log_user_data


def getProcessedDataForTrain():
    user_log = pd.read_csv(new_filename)
    return user_log

def getProcessedDataForTest():
    user_log = pd.read_csv(new_filename_for_test)
    return user_log

def getUserLogForTrain():
    # Take the users that appear in both user_log data and Train file
    user_data = readData()
    df = preprocessUserLogs(user_data)
    df.to_csv(new_filename)


def getUserLogForTest():
    # Take the users that appear in both user_log data and Test file
    user_data = readDataForTest()
    df = preprocessUserLogs(user_data)
    df.to_csv(new_filename_for_test)

if __name__ == '__main__':
    getUserLogForTrain()
    getUserLogForTest()