import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import numpy as np


"""
Pre-process the Members dataset
"""

filepath = "data\members_v3.csv"
new_filename = "processed_data\processed_members_new.csv"

def getProcessedData():
    df = readData(new_filename)
    return df

def readData(filepath):
    print("Reading data from ",filepath)
    members = pd.read_csv(filepath)
    print(members.info())
    print(members.head())
    return members

def writeData(members):
    members.to_csv(new_filename, index=False)

def plotRegisteredVia(members):
    plt.figure(figsize=(12, 12))
    plt.subplot(412)
    R_V_order = members['registered_via'].unique()
    R_V_order = sorted(R_V_order, key=lambda x: str(x))
    R_V_order = sorted(R_V_order, key=lambda x: float(x))
    # above repetition of commands are very silly, but this was the only way I was able to display what I wanted
    sns.countplot(x="registered_via", data=members, order=R_V_order)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Registered Via', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of Registered Via Count", fontsize=12)
    plt.show()
    RV_count = Counter(members['registered_via']).most_common()
    print("Registered Via Count " + str(RV_count))

def plotCity(members):
    plt.figure(figsize=(12, 12))
    plt.subplot(411)
    city_order = members['city'].unique()
    city_order = sorted(city_order, key=lambda x: float(x))
    sns.countplot(x="city", data=members, order=city_order)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('City', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of City Count", fontsize=12)
    plt.show()
    city_count = Counter(members['city']).most_common()
    print("City Count " + str(city_count))

def plotGender(members):
    plt.figure(figsize=(12, 12))
    plt.subplot(413)
    sns.countplot(x="gender", data=members)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Gender', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of Gender Count", fontsize=12)
    plt.show()
    gender_count = Counter(members['gender']).most_common()
    print("Gender Count " + str(gender_count))

def plotBD(members):
    plt.figure(figsize=(12, 8))
    bd_order = members['bd'].unique()
    bd_order = sorted(bd_order, key=lambda x: str(x))
    bd_order = sorted(bd_order, key=lambda x: float(x))
    sns.countplot(x="bd", data=members, order=bd_order)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('BD', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of BD Count", fontsize=12)
    plt.show()
    bd_count = Counter(members['bd']).most_common()
    print("BD Count " + str(bd_count))

def plotTime(members):
    if 'registration_init_time_year' in members.columns.tolist():
        plt.figure(figsize=(12, 12))
        plt.subplot(311)
        year_order = members['registration_init_time_year'].unique()
        year_order = sorted(year_order, key=lambda x: str(x))
        year_order = sorted(year_order, key=lambda x: float(x))
        sns.barplot(members.index, members.values, order=year_order)
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('Year', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.title("Yearly Trend of registration_init_time", fontsize=12)
        plt.show()
        year_count = Counter(members['registration_init_time_year']).most_common()
        print("Yearly Count " + str(year_count))

    if 'registration_init_time_month' in members.columns.tolist():
        plt.figure(figsize=(12, 12))
        plt.subplot(312)
        month_order = members['registration_init_time_month'].unique()
        month_order = sorted(month_order, key=lambda x: str(x))
        month_order = sorted(month_order, key=lambda x: float(x))
        sns.barplot(members.index, members.values, order=month_order)
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.title("Monthly Trend of registration_init_time", fontsize=12)
        plt.show()
        month_count_2 = Counter(members['registration_init_time_month']).most_common()
        print("Monthly Count " + str(month_count_2))

def plotData(members):
    plotCity(members)
    plotBD(members)
    plotGender(members)
    plotRegisteredVia(members)

def printDistinctValues(members):
    print("members['city']: ", members['city'].unique())
    print("members['bd']: ", members['bd'].unique())
    print("members['gender']: ", members['gender'].unique())
    print("members['registered_via']: ", members['registered_via'].unique())
    print("members['registration_init_time_year']: ", members['registration_init_time_year'].unique())
    print("members['registration_init_time_month']: ", members['registration_init_time_month'].unique())

def fillEmptySlots(members):
    members['city'] = members['city'].apply(lambda x: int(x) if pd.notnull(x) else "NAN")
    members['registered_via'] = members['registered_via'].apply(lambda x: int(x) if pd.notnull(x) else "NAN")
    members['gender'] = members['gender'].fillna('NAN')
    members['bd'] = members['bd'].apply(lambda x: int(x) if pd.notnull(x) else "NAN")
    return members

def imputeMissingValues(members):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=True)
    temp = imp.fit_transform(members.loc[:, 'city':'registration_init_time_month'])
    temp_df = pd.DataFrame(temp)
    # temp_df = temp_df.apply(lambda x: round(float(x),2))
    members['city'] = temp_df.loc[:, 0]
    members['bd'] = temp_df.loc[:, 1]
    members['gender'] = temp_df.loc[:, 2]
    members['registered_via'] = temp_df.loc[:, 3]
    members['registration_init_time_year'] = temp_df.loc[:, 4]
    members['registration_init_time_month'] = temp_df.loc[:, 5]
    # members.loc[:, 'city':'registration_init_time_month'] = temp_df.loc[:, :]
    del temp_df
    return members


def introduceNewFeatures(members):
    members['registration_init_time'] = members['registration_init_time'].apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")

    # registration_init_time yearly trend
    members['registration_init_time_year'] = pd.DatetimeIndex(members['registration_init_time']).year
    members['registration_init_time_year'] = members['registration_init_time_year'].apply(
        lambda x: int(x) if pd.notnull(x) else "NAN")
    print("members['registration_init_time_year']: ", members['registration_init_time_year'].unique())

    # registration_init_time monthly trend
    members['registration_init_time_month'] = pd.DatetimeIndex(members['registration_init_time']).month
    members['registration_init_time_month'] = members['registration_init_time_month'].apply(
        lambda x: int(x) if pd.notnull(x) else "NAN")
    print("members['registration_init_time_month']: ", members['registration_init_time_month'].unique())

    del members['registration_init_time']


def adjustingOutliers(members):
    print("\nAdjusting Outliers...")
    members['registered_via'] = members['registered_via'].apply(lambda x: 0 if float(x) < 1 else x)
    members['bd'] = members.bd.apply(lambda x: 0 if (float(x) <= 1 or float(x) >= 100) else x)
    print("members['bd']: ",members['bd'].unique())

def normalizeFeatures(members):
    print("Normalizing numeric values...")
    cols = ['city', 'bd', 'registered_via','registration_init_time_year', 'registration_init_time_month']
    new_df = members[cols]
    new_df = np.log1p(new_df)
    ss = StandardScaler()
    data = ss.fit_transform(new_df)
    norm_df = pd.DataFrame(data=data, index = new_df.index, columns = new_df.columns)
    members[cols] = norm_df[cols]
    del new_df, norm_df
    # print(members.head())
    return members

def transformCategoricalData(members):
    print("\nTransforming categorical values to numeric...")
    members['gender'] = members['gender'].fillna('NAN')
    le = preprocessing.LabelEncoder()
    members['gender'] = le.fit_transform(members['gender'])
    print("members['gender']: ",members['gender'].unique())

def encodeCategoricalFeatures(members):
    print("Encoding numeric values...")
    dataset = members['gender']
    enc = OneHotEncoder(sparse=False)
    output = enc.fit_transform(dataset)
    output_df = pd.DataFrame(output)
    new_member_df = pd.concat([members, output_df], ignore_index=True, axis=1)
    # print(new_member_df.head())
    # print(new_member_df.shape)
    del output_df
    del members['gender']
    return new_member_df


def processMembersData():
    members = readData(filepath)

    adjustingOutliers(members)
    introduceNewFeatures(members)
    normalizeFeatures(members)
    transformCategoricalData(members)
    encodeCategoricalFeatures(members)

    writeData(members)
    # printDistinctValues(members)
    # plotData(members)
    # return members

if __name__ == '__main__':
    processMembersData()


