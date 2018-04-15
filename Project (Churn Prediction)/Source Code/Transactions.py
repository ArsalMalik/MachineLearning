import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import seaborn as sns
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

"""
Pre-process the transactions dataset
"""

filepath = "data\\transactions.csv"
new_filename = "processed_data\processed_transactions_new.csv"

def getProcessedData():
    df = readData(new_filename)
    return df

def readData(filepath):
    print("Reading data from ", filepath)
    transactions = pd.read_csv(filepath)
    print(transactions.info())
    print(transactions.head())
    return transactions

def writeData(transactions):
    transactions.to_csv(new_filename, index=False)

def plotPMID(transactions):
    plt.figure(figsize=(18, 6))
    sns.countplot(x="payment_method_id", data=transactions)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('payment_method_id', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of payment_method_id Count in transactions Data Set", fontsize=12)
    plt.show()
    payment_method_id_count = Counter(transactions['payment_method_id']).most_common()
    print("payment_method_id Count " + str(payment_method_id_count))

def plotPaymentPlanDays(transactions):
    plt.figure(figsize=(18, 6))
    sns.countplot(x="payment_plan_days", data=transactions)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('payment_plan_days', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of payment_plan_days Count in transactions Data Set", fontsize=12)
    plt.show()
    payment_plan_days_count = Counter(transactions['payment_plan_days']).most_common()
    print("payment_plan_days Count " + str(payment_plan_days_count))

def plotPlanListPrice(transactions):
    plt.figure(figsize=(18, 6))
    sns.countplot(x="plan_list_price", data=transactions)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('plan_list_price', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of plan_list_price Count in transactions Data Set", fontsize=12)
    plt.show()
    plan_list_price_count = Counter(transactions['plan_list_price']).most_common()
    print("plan_list_price Count " + str(plan_list_price_count))

def plotActualAmountPaid(transactions):
    plt.figure(figsize=(18, 6))
    sns.countplot(x="actual_amount_paid", data=transactions)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('actual_amount_paid', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of actual_amount_paid Count in transactions Data Set", fontsize=12)
    plt.show()
    actual_amount_paid_count = Counter(transactions['actual_amount_paid']).most_common()
    print("actual_amount_paid Count " + str(actual_amount_paid_count))

def plotIsAutoRenew(transactions):
    plt.figure(figsize=(4, 4))
    sns.countplot(x="is_auto_renew", data=transactions)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('is_auto_renew', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of is_auto_renew Count in transactions Data Set", fontsize=6)
    plt.show()
    is_auto_renew_count = Counter(transactions['is_auto_renew']).most_common()
    print("is_auto_renew Count " + str(is_auto_renew_count))

def plotIsCancel(transactions):
    plt.figure(figsize=(4, 4))
    sns.countplot(x="is_cancel", data=transactions)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('is_cancel', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of is_cancel Count in transactions Data Set", fontsize=6)
    plt.show()
    is_cancel_count = Counter(transactions['is_cancel']).most_common()
    print("is_cancel Count " + str(is_cancel_count))

def plotData(transactions):
    plotPMID(transactions)
    plotPaymentPlanDays(transactions)
    plotPlanListPrice(transactions)
    plotActualAmountPaid(transactions)
    plotIsAutoRenew(transactions)
    plotIsCancel(transactions)

def printDistinctValues(transactions):
    print("transactions['payment_method_id']: ", transactions['payment_method_id'].unique())
    print("transactions['payment_plan_days']: ", transactions['payment_plan_days'].unique())
    print("transactions['plan_list_price']: ", transactions['plan_list_price'].unique())
    print("transactions['actual_amount_paid']: ", transactions['actual_amount_paid'].unique())

def fillEmptySlots(transactions):
    transactions['payment_plan_days'] = transactions['payment_plan_days'].fillna('NAN')
    transactions['payment_method_id'] = transactions['payment_method_id'].fillna('NAN')
    transactions['plan_list_price'] = transactions['plan_list_price'].fillna('NAN')
    return transactions

def imputeMissingValues(transactions):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=True)
    temp = imp.fit_transform(transactions.loc[:, 'payment_method_id':'membership_duration_days'])
    temp_df = pd.DataFrame(temp)
    # temp_df = temp_df.apply(lambda x: round(float(x), 2))
    transactions['payment_method_id'] = temp_df.loc[:, 0]
    transactions['payment_plan_days'] = temp_df.loc[:, 1]
    transactions['plan_list_price'] = temp_df.loc[:, 2]
    transactions['is_auto_renew'] = temp_df.loc[:, 3]
    transactions['is_cancel'] = temp_df.loc[:, 4]
    transactions['membership_duration_days'] = temp_df.loc[:, 5]
    del temp_df
    return transactions


def someObservations(transactions):
    # Correlation between plan_list_price and actual_amount_paid
    print("Correlation between plan_list_price and actual_amount_paid: ", transactions['plan_list_price'].corr(transactions['actual_amount_paid'], method='pearson'))
    del transactions['actual_amount_paid']


def introduceNewFeatures(transactions):
    transactions['transaction_date'] = transactions['transaction_date'].apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")
    transactions['membership_expire_date'] = transactions['membership_expire_date'].apply(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")

    transactions['membership_duration_days'] = (transactions['membership_expire_date'] - transactions['transaction_date'])
    transactions['membership_duration_days'] = transactions['membership_duration_days'].apply(lambda x: int(re.search(r'\d+', str(x)).group()))

    print("transactions['membership_duration_days']: ", transactions['membership_duration_days'].unique())

    del transactions['transaction_date']
    del transactions['membership_expire_date']


def normalizeFeatures(transactions):
    print("Normalizing numeric values...")
    cols = ['payment_method_id', 'payment_plan_days', 'plan_list_price', 'membership_duration_days']
    new_df = transactions[cols]
    new_df = np.log1p(new_df)
    ss = StandardScaler()
    data = ss.fit_transform(new_df)
    norm_df = pd.DataFrame(data=data, index = new_df.index, columns = new_df.columns)
    transactions[cols] = norm_df[cols]
    del new_df, norm_df
    # print(transactions.head())
    return transactions

def encodeCategoricalFeatures(transactions):
    # df = transactions.loc[1:100, :]
    df = transactions
    print("Encoding numeric values...")
    cols = ['is_auto_renew', 'is_cancel']
    new_df = df[cols]
    enc = OneHotEncoder(sparse=False)
    data = enc.fit_transform(new_df)
    output_df = pd.DataFrame(data=data)
    for col in cols:
        del df[col]
    df = pd.concat([df, output_df], axis=1)
    # df.rename(columns={'0': 'is_auto_renew1', '1': 'is_auto_renew2'}, inplace=True)
    print(df.head())
    return df


def processTransactionsData():
    transactions = readData(filepath)
    someObservations(transactions)
    introduceNewFeatures(transactions)
    normalizeFeatures(transactions)
    encodeCategoricalFeatures(transactions)
    writeData(transactions)
    # printDistinctValues(transactions)
    # plotData(transactions)
    # return transactions


if __name__ == '__main__':
    processTransactionsData()

