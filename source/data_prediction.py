import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from statsmodels.tsa.stattools import adfuller


def data_prep_death_cases(mongo_url):
    client = MongoClient(mongo_url)
    db = client.mydata
    specified_collection = db["deaths_cases"]  # collection name for data prep
    mydata = specified_collection.find({}, {"date": 1, "value": 1})
    df_deathcases = pd.DataFrame.from_records(list(mydata))  # convert dict to list df
    df_deathcases['date'] = pd.to_datetime(df_deathcases['date'], format='%m/%d/%y')  # format date
    df1 = df_deathcases.groupby('date', as_index=True)['value'].sum().reset_index().sort_values(by=['date'],
                                                                                                ascending=True)
    df1.drop(df1.tail(7).index, inplace=True)
    df1['value_diff'] = df1['value'] - df1['value'].shift(-1)
    df1['value_diff'] = df1['value_diff'].abs()  # fill null with 0 and take absolute value
    df1 = df1.drop(columns=['value'])
    print(df1.tail(30))
    return df1


def plot_death_over_time(data_for_dc):
    # client = MongoClient(mongo_url)
    # db = client.mydata
    data_ml = data_for_dc
    describe_data = data_ml.describe()
    print(describe_data)
    data_ml.set_index(['date'], inplace=True)
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title("Death cases over Time")
    x = plt.plot(data_ml)
    plt.show()
    # specified_collection1 = db["deaths_cases_stats"]
    # specified_collection1.insert_many(describe_data)
    # specified_collection2 = db["deaths_cases_over_time"]
    # specified_collection2.insert_many(x)
    print("will call dickey now")
    dickey_fuller_test(data_ml)
    estimate_trend(data_ml)
    return data_ml


def dickey_fuller_test(var2):
    print("In Dickey function")
    X = var2.values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def estimate_trend(var3):
    print("In Estimate trend function")
    df = var3
    df = df[(df[['value_diff']] != 0).all(axis=1)]
    index_dataset_log = np.log(df)
    plt.plot(index_dataset_log)
    plt.show()
    calculate_moving_average(index_dataset_log)
    return df


def calculate_moving_average(var4):
    df = var4
    moving_average = df.rolling(window=30).mean()
    moving_std = df.rolling(window=30).std()
    plt.plot(moving_average)
    plt.plot(moving_std, color='red')
    plt.show()