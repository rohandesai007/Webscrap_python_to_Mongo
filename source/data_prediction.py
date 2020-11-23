import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pymongo import MongoClient
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from source.connection_url import mongo_url
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from random import random
from pandas.plotting import autocorrelation_plot
import pmdarima as pm


def data_prep_death_cases(mongo_url):
    client = MongoClient(mongo_url)
    db = client.mydata
    specified_collection = db["deaths_cases"]  # collection name for data prep
    mydata = specified_collection.find({}, {"date": 1, "value": 1})
    df_deathcases = pd.DataFrame.from_records(list(mydata))  # convert dict to list df
    df_deathcases['date'] = pd.to_datetime(df_deathcases['date'], format='%m/%d/%y')  # format date
    groupby_data_df = df_deathcases.groupby('date', as_index=True)['value'].sum().reset_index().sort_values(by=['date'],
                                                                                                            ascending=True)
    # df1.drop(df1.tail(7).index, inplace=True)
    groupby_data_df['value_diff'] = groupby_data_df['value'] - groupby_data_df['value'].shift(-1)
    groupby_data_df['value_diff'] = groupby_data_df['value_diff'].abs()  # fill null with 0 and take absolute value
    groupby_data_df = groupby_data_df.drop(columns=['value'])
    groupby_data_df = groupby_data_df.dropna()
    print(groupby_data_df.tail(30))
    test_mpl(groupby_data_df)
    return groupby_data_df


def split_data_and_calc_mean_variance(df1):
    x = df1['value_diff']
    split = int(len(x) / 5)
    x1 = x[0:split]
    x2 = x[split:split * 2]
    x3 = x[split * 2:split * 3]
    x4 = x[split * 3:split * 4]
    x5 = x[split * 4:]
    mean1, mean2, mean3, mean4, mean5 = x1.mean(), x2.mean(), x3.mean(), x4.mean(), x5.mean()
    var1, var2, var3, var4, var5 = x1.var(), x2.var(), x3.var(), x4.var(), x5.var()
    print('\n')
    print('mean1=%.2f, mean2=%.2f, mean3=%.2f, mean4=%.2f' % (mean1, mean2, mean3, mean4))
    print('variance1=%.2f, variance2=%.2f, variance3=%.2f, variance4=%.2f\n' % (var1, var2, var3, var4))


def test_mpl(data_for_dc):
    data_ml = data_for_dc
    plt.plot(data_ml['date'], data_ml['value_diff'])
    plt.show()


def death_over_time_day_wise(data_for_dc):
    data_ml = data_for_dc
    describe_data = data_ml.describe()
    print(describe_data)
    data_ml.set_index(['date'], inplace=True)
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title("Death cases over Time - Day Wise")
    plt.plot(data_ml, label='original Series')
    plt.legend(loc='best')
    plt.show()
    diff_first = data_ml.diff()
    estimate_trend_log_of_original(data_ml)
    return diff_first


def estimate_trend_log_of_original(var3):
    print("In Estimate trend function")
    df = var3
    df = df[(df[['value_diff']] != 0).all(axis=1)]
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title("Log of Death cases over Time - moving Average and moving STD")
    index_dataset_log = np.log(df)
    plt.plot(index_dataset_log)
    moving_average = index_dataset_log.rolling(window=30).mean()
    moving_std = index_dataset_log.rolling(window=30).std()
    plt.plot(moving_average, color='orange', label='Rolling Average')
    plt.legend(loc='best')
    plt.plot(moving_std, color='red', label='Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.show()
    plot_acf_func_of_log(index_dataset_log)
    return index_dataset_log


def plot_acf_func_of_log(index_dataset_log):
    x = index_dataset_log
    plot_acf(x[1:])
    plt.show()


def first_order_estimate_trend(diff_first):
    print("In Estimate trend function \n")
    df = diff_first
    print(df.tail(30))
    df = df[(df[['value_diff']] != 0).all(axis=1)]
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title("Death cases over Time - First order")
    plt.plot(df)
    plt.show()
    df = df.dropna()
    return df


def calculate_moving_average_first_order(df):
    moving_average = df.rolling(window=30).mean()
    moving_std = df.rolling(window=30).std()
    plt.plot(moving_average, color='orange', label='Rolling Average')
    plt.legend(loc='best')
    plt.plot(moving_std, color='red', label='Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.plot(df)
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title("First order Death cases over Time - moving Average and moving STD")
    plt.show()


def plot_acf_func_first_order(df):
    x = df
    plot_acf(x[1:], lags=7)
    plt.show()
    plt.show()


def plot_pacf_func_first_order(df):
    x = df
    plot_pacf(x[1:], lags=7)
    plt.show()


def dickey_fuller_test_log(data_set_3):
    print("\n")
    print("Dickey Fuller Test for Log Data")
    x = data_set_3
    result = adfuller(x)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    # p - value <= 0.05: Reject the null hypothesis(H0), the data does not have a unit  root and is stationary. When
    # the test statistic is lower than the critical value shown, you reject the null hypothesis and infer that the
    # time series is stationary.


def plot_ts_decomposition(data_set_3):
    decomposition = seasonal_decompose(data_set_3, model='additive', period=7)

    trend = decomposition.trend
    seasonal = decomposition.seasonal

    plt.subplot(411)
    plt.plot(data_set_3, label='Original')
    plt.legend(loc='best')

    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')

    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')

    plt.tight_layout()

    # there can be cases where an observation simply consisted of trend & seasonality. In that case, there won't be
    # any residual component & that would be a null or NaN. Hence, we also remove such cases.



def ar_model_ts(data_set_3):
    # AR example
    data = data_set_3.dropna()
    # fit model
    model = AutoReg(data, lags=7)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)


data_set_1 = data_prep_death_cases(mongo_url)
split_data_and_calc_mean_variance(data_set_1)
data_set_2 = death_over_time_day_wise(data_set_1)
data_set_3 = first_order_estimate_trend(data_set_2)
calculate_moving_average_first_order(data_set_3)
plot_acf_func_first_order(data_set_3)
plot_pacf_func_first_order(data_set_3)
dickey_fuller_test_log(data_set_3)
plot_ts_decomposition(data_set_3) after TS
ar_model_ts(data_set_3)
