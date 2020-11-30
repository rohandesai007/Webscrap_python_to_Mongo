import warnings

import matplotlib.pyplot as plt
import mpmath
import numpy as np
import pandas as pd
import pmdarima as pm
from matplotlib import pyplot
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from source.connection_url import mongo_url

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, \
    mean_squared_log_error


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
    # groupby_data_df['value_diff'] = groupby_data_df['value']
    groupby_data_df['value_diff'] = groupby_data_df['value_diff'].abs()  # fill null with 0 and take absolute value
    groupby_data_df = groupby_data_df.drop(columns=['value'])
    groupby_data_df = groupby_data_df.dropna()
    print(groupby_data_df.tail(30))
    # test_mpl(groupby_data_df)
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


# def test_mpl(data_for_dc):
# data_ml = def test_mpdata_for_dc
# plt.plot(data_ml['date'], data_ml['value_diff'])
# plt.show()


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
    plot_acf(x[1:], lags=20)
    plt.title('Autocorrelation Function')
    plt.show()


def plot_pacf_func_first_order(df):
    x = df
    plot_pacf(x[1:], lags=20)
    plt.show()
    pyplot.title('Partial Auto-correlation Function')


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
    residual = decomposition.resid
    plt.subplot(411)
    plt.plot(data_set_3, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.show()
    # plt.tight_layout()
    # there can be cases where an observation simply consisted of trend & seasonality. In that case, there won't be
    # any residual component & that would be a null or NaN. Hence, we also remove such cases.
    decomposedLogData = residual
    decomposedLogData.dropna(inplace=True)


def grid_search_sarima(data_set_3):
    train_pct_index = int(0.9 * len(data_set_3))
    train, test = data_set_3[:train_pct_index], data_set_3[
                                                train_pct_index:]  # Define the p, d and q parameters to take any value between 0 and 2

    p_values = range(1, 6)
    d_values = range(0, 3)
    q_values = range(1, 6)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                train, test = data_set_3[:train_pct_index], data_set_3[train_pct_index:]
                predictions = []
                for i in range(len(test)):
                    try:
                        model = ARIMA(train, order)
                        model_fit = model.fit(disp=0)
                        pred_y = model_fit.forecast()[0]
                        predictions.append(pred_y)
                        error = np.sqrt(mean_squared_error(test, predictions))
                        print('Arima%s RMSE = %.2f' % (order, error))
                    except:
                        continue


def arimamodel(data_set_3):
    autoarima_model = pm.auto_arima(data_set_3,
                                    start_p=1,
                                    start_q=1,
                                    test="adf",
                                    trace=True)
    return autoarima_model


def sarimamodel(data_set_3):
    train_data, test_data = data_set_3[:int(0.75 * (len(data_set_3)))], data_set_3[int(0.75 * (len(data_set_3))):]
    sautoarima_model = pm.auto_arima(train_data, trace=True, error_action='ignore',
                                     start_p=0, start_q=0, max_p=6, max_q=6, m=7,
                                     suppress_warnings=True, stepwise=True, seasonal=True)
    print(sautoarima_model.summary())
    sautoarima_model.fit(train_data)
    sautoarima_model.plot_diagnostics()
    plt.show()
    start_index = test_data.index.min()
    end_index = test_data.index.max()

    # Predictions
    pred = sautoarima_model.predict()
    pred = sautoarima_model.predict(n_periods=len(test_data))
    pred = pd.DataFrame(pred, index=test_data.index, columns=['Prediction'])

    forecast = sautoarima_model.predict(n_periods=len(test_data))
    forecast = pd.DataFrame(forecast, index=test_data.index, columns=['Prediction'])

    # plot the predictions for validation set
    plt.plot(data_set_3, label='Train')
    # plt.plot(valid, label='Valid')
    plt.plot(forecast, label='Prediction')
    plt.show()
    evaluate_forecast(data_set_3[start_index:end_index], forecast)
    return sautoarima_model


def evaluate_forecast(y,pred):
    results = pd.DataFrame({'r2_score':r2_score(y, pred),
                           }, index=[0])
    results['mean_absolute_error'] = mean_absolute_error(y, pred)
    results['median_absolute_error'] = median_absolute_error(y, pred)
    results['mse'] = mean_squared_error(y, pred)
    results['msle'] = mean_squared_log_error(y, pred)
    results['mape'] = mean_absolute_percentage_error(y, pred)
    results['rmse'] = np.sqrt(results['mse'])
    return results


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def predict(data_set_3):
    # split data into train and training set
    # train_data, test_data = data_set_3[3:int(len(data_set_3) * 0.9)], data_set_3[int(len(data_set_3) * 0.9):]
    train_data, test_data = data_set_3[:int(0.75 * (len(data_set_3)))], data_set_3[int(0.75 * (len(data_set_3))):]

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Death cases')
    plt.plot(data_set_3, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()
    # Build Model
    model = ARIMA(train_data, order=(5, 1, 4))
    fitted = model.fit(disp=-1)
    print(fitted.summary())

    # Forecast
    fc, se, conf = fitted.forecast(26, alpha=0.05)  # 95% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=test_data.index)
    lower_series = pd.Series(conf[:, 0], index=test_data.index)
    upper_series = pd.Series(conf[:, 1], index=test_data.index)
    # Plot
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(train_data, label='training')
    plt.plot(test_data, color='blue', label='Death Cases')
    plt.plot(fc_series, color='orange', label='Predicted Death Cases')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.10)
    plt.title('Death cases Prediction')
    plt.xlabel('Time')
    plt.ylabel('Death Cases')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    # report performance
    mse = mean_squared_error(test_data, fc)
    print('MSE: ' + str(mse))
    mae = mean_absolute_error(test_data, fc)
    print('MAE: ' + str(mae))
    rmse = mpmath.sqrt(mean_squared_error(test_data, fc))
    print('RMSE: ' + str(rmse))
    # Around 3.5% MAPE implies the model is about 96.5% accurate in predicting the next 15 observations.


data_set_1 = data_prep_death_cases(mongo_url)
split_data_and_calc_mean_variance(data_set_1)
data_set_2 = death_over_time_day_wise(data_set_1)
data_set_3 = first_order_estimate_trend(data_set_2)
calculate_moving_average_first_order(data_set_3)
plot_acf_func_first_order(data_set_3)
plot_pacf_func_first_order(data_set_3)
dickey_fuller_test_log(data_set_3)
plot_ts_decomposition(data_set_3)
# grid_search_sarima(data_set_3)
# arima_model = arimamodel(data_set_1)
# arima_model.summary()
sarimamodel(data_set_3)
predict(data_set_3)
