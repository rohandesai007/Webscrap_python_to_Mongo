import pymongo
from mysql import connector
from pymongo import MongoClient, collection
import pandas as pd
import re
import datetime
import mysql.connector
from pymongo.database import Database


def print_hi(name: object) -> object:
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

class DataFormat:

    def __init__(self, id, county, province_state, country_region, lat, lon, date, value):
        self.id = id
        self.county = county
        self.province_state = province_state
        self.country_region = country_region
        self.lat = lat
        self.lon = lon
        self.date = date
        self.value = value


def pass_date_param():
    today_date = datetime.date.today()
    yesterday_date = today_date - datetime.timedelta(days=1)
    formatted_date = datetime.date.strftime(yesterday_date, "X%m/X%d/%y").replace('X0', 'X').replace('X', '')
    return str(formatted_date)


def get_data_in_url(url, pattern):
    data_frame = pd.read_csv(url)
    data_objects = []
    re.compile(pattern)
    pattern_matcher = re.compile(pattern)
    dates = []
    for col in data_frame.columns:
        if pattern_matcher.match(str(col)) is not None:
            dates.append(str(col))

    for index, row in data_frame.iterrows():
        county = row['Admin2']
        state = row['Province_State']
        region = row['Country_Region']
        latitude = row['Lat']
        longit = row['Long_']
        for date in dates:
            documentId = str(county) + "_" + str(state) + "_" + str(region) + "_" + str(date)
            value = row[date]
            confirmed = DataFormat(documentId, county, state, region, latitude, longit, date, value)
            data_objects.append(vars(confirmed))
    return data_objects


def get_mongo_collection(url, collection_name):
    client = MongoClient(url)
    db = client.mydata
    return db[collection_name]


def insert_data_into_mongo(collection, data_objects):
    collection.insert_many(data_objects)


def load_case_data(url, mongo_url, collection_name, pattern):
    data_objects = get_data_in_url(url, pattern)
    collection = get_mongo_collection(mongo_url, collection_name)
    insert_data_into_mongo(collection, data_objects)
    linear_regression_model(mongo_url)


def linear_regression_model(mongo_url):
    client = MongoClient(mongo_url)
    db = client.mydata
    specified_collection = db["deaths_cases"]
    mydata = specified_collection.find({}, {"province_state": 1, "date": 1, "value": 1})
    df_deathcases = pd.DataFrame.from_records(list(mydata))
    print(df_deathcases)


url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                '/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv '
url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
            '/csse_covid_19_time_series/time_series_covid19_deaths_US.csv '
url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                '/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv '
# pattern = '.*/.*/.*'
mongo_url = 'mongodb+srv://my_data:test@cluster0.9q8b8.mongodb.net/mydata?retryWrites=true&w=majority'


pattern = pass_date_param()
load_case_data(url_death, mongo_url, 'deaths_cases', pattern)
load_case_data(url_confirmed, mongo_url, 'confirmed_cases', pattern)
load_case_data(url_recovered, mongo_url, 'recovered_cases', pattern)
