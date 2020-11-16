import pandas as pd
import re
from source.data_objects import DataFormat
from source.mongo_helper import get_mongo_collection, insert_data_into_mongo


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


def load_case_data(url, mongo_url, collection_name, pattern):
    data_objects = get_data_in_url(url, pattern)
    collection = get_mongo_collection(mongo_url, collection_name)
    insert_data_into_mongo(collection, data_objects)
