from pymongo import MongoClient, collection


def get_mongo_collection(url, collection_name):
    client = MongoClient(url)
    db = client.mydata
    return db[collection_name]


def insert_data_into_mongo(collection, data_objects):
    collection.insert_many(data_objects)