import os, sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
from pymongo import MongoClient
from zipfile import Path
#from src.utils import MainUtils
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.constant import *
from src.configuration.mongo_db_connection import MongoDBClient


@dataclass
class DataIngestionconfig:
    '''
    creating a class to config paths for the train,test and raw
    data to stored separartely.
    '''
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw_waferfault.csv")
    

class DataIngestion:
    '''
    creating the main DataIngestion class.
    '''
    # Initializing data ingetion config. 
    def __init__(self):
        self.ingestionconfig = DataIngestionconfig()
        #self.utils = MainUtils()

    def export_collection_as_dataframe(self,collection_name, db_name):
        '''
        Method Name :   export_collection_as_dataframe
        Description :   This method collects the data from MongoDB and return as DataFrame
        '''
        try:
            mongo_client = MongoClient(MONGO_DB_URL)

            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():  # dropping the '_id' feature of MongoDB
                df = df.drop(columns=["_id"], axis=1)

            return df

        except Exception as e:
            logging.info("Error occured in exporting from MongoDB")
            raise CustomException(e, sys)

    

    # Initiating data ingetion process and returning paths. 
    def initiate_data_ingestion(self):
        '''
        Method Name :   initiate_data_ingestion
        Description :   This method stores the train, test and raw data and return their paths.
        '''
        try:
            logging.info(f"Exporting data from mongodb")

            sensor_data = self.export_collection_as_dataframe(
                                                              collection_name= MONGO_COLLECTION_NAME,
                                                              db_name = MONGO_DATABASE_NAME)
            

            raw_file_path  = self.ingestionconfig.raw_data_path               
            os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)

            sensor_data.to_csv(raw_file_path, index=False, header=True)
            logging.info(f"Saving exported data in: {raw_file_path}")
          
            train_set,test_set = train_test_split(sensor_data,test_size=0.2,random_state=85)

            train_set.to_csv(self.ingestionconfig.train_data_path,index=False, header=True)
            logging.info('Train Data is Created')

            test_set.to_csv(self.ingestionconfig.test_data_path, index=False, header = True)
            logging.info('Test Data is Created')
            
            return (
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path)

        except Exception as e:
            logging.info("The error is occured in data ingestion process")
            raise CustomException (e,sys)



