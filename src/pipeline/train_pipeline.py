import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_trainer



if __name__ == '__main__': # Initializing the main python environment

    try:
        logging.info("Data Ingestion Initiated")
        obj=DataIngestion()  # initializing DataIngestion() class by creating a variable obj
        train_data_path,test_data_path = obj.initiate_data_ingestion()  # calling method initiate_data_ingestion()
        print(train_data_path,test_data_path)
        logging.info("Data Transformation Completed")

        logging.info("Data Transformation Initiated")
        data_transformation = DataTransformation() # initializing DataTranformation() class by creating a variable 'data_transformation'
        # calling method initiate_data_transfromation()
        train_arr,test_arr,obj_path=data_transformation.initiate_DataTransformation(train_data_path,test_data_path)
        print(obj_path)
        logging.info("Data Transformation Completed")

        logging.info("Model Trainer Initiated")
        model_trainer = Model_trainer()
        model_trainer.initiate_model_training(train_arr,test_arr)

    except Exception as e:
        logging.info("Error occured in training pipeline")
        raise CustomException (e,sys)

