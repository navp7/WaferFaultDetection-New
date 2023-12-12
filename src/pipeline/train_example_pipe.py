import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_trainer



if __name__ == '__main__':
    obj=DataIngestion()  # initializing DataIngestion() class by creating a variable obj
    train_data_path,test_data_path = obj.initiate_data_ingestion()  # calling method initiate_data_ingestion()
    print(train_data_path,test_data_path)

    data_transformation = DataTransformation() # initializing DataTranformation() class by creating a variable 'data_transformation'
    # calling method initiate_data_transfromation()
    train_arr,test_arr,obj_path=data_transformation.initiate_DataTransformation(train_data_path,test_data_path)
    print(obj_path)

    ModelTrainer_obj = Model_trainer()
    ModelTrainer_obj.initiate_model_training(train_arr=train_arr,test_arr=test_arr)
    print("Model training Completed")
