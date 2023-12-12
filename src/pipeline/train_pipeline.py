import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_trainer


class TrainPipeline:
    '''
    Main class for Training Pipeline
    '''
    def __init__(self):
        self.data_ingestion = DataIngestion()

        self.data_transformation = DataTransformation()

        self.model_trainer = Model_trainer() 

    def run_pipeline(self):
        """
            Method Name :   run_pipeline
            Description :   This method runs the pipeline and perform the training. 
            
        """
        try:
            logging.info("Data Ingestion Initiated")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion Completed")

            logging.info("Data Transformation Initiated")
            train_arr,test_arr,preprocessor_path = self.data_transformation.initiate_DataTransformation(train_data_path=train_path,test_data_path=test_path)
            logging.info("Data Transformation Completed")

            logging.info("Model Trainer Initiated")
            best_model_name,best_acc_score = self.model_trainer.initiate_model_training(train_arr,test_arr)

        except Exception as e:
            logging.info("Error occured in training pipeline")
            raise CustomException (e,sys)

