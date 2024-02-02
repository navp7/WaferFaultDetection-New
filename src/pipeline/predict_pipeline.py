import shutil
import os,sys
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from flask import request
from src.constant import *
from src.utils import load_object
from dataclasses import dataclass
        
        
@dataclass
class PredictionPipelineConfig:
    ''' 
    Creating class to config prediction pipeline path to store output file
    '''
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    model_file_path: str = os.path.join('artifacts', "model.pkl" )
    preprocessor_path: str = os.path.join('artifacts', "preprocessor.pkl")
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)



class PredictionPipeline:
    '''
    Main class for Prediction Pipeline
    '''
    def __init__(self, request: request):
        # initializing 
        self.request = request
        self.prediction_pipeline_config = PredictionPipelineConfig()


    def save_input_files(self)-> str:
        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
        """
        try:
            #Reading the file
            input_csv_file = self.request.files['file']
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_file_path = os.path.join(pred_file_input_dir, 'input_file.csv')
            input_csv_file.save(input_file_path)

            return input_file_path

        except Exception as e:
            logging.info(f"Error occured while saving the input file")
            raise CustomException(e,sys)

    def predict(self, features):
        """
            Method Name :   predict
            Description :   This method makes the prediction for the uploaded test file & returns output. 
        """
        try:
            model = load_object(file_path=self.prediction_pipeline_config.model_file_path)
            preprocessor = load_object(file_path=self.prediction_pipeline_config.preprocessor_path)

            transformed_x = preprocessor.transform(features)

            preds = model.predict(transformed_x)
            
            return preds

        except Exception as e:
            logging.info("Error while making predictions.")
            raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):

        """
            Method Name :   get_predicted_dataframe
            Description :   This method returns the dataframe with a new column containing predictions

        """
        try:

            prediction_column_name : str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            
            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe
            # Dropping unwanted features:
            input_dataframe = input_dataframe.drop(columns = UNWANTED_COL)
            logging.info("Making the predictions.")
            predictions = self.predict(input_dataframe)
            logging.info("Predictions completed")
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'bad', 1:'good'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
            logging.info(f"Predicted File saved in {self.prediction_pipeline_config.prediction_file_path}.")

        except Exception as e:
            logging.info("Error occured while getting predicted dataframe")
            raise CustomException(e, sys)
        
        
    def run_pipeline(self):
        """
            Method Name :   run_pipeline
            Description :   This method runs pipeline for saving uploaded file and returning path for predicted file.
        """
        try:
            logging.info("Saving input file to the prediction_artifacts directory")
            input_csv_path = self.save_input_files()
            logging.info(f"Input file saved in: {input_csv_path}")
            
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_pipeline_config

        except Exception as e:
            raise CustomException(e,sys)
            
        

 
        

        