import os, sys
import pandas as pd 
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from src.utils import save_object,unwanted_feat
from src.constant import *


@dataclass
class DataTransformation_config:
    '''
    creating a class to config path for the preprocessor object
    '''
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    '''
    creating a main DataTransformation class.
    '''

    # Initializing data transformation config.
    def __init__(self):
        self.data_transformation_config = DataTransformation_config()
        #self.utils = MainUtils()

    # creating method to get a preprocessor object.
    def get_DataTransformation_obj(self):
        '''
        Method Name :   get_DataTransformation_obj
        Description :   This method conatins the pipeline for transformation and it returns a preprocessor object.
        '''
        try:
            #Creating Pipeline Steps:

            preprocessor = Pipeline(
                steps=[ ('imputer',KNNImputer(n_neighbors=3)),
                        ('scaler',RobustScaler())
                        ])

            return preprocessor
                
        except Exception as e:
            logging.info("Error is occured in creating data transformation object")
            raise CustomException (e,sys)

    # creating method to initiate DataTransformation process.
    # It returns train and test arrays for model training.
    def initiate_DataTransformation(self,train_data_path,test_data_path):
        '''
        Method Name :   initiate_DataTransformation
        Description :   This method read the train and test data and returns preprocessed data.
        '''
        try:
            train_df =  pd.read_csv(train_data_path)
            test_df =  pd.read_csv(test_data_path)

            train_df.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
            test_df.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)

            logging.info('Train and test data read sucessful')
            logging.info(f'Train DataFrame Head: /n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: /n{test_df.head().to_string()}')

            logging.info("Preprocessing Initiated")

            col = unwanted_feat(data=train_df,missing=0.75)
            logging.info(f"Dropping unwanted features: /n{col}")

            df_train_2 = train_df.drop(columns=col,axis=1)
            df_test_2 = test_df.drop(columns=col,axis =1)

            #train dataframe
            input_train_df = df_train_2.drop(columns=[TARGET_COLUMN], axis=1)
            target_train_df = np.where(df_train_2[TARGET_COLUMN]==-1,0,1)

            #test dataframe
            input_test_df = df_test_2.drop(columns=[TARGET_COLUMN], axis=1)
            target_test_df = np.where(df_test_2[TARGET_COLUMN]==-1,0,1)

            #Obtaining Preprocessor File/Object
            preprocessing_obj = self.get_DataTransformation_obj()

            ## Data Transformation:
            input_train_df_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_df_arr = preprocessing_obj.transform(input_test_df)

            #Resampling
            ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
            

            input_train_df_final, target_train_df_final = ros.fit_resample(
               input_train_df_arr, target_train_df)
            
            input_test_df_final, target_test_df_final = ros.fit_resample(input_test_df_arr, target_test_df)

            logging.info("Data Resampling completed")

            train_arr = np.c_[input_train_df_final,np.array(target_train_df_final)]
            test_arr = np.c_[input_test_df_final,np.array(target_test_df_final)]

            logging.info("Preprocessing Completed")            
             
            # Saving preprocessing_obj at the artifacts destination:
            logging.info("Saving Preprocessor File")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                      obj=preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error Occured in Data transformation") 
            raise CustomException (e,sys) 