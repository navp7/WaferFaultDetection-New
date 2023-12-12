import os
import sys
import yaml
import pickle
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from src.logger import logging
from src.exception import CustomException
from typing import List
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


class MainUtils:
    def __init__(self) -> None:
        pass
    
    # Reading the Yaml file and storing it in dictionary form with help of yaml.safe_load():
    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e
    
    # Reading yaml file through read_yaml_file() and storing it in schema_config file:
    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            raise CustomException(e, sys) from e


def save_object(file_path, obj):
    '''
    save_object function is used to save files at the given path. 
    It takes 2 arguments: 
    file_path -- path to save the file in dir 
    obj -- file object which needed to be saved
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error occured in saving pkl file")
        raise CustomException(e, sys)


def load_object(file_path):
    '''
    load_object function is used to load files from their path whenever it is called.
    It takes only 1 argument: 
    file_path -- path where the file is stored in dir  
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        logging.info("Error occured in loading pkl file")
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    '''
    evalaute_models is used to train the data across a list of models by performing 
    Hyperparameter tunning and finding the best parameter within it. It returns the 
    accuracy report as well to figure out the best model to be deployed.
    '''
    try:
        report={}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]
            #randomcv = RandomizedSearchCV(estimator = model, param_distributions=param, n_iter=10, cv = 5)
            gs = GridSearchCV(estimator=model,param_grid=param, cv=3)
            #randomcv.fit(X_train,y_train)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #Make Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred=model.predict(X_test)
    
            #cl_report = classification_report(y_test, y_test_pred)
            con_mat = confusion_matrix(y_test, y_test_pred)
            roc_score = roc_auc_score(y_test, y_test_pred)
            test_acc_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_acc_score

        return report

    except Exception as e:
        logging.info("Error occured while model training and hyperparameter tunning")
        raise CustomException (e,sys)


def unwanted_feat(data,missing):
    '''
    unwanted_feat computes and returns the  features which have missing values greater than certain percentage
    or the features which have standard deviation as 0.
    '''
    try:
        feature = data.columns
        col = []
        for i in feature:
            if (data[i].std() == 0) or (data[i].isna().mean() > missing):
                col.append(i)

        return col
    except Exception as e:
        logging.info("Error occured while removing unwanted features")
        raise CustomException (e,sys)