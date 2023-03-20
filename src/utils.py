# ----------------------------------------------- importing libraries -------------------------------------

import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


# ----------------------------------------- utility functions ----------------------------------------------

# Save object function
def save_object(file_path, obj):
    """
    Function to save objects/pkl files
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)


    except Exception as e:
        CustomException(e, sys)


# model evauation function
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    To evaluate all models
    """

    try:
        report = {}

        for i in range(len(list(models))):
            
            # selecting i'th model instance and its parameters
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # finding out best parameters
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            # applying best parameters
            model.set_params(**gs.best_params_)


            # model fitting
            model.fit(X_train,y_train)

            # predection on train side
            y_train_pred = model.predict(X_train)

            # predection on test side
            y_test_pred = model.predict(X_test)

            # evaluating model r2 score on train side
            train_model_score = r2_score(y_train, y_train_pred)

            # evaluating model r2 score on test side
            test_model_score = r2_score(y_test, y_test_pred)

            # creating report dictionary
            report[list(models.keys())[i]] = test_model_score

        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    

# pkl file loading function
def load_object(file_path):
    """
    To load pkl object
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)