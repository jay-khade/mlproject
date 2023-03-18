#----------------------------------------------- Importing Libraries --------------------------------------

import os
import sys
from dataclasses import dataclass

#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

# ----------------------------------------------------- Model Training ----------------------------------------------

@dataclass
class ModelTrainerConfig:
    """
    defining path to save trained model pkl file
    """
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    """
    using different different algorithms to train model
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Split train and test input data")

            X_train, y_train, X_test, y_test = (

                train_array[:,:-1],# train data
                train_array[:,-1], # train data correct answers

                test_array[:,:-1], # test data
                test_array[:,-1], # test data correct answers
            )

            # making a dictionary of the all trying out models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                #"CatBoosting Classifier": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Hyper Parameter tuning
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbours Regressor":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # crating model evaluation report
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, 
                                                X_test=X_test, y_test=y_test, models=models, param=params)

            # to get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # picking best model
            best_model = models[best_model_name]

            # putting threshold
            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            
            logging.info(f"Best Model Found On Both Training And Testing Dataset.")

            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            
            # predection and evaluation
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square

            
        except Exception as e:
            raise CustomException(e,sys)