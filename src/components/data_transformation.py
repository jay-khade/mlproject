# ------------------------------- Importing Libriries -----------------------------------

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # use to create pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ---------------------------------------- Data Transformation ----------------------------------

@dataclass
class DataTransformationConfig:
    """
    To Get any input required. eg. Paths
    """

    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl') # data transformation componant which will transform the data


class DataTransformation:

    """
    transforming the input data to suitable format.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
        creating pickel files. will be responsible to data transformation.
        """

        try:
            
            # mentioning categorical and numerical features
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            # creating pipeline to process numerical data and defining steps of process such as missing value, scaling
            num_pipeline = Pipeline(
                    steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                    ]
            )

            
            #logging.info("numerical columns scaling completed")

            # crating pipeline to process categorical data and defining steps of process such as missing value, encoding etc
            cat_pipeline = Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                    ]
            )

            logging.info(f"numerical columns: {numerical_columns}")
            logging.info(f"categorical columns: {categorical_columns}")
            #logging.info("Categorical columns encoding completed")

            # Combining both num and cat pipeline
            preprocessor = ColumnTransformer(
                    [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                    ]
            )

            return preprocessor
                
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Here data transformation process will begin.
        """

        try:

            #reading train & test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train & test data completed.")
            logging.info("Obtaining preprocessing object.")

            # creating preprocessing obj ---- converting it into a pickle file
            preprocessing_obj = self.get_data_transformer_obj()

            # defining column types
            target_column_name = "math_score"
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            # dropping target column from train data
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            # dropping target column from test data
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessing object on training dataframe and testing dataframe")

            # apply preprocessing on train and test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # converting into a suitable format
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                ]
            
            test_arr = np.c_[
                 input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            # save okl file
            save_object(
                file_path =self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
                )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

            
        
            
