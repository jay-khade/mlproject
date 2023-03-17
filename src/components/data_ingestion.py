# ------------------------------------------- Importing Libraries --------------------------------------------

import os
import sys # for custom exception
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # to create class variables. allows to define class variable without init.



# ----------------------------------------- Data Ingestion ----------------------------------------

@dataclass
class DataIngestionConfig:

    # defining paths where raw, train and test data will be stored
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # getting all the paths stored here.

    
    def initiate_data_ingestion(self):
        """
        Writing a code to read data from database
        """

        logging.info("Entered the Data ingestion method")
        try:
            # reading a dataset--- can be read from anywhere mogodb, Api, Clipboard etc.
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # making directory to save data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # saving raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Train test split
            logging.info("Train Test Split Initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # resturing train test data paths for data transformation
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()



