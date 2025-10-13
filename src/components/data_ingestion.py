from src.exception import MyException
import sys
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.data_access.proj1_data import Proj1Data
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)

    def store_raw_data(self):
        try:
            logging.info("Starting the raw data fetching from local directory")
            #Get data
            df_obj = Proj1Data()
            df = df_obj.get_data()
            logging.info("Completed the raw data fetching from local directory")
            logging.info("Starting saving the raw data")
            #save data
            os.makedirs(self.data_ingestion_config.RAW_DATA_ARTIFACT_DIR, exist_ok=True)
            df.to_csv(self.data_ingestion_config.RAW_DATA_ARTIFACT_FILE, index=False,header=True)
            logging.info("Completed saving the raw data")
            return df
        except Exception as e:
            raise MyException(e,sys)

    def data_pre_processing(self, data):
        try:
            logging.info("Starting the data pre-processing")
            for column in data.columns:
                if column != 'class':
                    data[column] = pd.to_numeric(data[column], errors="coerce")
            logging.info("Completed the data pre-processing")
            return data
        except Exception as e:
            raise MyException(e,sys)

    def train_test_split(self, data):
        try:
            logging.info("Starting the train test split")
            train_set, test_set = train_test_split(
                data, 
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,     # reproducibility
                stratify=data[self.data_ingestion_config.OUTPUT_FEATURE_FOR_MODEL]          # optional: preserve class distribution
            )
            logging.info("Completed the train test split")
            return train_set, test_set
        except Exception as e:
            raise MyException(e,sys)
        
    def store_train_test_data(self, train_set, test_set):
        try:
            logging.info("Starting saving the train and test data")
            os.makedirs(self.data_ingestion_config.PROCESSED_DATA_ARTIFACT_DIR, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE, index=False, header=True)
            logging.info("Completed saving the train and test data")
            return self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE, self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE
        except Exception as e:
            raise MyException(e,sys)

    def initiate_data_ingestion(self):
        try:
            raw_df = self.store_raw_data()
            processed_data = self.data_pre_processing(raw_df)
            train_set, test_set = self.train_test_split(processed_data)
            train_file_path, test_file_path = self.store_train_test_data(train_set, test_set)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=train_file_path,
                test_file_path=test_file_path
                )
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)