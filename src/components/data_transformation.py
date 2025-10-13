import pandas as pd
import numpy as np
import os
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
import sys
from src.utils.main_utils import *
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler

class DataTransformation:

    def __init__(self, 
                data_transformation_config: DataTransformationConfig,
                data_ingestion_artifact: DataIngestionArtifact,
                data_validation_artifact: DataValidationArtifact
                ):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.schema_info = read_yaml_file(self.data_transformation_config.SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    def read_file(self, file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    def dropping_unwanted_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            to_delete_columns = self.schema_info["to_delete_columns"]
            df = df.drop(columns=to_delete_columns, axis=1)
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    def mapping_output_column(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            output_column = self.schema_info["output_column"]
            df[output_column] = df[output_column].map({"neg":0, "pos":1})
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    def splitting_input_output_feature(self, df: pd.DataFrame):
        try:
            output_column = self.schema_info["output_column"]
            X = df.drop(columns=[output_column], axis=1)
            y = df[output_column]
            return X, y
        except Exception as e:
            raise MyException(e, sys)
        
    def pipeline_formation(self) -> Pipeline:
        try:
            # Pipeline with only imputer and scaler
            pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ]
            )
            return pipeline
        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting: Checking the data validation status before data transformation")
            #Checking validation status
            if self.data_validation_artifact.validation_result == False:
                raise Exception(self.data_validation_artifact.message)
            logging.info("Completed: Data validation status Pass")
            
            logging.info("Starting: Reading ingested data train and test file")
            #Reading training and testing file
            train_df = self.read_file(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_file(self.data_ingestion_artifact.test_file_path)
            logging.info("Completed: Reading ingested data train and test file")

            logging.info("Starting: Dropping unwanted columns from train and test file")
            #Dropping unwanted columns
            train_df = self.dropping_unwanted_columns(train_df)
            test_df = self.dropping_unwanted_columns(test_df)
            logging.info("Completed: Dropping unwanted columns from train and test file")

            logging.info("Starting: Mapping the output column")
            #Mapping output column
            train_df = self.mapping_output_column(train_df)
            test_df = self.mapping_output_column(test_df)
            logging.info("Completed: Mapping the output column")

            logging.info("Starting: Splitting the input and output features")
            #Splitting input and output features
            X_train_df, y_train_df = self.splitting_input_output_feature(train_df)
            X_test_df, y_test_df = self.splitting_input_output_feature(test_df)
            logging.info("Completed: Splitting the input and output features")

            logging.info("Starting: Transformation pipeline")
            #Pipeline formation
            pipeline = self.pipeline_formation()
            X_train_transformed = pipeline.fit_transform(X_train_df)
            X_test_transformed = pipeline.transform(X_test_df)
            logging.info("Completed: Transformation pipeline")

            logging.info("Starting: Handling Unbalance data")
            #Handling imbalanced dataset
            rus = RandomUnderSampler(random_state=42)
            X_train_res, y_train_res = rus.fit_resample(X_train_transformed, y_train_df)
            logging.info("Completed: Handling Unbalance data")
            
            logging.info("Starting: Concatinating the input and output columns")
            #Forming final train and test array
            train_arr = np.c_[X_train_res, y_train_res]
            test_arr = np.c_[X_test_transformed, y_test_df]
            logging.info("Completed: Concatinating the input and output columns")

            logging.info("Starting: Saving the Pipeline .pkl, transformed train and test file")
            #Saving numpy arrays
            os.makedirs(self.data_transformation_config.TRANSFORMED_DATA_ARTIFACT_DIR, exist_ok=True)
            os.makedirs(self.data_transformation_config.TRANSFORMED_PIPELINE_ARTIFACT_DIR, exist_ok=True)
            save_object(self.data_transformation_config.PIPELINE_FILE_PATH, pipeline)
            save_numpy_array_data(self.data_transformation_config.TRANSFORMED_TRAIN_DATA_FILE_PATH, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.TRANSFORMED_TEST_DATA_FILE_PATH, array=test_arr)
            logging.info("Completed: Saving the Pipeline .pkl, transformed train and test file")

            #data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file = self.data_transformation_config.TRANSFORMED_TRAIN_DATA_FILE_PATH,
                transformed_test_file = self.data_transformation_config.TRANSFORMED_TEST_DATA_FILE_PATH,
                pipeline_transformation_file = self.data_transformation_config.PIPELINE_FILE_PATH
            )

            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys)
