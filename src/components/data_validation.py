import pandas as pd
import os
from src.logger import logging
from src.exception import MyException
import sys
import os
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.utils.main_utils import read_yaml_file
import json

class DataValidation:

    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_info = read_yaml_file(self.data_validation_config.SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    def read_data(self, file_path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise MyException(e,sys)
        
    def validate_all_columns(self, df:pd.DataFrame) -> bool:
        try:
            absent_columns = []
            expected_columns = self.schema_info[self.data_validation_config.columns_name_in_schema_file]
            for column in expected_columns:
                if column not in df.columns:
                    absent_columns.append(column)
            if len(absent_columns) > 0:
                return False
            return True
        except Exception as e:
            raise MyException(e,sys)
        

    def validate_numerical_columns(self, df:pd.DataFrame) -> bool:
        try:
            numerical_columns = self.schema_info[self.data_validation_config.numerical_columns_key]
            absent_numerical_columns = []
            for column in numerical_columns:
                if column not in df.columns:
                    absent_numerical_columns.append(column)
            if len(absent_numerical_columns) > 0:
                return False
            return True
        except Exception as e:
            raise MyException(e,sys)
        
    def validate_ouput_column(self, df:pd.DataFrame) -> bool:
        try:
            output_column = self.schema_info[self.data_validation_config.output_column_key]
            if output_column not in df.columns:
                return False
            return True
        except Exception as e:
            raise MyException(e,sys)
    
    def initiate_data_validation(self):
        try:
            #validation message
            validation_message = ""
            
            #Train and test file path
            logging.info("Starting: Read train and test files from data ingestion artifact")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info("Completed: Reading the train and test files from data ingestion artifact")

            #Vaidating all columns
            logging.info("Starting: Validation of all columns for train data")
            are_all_columns_present_in_train_set = self.validate_all_columns(train_df)
            if not are_all_columns_present_in_train_set:
                validation_message += f"Not all columns are Present in train data. "
            logging.info("Completed: Validation of all columns for train data")
            logging.info("Starting: Validation of all columns for test data")
            are_all_columns_present_in_test_set = self.validate_all_columns(test_df)
            if not are_all_columns_present_in_test_set:
                validation_message += f"Not all columns are Present in test data. "
            logging.info("Completed: Validation of all columns for test data")

            #Validating numerical columns
            logging.info("Starting: Validation of numerical columns for train data")
            are_all_numerical_columns_present_in_train_set = self.validate_numerical_columns(train_df)
            if not are_all_numerical_columns_present_in_train_set:
                validation_message += f"Not all numerical columns are Present in train data. "
            logging.info("Completed: Validation of numerical columns for train data")
            logging.info("Starting: Validation of numerical columns for test data")
            are_all_numerical_columns_present_in_test_set = self.validate_numerical_columns(test_df)
            if not are_all_numerical_columns_present_in_test_set:
                validation_message += f"Not all numerical columns are Present in test data. "
            logging.info("Completed: Validation of numerical columns for test data")

            #Vaidating output column
            logging.info("Starting: Validation of output column for train data")
            is_train_output_column_present = self.validate_ouput_column(train_df)
            if not is_train_output_column_present:
                validation_message += f"Output column not Present in train data. "
            logging.info("Completed: Validation of output column for train data")
            logging.info("Starting: Validation of output column for test data")
            is_test_output_column_present = self.validate_ouput_column(test_df)
            if not are_all_numerical_columns_present_in_test_set:
                validation_message += f"Output column not Present in test data. "
            logging.info("Completed: Validation of output column for test data")

            #Validation status
            logging.info("Checking final Validation status")
            validation_status_true_false = len(validation_message) == 0
            if validation_status_true_false:
                validation_status = "Pass"
            else:
                validation_status = "Fail"
            logging.info(f"Final Validation status is: {validation_status}")

            #Write the report file
            logging.info("Generating final Validation report")
            report = {
                'Validation_status': validation_status,
                "message": validation_message.strip()
            }

            #data validation artifact
            data_validation_artifact = DataValidationArtifact(
                report_file_path = self.data_validation_config.DATA_VALIDATION_REPORT_FILE_PATH,
                validation_result=validation_status,
                message=validation_message
            )
            logging.info("Starting: Saving the final Validation report")
            #Saving report
            os.makedirs(self.data_validation_config.DATA_VALIDATION_ARTIFACT_PATH, exist_ok=True)
            with open(self.data_validation_config.DATA_VALIDATION_REPORT_FILE_PATH, "w") as report_file:
                json.dump(report, report_file, indent=4)
            logging.info("Completed: Saving the final Validation report")

            return data_validation_artifact

        except Exception as e:
            raise MyException(e,sys)
    