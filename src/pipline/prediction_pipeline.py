from src.entity.config_entity import APSSensorPredictorConfig
import os
import sys
from src.logger import logging
from src.exception import MyException
import pandas as pd
from src.utils.main_utils import *
from src.entity.s3_estimator import Proj1Estimator

class APSSensorDataFrame:

    def __init__(self, dictionary:dict):
        try:
            self.APSSensor_predictor_config = APSSensorPredictorConfig()
            self.schema_info = read_yaml_file(self.APSSensor_predictor_config.SCHEMA_FILE_PATH)
            self.data_dict = dictionary
        except Exception as e:
            raise MyException(e, sys)
        
    def dropping_unwanted_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            to_delete_columns = self.schema_info["to_delete_columns"]
            df = df.drop(columns=to_delete_columns, axis=1)
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    def dictionary_to_dataframe(self, dictionary: dict):
        try:
            df = pd.DataFrame.from_dict(dictionary)
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    def final_input_data(self):
        try:
            df = self.dictionary_to_dataframe(self.data_dict)
            df = self.dropping_unwanted_columns(df)
            return df
        except Exception as e:
            raise MyException(e, sys)
        
class APSSensorPredictor:

    def __init__(self, APSSensor_predictor_config = APSSensorPredictorConfig()):
        try:
            self.APSSensor_predictor_config = APSSensor_predictor_config
        except Exception as e:
            raise MyException(e, sys)
        
    def predict(self, df):
        try:
            model = Proj1Estimator(
                bucket_name=self.APSSensor_predictor_config.bucket_name,
                model_path=self.APSSensor_predictor_config.s3_model_key_path,
            )
            result =  model.predict(df)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)