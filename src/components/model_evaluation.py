from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import *
import os
import sys
from src.entity.s3_estimator import Proj1Estimator
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class EvaluateModelResponse:
    trained_model_recall_score: float
    best_model_recall_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:

    def __init__(self,model_eval_config:ModelEvaluationConfig,
                 data_ingestion_Artifact:DataIngestionArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_Artifact = data_ingestion_Artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.schema_info = read_yaml_file(self.model_eval_config.SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    def get_best_model(self):
        try:
            s3_model = Proj1Estimator(bucket_name=self.model_eval_config.bucket_name,
                                     model_path=self.model_eval_config.s3_model_key_path)
            if s3_model.is_model_present(model_path=self.model_eval_config.s3_model_key_path):
                return s3_model.load_model()
            else:
                return None
        except Exception as e:
            raise MyException(e, sys)
        
    def read_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df
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
        
    def evaluate_model(self,best_model,trained_model)->float:
        try:
            logging.info("Starting: Fetching the ingested untransformed test data for predicting")
            #Fetching the transformed test data
            test_data_path = self.data_ingestion_Artifact.test_file_path
            test_data = self.read_data(test_data_path)
            logging.info("Completed: Fetching the ingested untransformed test data for predicting")

            #Performing some pre-processing steps on the test data
            logging.info("Starting: Performing som pre-processing steps on the test data")
            test_data = self.dropping_unwanted_columns(test_data)
            test_data = self.mapping_output_column(test_data)
            X, y = self.splitting_input_output_feature(test_data)
            logging.info("Completed: Performing some pre-processing steps on the test data")

            logging.info("Starting: Checking the accuracy of best model from AWS")
            #Checking the accuracy of best model from AWS
            best_model_recall_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                best_model_output = best_model.predict(X)
                best_model_recall_score = recall_score(y, best_model_output)
            logging.info("Completed: Checking the accuracy of best model from AWS")
            
            logging.info("Starting: Fetching accuracy of new trained model from model trainer artifact")
            #Trained model accuracy
            trained_model_recall_score = self.model_trainer_artifact.metric_artifact.recall_score
            logging.info("Completed: Fetching accuracy of new trained model from model trainer artifact")

            logging.info("Generating the response after evaulation")
            #EvaluateModelResponse object
            evaluate_model_response = EvaluateModelResponse(
                trained_model_recall_score=trained_model_recall_score,
                best_model_recall_score=best_model_recall_score,
                is_model_accepted=True if best_model_recall_score is None else trained_model_recall_score > (best_model_recall_score + self.model_eval_config.changed_threshold_score),
                difference=0.0 if best_model_recall_score is None else trained_model_recall_score - best_model_recall_score
            )
            return evaluate_model_response

        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            logging.info("Starting: Getting best model from s3 bucket")
            # Get the best model from S3 bucket
            best_model = self.get_best_model()
            logging.info("Completed: Getting best model from s3 bucket")
            
            logging.info("Starting: Getting trained model from the model trainer artifact")
            # Get the trained model
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Completed: Getting trained model from the model trainer artifact")

            logging.info("Starting: Evaluating both the models and finding best one")
            # Evaluate the models
            evaluation_response = self.evaluate_model(best_model=best_model, trained_model=trained_model)
            logging.info("Completed: Evaluating both the models and finding best one")
            logging.info(f"Is new trained model accepted: {evaluation_response.is_model_accepted}")

            # Model Evaluation Artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation_response.is_model_accepted,
                changed_accuracy=evaluation_response.difference,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)
        
