import pandas as pd
import numpy as np 
import os
import sys
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.utils.main_utils import *
from src.entity.estimator import MyModel
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)
        
    def custom_metric(self, y_true, y_pred) -> int:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cost = (self.model_trainer_config.fp_cost * fp) + (self.model_trainer_config.fn_cost * fn)
            return cost
        except Exception as e:
            raise MyException(e, sys)

    def build_model(self, train_df, test_df):
        try:
            logging.info("Starting : Splitting the input and output columns for both train and test data")
            # Split features and target
            X_train, y_train, X_test, y_test = train_df[:, :-1], train_df[:, -1], test_df[:, :-1], test_df[:, -1]
            logging.info("Completed : Splitting the inout and output columns for both train and test data")

            logging.info("Starting : Building the model")
            # Make the model
            model = XGBClassifier(random_state=42,
                                  learning_rate=self.model_trainer_config.model_learning_rate,
                                  max_depth=self.model_trainer_config.model_max_depth,
                                  n_estimators=self.model_trainer_config.model_n_estimators
                                  )
            model.fit(X_train, y_train)
            logging.info("Completed : Building the model")

            logging.info("Starting : Predicting the test data")
            # Make predictions
            y_pred = model.predict(X_test)
            logging.info("Completed : Predicting the test data")

            logging.info("Starting : Evaluating the model")
            #Evaluation on test data
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            total_cost = self.custom_metric(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            logging.info("Completed : Evaluating the model")

            # Prepare metric artifact
            metric_artifact = ClassificationMetricArtifact(
                                    total_cost = total_cost,
                                    f1_score = f1,
                                    precision_score = precision,
                                    recall_score = recall
                                )
            
            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:

            logging.info("Starting : Loading the train and test data")
            # Read transformed training and testing data
            train_df = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file)
            test_df = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file)
            logging.info("Completed : Loading the train and test data")

            logging.info("Starting : Model building phase")
            # build_model
            model, metric_artifact = self.build_model(train_df, test_df)
            logging.info("Completed : Model building phase")

            # Check if the model's accuracy meets the expected threshold
            if metric_artifact.total_cost > self.model_trainer_config.fp_cost*len(test_df):
                logging.info("No model found with less cost than the base cost")
                raise Exception("No model found with less cost than the base cost")

            logging.info("Starting : Loading the preprocessor")
            # Load preprocessor object
            preprocessor = load_object(self.data_transformation_artifact.pipeline_transformation_file)
            logging.info("Completed : Loading the preprocessor")

            logging.info("Starting : Saving the custom model")
            #Saving into MyModel
            my_model = MyModel(model, preprocessor)
            os.makedirs(self.model_trainer_config.model_saving_dir, exist_ok=True)
            save_object(self.model_trainer_config.model_file_path, my_model)
            logging.info("Completed : Saving the custom model")

            # Prepare ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                                        trained_model_file_path = self.model_trainer_config.model_file_path,
                                        metric_artifact = metric_artifact
            )

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)