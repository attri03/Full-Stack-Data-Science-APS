from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.logger import logging
from src.exception import MyException
import sys
import os
from src.utils.main_utils import *
from src.entity.s3_estimator import Proj1Estimator

class ModelPusher:

    def __init__(self, model_pusher_config:ModelPusherConfig,
                 model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_model_push(self):
        try:
            logging.info("Starting: Loading the new trained model")
            # Get the new trained model
            new_trainer_model = load_object(file_path=self.model_evaluation_artifact.trained_model_path)
            logging.info("Completed: Loading the new trained model")

            logging.info("Starting: Uploading the model on AWS")
            #Getting Proj1Estimator
            proj1_estimator = Proj1Estimator(bucket_name=self.model_pusher_config.bucket_name,
                                             model_path=self.model_pusher_config.s3_model_key_path)
            
            
            #Saving the model
            proj1_estimator.save_model(self.model_evaluation_artifact.trained_model_path)
            logging.info("Completed: Uploading the model on AWS")

            #Artifact
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name = self.model_pusher_config.bucket_name,
                s3_model_path = self.model_pusher_config.s3_model_key_path
            )

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys)