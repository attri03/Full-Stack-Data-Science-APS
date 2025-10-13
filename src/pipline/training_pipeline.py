import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact

class TrainPipeline:

    def __init__(self):
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_training_config = ModelTrainerConfig()
        except Exception as e:
            raise MyException(e, sys)
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the data ingestion method")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Exited the data ingestion method")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        try:
            logging.info("Entered the data validation method")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Exited the data validation method")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)
        
    def start_data_transformation(self, 
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact,
                                  ) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        try:
            logging.info("Entered the data transformation method")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_artifact=data_validation_artifact,
                                             data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Exited the data transformation method")
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)
        
    def start_model_training(self, 
                            data_transformation_artifact: DataTransformationArtifact,
                            ) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        try:
            logging.info("Entered the model training method")
            model_trainer = ModelTrainer(model_trainer_config=self.model_training_config,
                                        data_transformation_artifact=data_transformation_artifact
                                        )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Exited the model training method")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            logging.info("Starting the training pipeline")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact, data_validation_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact)
            logging.info("Completed the training pipeline")
        except Exception as e:
            raise MyException(e, sys)