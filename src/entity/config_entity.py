from dataclasses import dataclass
from datetime import datetime
from src.constants import *
import os

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(artifact_dir, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    DATA_INGESTION_ARTIFACT_PATH: str = os.path.join(training_pipeline_config.artifact_dir, data_ingestion_artifact)
    RAW_DATA_ARTIFACT_DIR: str = os.path.join(DATA_INGESTION_ARTIFACT_PATH, raw_data_artifact_dir)
    PROCESSED_DATA_ARTIFACT_DIR: str = os.path.join(DATA_INGESTION_ARTIFACT_PATH, processed_data_artifact_dir)
    RAW_DATA_ARTIFACT_FILE: str = os.path.join(RAW_DATA_ARTIFACT_DIR, raw_data_artifact_file)
    TRAIN_DATA_ARTIFACT_FILE: str = os.path.join(PROCESSED_DATA_ARTIFACT_DIR, train_data_artifact_file)
    TEST_DATA_ARTIFACT_FILE: str = os.path.join(PROCESSED_DATA_ARTIFACT_DIR, test_data_artifact_file)
    train_test_split_ratio: float = train_test_split_ratio
    OUTPUT_FEATURE_FOR_MODEL: str = OUTPUT_FEATURE_FOR_MODEL 

@dataclass
class DataValidationConfig:
    DATA_VALIDATION_ARTIFACT_PATH: str = os.path.join(training_pipeline_config.artifact_dir, data_validation_artifact)
    DATA_VALIDATION_REPORT_FILE_PATH: str = os.path.join(DATA_VALIDATION_ARTIFACT_PATH, data_validation_report_file)
    SCHEMA_FILE_PATH: str = os.path.join(schema_folder_name, schema_file_name)
    columns_name_in_schema_file: str = columns_name_in_schema_file
    numerical_columns_key: str = numerical_columns_key
    output_column_key: str = output_column_key

@dataclass
class DataTransformationConfig:
    DATA_TRANSFORMATION_ARTIFACT_DIR: str = os.path.join(training_pipeline_config.artifact_dir, data_transformation_artifact)
    TRANSFORMED_DATA_ARTIFACT_DIR: str = os.path.join(DATA_TRANSFORMATION_ARTIFACT_DIR, transformed_data_dir)
    TRANSFORMED_PIPELINE_ARTIFACT_DIR: str = os.path.join(DATA_TRANSFORMATION_ARTIFACT_DIR, transformed_pipeline_dir)
    TRANSFORMED_TRAIN_DATA_FILE_PATH: str = os.path.join(TRANSFORMED_DATA_ARTIFACT_DIR, transformed_train_data_file.replace("csv", "npy"))
    TRANSFORMED_TEST_DATA_FILE_PATH: str = os.path.join(TRANSFORMED_DATA_ARTIFACT_DIR, transformed_test_data_file.replace("csv", "npy"))
    PIPELINE_FILE_PATH: str = os.path.join(TRANSFORMED_PIPELINE_ARTIFACT_DIR, pipeline_transformation_file)
    SCHEMA_FILE_PATH: str = os.path.join(schema_folder_name, schema_file_name)

@dataclass
class ModelTrainerConfig:
    model_saving_dir: str = os.path.join(training_pipeline_config.artifact_dir, trained_model_dir)
    model_file_path: str = os.path.join(model_saving_dir, model_saving_name)
    fp_cost: int = FP_COST
    fn_cost: int = FN_COST
    model_learning_rate: float = model_learning_rate
    model_max_depth: int = model_max_depth
    model_n_estimators: int = model_n_estimators

@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME
