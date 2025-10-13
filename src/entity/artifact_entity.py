from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path: str
    validation_result: bool
    message: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file: str
    transformed_test_file: bool
    pipeline_transformation_file: str

@dataclass
class ClassificationMetricArtifact:
    total_cost:int
    f1_score:float
    precision_score:float
    recall_score:float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str