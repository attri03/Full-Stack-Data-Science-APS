############## Pipeline ##############################
PIPELINE_NAME: str = ""
######################################################

############## Artifact ##############################
artifact_dir: str = "artifact"
######################################################

############## Data Ingestion Constants ##############
raw_data_dir: str = "notebook"
raw_data_file: str = "aps_data.csv"
data_ingestion_artifact: str = "data_ingestion"
raw_data_artifact_dir: str = "raw_data"
processed_data_artifact_dir: str = "processed_data"
raw_data_artifact_file: str = "raw_data.csv"
train_data_artifact_file: str = "train_data.csv"
test_data_artifact_file: str = "test_data.csv"
train_test_split_ratio:float = 0.2
OUTPUT_FEATURE_FOR_MODEL: str = "class"
######################################################

############## Data Validation Constants ##############
data_validation_artifact: str = "data_validation"
data_validation_report_file: str = "report.yaml"
schema_folder_name: str = "config"
schema_file_name: str = "schema.yaml"
columns_name_in_schema_file: str = "columns"
numerical_columns_key: str = "numerical_columns"
output_column_key: str = "output_column"
######################################################

############## Data Transformation Constants ##############
data_transformation_artifact: str = "data_transformation"
transformed_data_dir: str = "transformed_data"
transformed_pipeline_dir: str = "pipeline"
transformed_train_data_file: str = "train.csv"
transformed_test_data_file: str = "test.csv"
pipeline_transformation_file: str = "preprocessing.pkl"
######################################################

############## Model Trainer Constants ##############
trained_model_dir: str = "trained_model"
model_saving_name: str = "model.pkl"
FP_COST: int = 10
FN_COST: int = 500
model_learning_rate: float = 0.05
model_max_depth: int = 3
model_n_estimators: int = 300
#####################################################

############## Model Evaluation Constants ##############
AWS_ACCESS_KEY_ID_ENV_KEY: str = "AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY_ENV_KEY: str = "AWS_SECRET_KEY"
REGION_NAME: str = "AWS_REGION_NAME"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME: str = "personal-proj-aps-sensor-project"
MODEL_PUSHER_S3_KEY: str = "model-registry"
MODEL_FILE_NAME: str = "model.pkl"
#####################################################