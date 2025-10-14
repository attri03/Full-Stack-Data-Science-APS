from src.pipline.training_pipeline import TrainPipeline
# from src.pipline.prediction_pipeline import APSSensorDataFrame, APSSensorPredictor
# import pandas as pd

# Training Pipeline Demo
if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()

#Prediction Pipeline Demo (Run Training pipeline atleast once before predictions.)
# if __name__ == "__main__":
#     dataframe = pd.read_csv("artifact/10_14_2025_16_52_42/data_ingestion/processed_data/test_data.csv")
#     dictionary = dataframe.drop("class", axis = 1).iloc[[0, 1, 2]].to_dict(orient='list')
#     obj = APSSensorDataFrame(dictionary = dictionary)
#     input_df = obj.final_input_data()
#     print(input_df)
#     obj2 = APSSensorPredictor()
#     prediction = obj2.predict(input_df)
#     print(prediction)

