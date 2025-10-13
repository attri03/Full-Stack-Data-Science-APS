import sys
from src.exception import MyException
import pandas as pd
import numpy as np
import os
from src.logger import logging

class MyModel:

    def __init__(self, model, preprocessor):
        try:
            self.model = model
            self.preprocessor = preprocessor
        except Exception as e:
            raise MyException(e, sys)
        
    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            return self.model.predict(X_transformed)
        except Exception as e:
            raise MyException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"