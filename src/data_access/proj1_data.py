from src.constants import *
from src.exception import MyException
import sys
import os
import pandas as pd

class Proj1Data:

    def __init__(self):
        try:
            self.raw_data_dir = raw_data_dir
            self.raw_data_file = raw_data_file
        except Exception as e:
            raise MyException(e,sys)
        
    def get_data(self):
        try:
            data_path = os.path.join(self.raw_data_dir, self.raw_data_file)
            df = pd.read_csv(data_path)
            return df
        except Exception as e:
            raise MyException(e,sys)
