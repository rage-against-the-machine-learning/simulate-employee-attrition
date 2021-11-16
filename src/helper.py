from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

import config

class IBMData:
    def __init__(self, ibm_data_filepath: str = config.data_filepath, verbose: bool = False):
        self.data_path = ibm_data_filepath 
        self.verbose = verbose
        self.raw = self.load_ibm_data(self.data_path)

    def load_ibm_data (self, ibm_data_filepath: str) -> pd.DataFrame:
        return pd.read_csv(ibm_data_filepath)
        
    def separate_response(self, response_var: str = 'Attrition') -> Tuple(pd.Series):
        """
        :returns: X, and y
            X: all the predictor variables (features)
            y: the response variable
        """
        X = self.load_ibm_data(self.data_path).drop(response_var, axis=1)
        y = self.load_ibm_data(self.data_path)[response_var]
        return X, y

    def train_test_split(self, test_size: float = 0.25):
        """
        
        """