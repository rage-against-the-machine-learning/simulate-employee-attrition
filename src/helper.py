import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import config


class PreProcess:

    def __init__(self, X_data: pd.DataFrame, y_data: pd.DataFrame):
        self.X = X_data
        self.y = y_data
        self.pp_X = X_data.copy()

    def label_encode(self):
        categorical_ojbect = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
        categorial_numeric = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
                              'JobSatisfaction', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']
        categorical_columns = categorical_ojbect + categorial_numeric
        assert all(col in self.data.columns for col in categorical_columns)

        encoders = {}
        for c in categorical_columns:
            encoders[c] = LabelEncoder()
            encoders[c].fit(self.X[c])
            self.pp_X[c] = encoders[c].transform(self.X[c])
            
        # need this in order to inverse transform back to the categorical labels
        self.label_encoders = encoders
            
    def binary_encode(self):
        binarizable_columns = ['Gender', 'Over18', 'OverTime']
        assert all(col in self.data.columns for col in binarizable_columns)
        
        encoders = {}
        for b in binarizable_columns:
            encoders[b] = LabelBinarizer()
            encoders[b].fit(self.X[b])
            self.pp_X[b] = encoders[b].transform(self.X[b])
            
        # need this in order to inverse transform back to the binary category labels
        self.binary_encoders = encoders
        
class IBMData:
    def __init__(self, ibm_data_filepath: str = config.DATA_PATH, verbose: bool = False):
        self.data_path = ibm_data_filepath 
        self.verbose = verbose
        self.raw = pd.read_csv(self.data_path)

    @property
    def separate_response(self, response_var: str = 'Attrition'):
        """
        :returns: X, and y
            X: all the predictor variables (features)
            y: the response variable
        """
        self.response_variable = response_var

        self.X = self.raw.drop(response_var, axis=1)
        self.y = self.raw[response_var]

        if self.verbose:
            print("\nSplit X and y...")
            print("self.X.shape", self.X.shape)
            print("self.y.shape", self.y.shape)

        return self.X, self.y

    def train_test_split(self, test_size: float = 0.25):
        """
        split data into train/val vs. test(hold out) data subsets 
        :returns: X_train, X_test, y_train, y_test
        """
        self.separate_response
        
        # Preprocess prior to train-test-splitting
        pp = PreProcess(self.X, self.y)
        

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=config.SEED)

        if self.verbose:
            print("\nTrain Test Split...")
            print("X_train.shape", self.X_train.shape)
            print("X_test.shape", self.X_test.shape)
            print("y_train.shape", self.y_train.shape)
            print("y_test.shape", self.y_test.shape)

        return self.X_train, self.X_test, self.y_train, self.y_test

    


