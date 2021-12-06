import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

import config
        

class PreProcess:
    
    def __init__(self, X_data: pd.DataFrame, verbose: bool = False):
        self.X = X_data
        self.pp_X = X_data.copy()
        self.verbose = verbose
        
    @property
    def label_encode(self):
        categorical_ojbect = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
        categorical_numeric = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
                              'JobSatisfaction', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']
        categorical_columns = categorical_ojbect + categorical_numeric
        assert all(col in self.X.columns for col in categorical_columns)

        encoders = {}
        for c in categorical_columns:
            encoders[c] = LabelEncoder()
            encoders[c].fit(self.pp_X[c])
            self.pp_X[c] = encoders[c].transform(self.pp_X[c])
            
        # need this in order to inverse transform back to the categorical labels
        self.label_encoders = encoders
        
        if self.verbose:
            print(f'\nThe following categorical_columns have been label encoded: {categorical_columns}')
            
    @property
    def binary_encode(self):
        binarizable_columns = ['Gender', 'Over18', 'OverTime']
        assert all(col in self.pp_X.columns for col in binarizable_columns)
        
        encoders = {}
        for b in binarizable_columns:
            encoders[b] = LabelBinarizer()
            encoders[b].fit(self.pp_X[b])
            self.pp_X[b] = encoders[b].transform(self.pp_X[b])
            
        # need this in order to inverse transform back to the binary category labels
        self.binary_encoders = encoders
        
        if self.verbose:
            print(f'\nThe following categorical_columns have been binarized: {binarizable_columns}')
        
    def preprocess_data(self):
        self.label_encode
        self.binary_encode


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
        pp = PreProcess(self.X, self.verbose)
        pp.preprocess_data()
        self.pp_X = pp.pp_X

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.pp_X, self.y, random_state=config.SEED)

        if self.verbose:
            print("\nTrain Test Split...")
            print("X_train.shape", self.X_train.shape)
            print("X_test.shape", self.X_test.shape)
            print("y_train.shape", self.y_train.shape)
            print("y_test.shape", self.y_test.shape)

        return self.X_train, self.X_test, self.y_train, self.y_test


class CoMultiLinearity:

    def __init__(self, all_data: pd.DataFrame, verbose: bool = False):
        """Determine if there is colinearity or multilinearity in the data
        :all_data: dataframe that has the response variable as well
            data with all features represented numerically is best
        """
        self.data = all_data
        self.verbose = verbose
        self.corr = self.calc_correlation(plot=self.verbose)
        self.VIF = self.calc_VIF

    @property
    def calc_correlation (self, plot: bool = False):
        """Calculate correlation dataframe
        :plot: if Truue, plot will be generated
        """
        if plot:
            corr = self.data.corr()
            fig = plt.figure(1, figsize=(20, 20))
            mask = np.triu(np.ones_like(corr, dtype=bool))

            sns.heatmap(corr.round(2),
                        cbar=True,
                        mask=mask,
                        annot=True,
                        center=0,
                        square=True, 
                        linewidths=.5, 
                        cbar_kws={"shrink": .5},
                        cmap=sns.diverging_palette(230, 20, as_cmap=True))
            plt.xticks(rotation=75)
            
            if self.verbose:
                print('Saving Correlation heatmat at: "../reports/figures/corr-heatmap-annot.png"')
            fig.savefig('../reports/figures/corr-heatmap-annot.png')

        self.corr = self.data.corr()

    @property
    def calc_VIF(self):
        """Caclulate VIF on all features
        """
        def calculate_vif(df):    
            vif, tolerance = {}, {}
            features = df.columns
            
            for feature in features:
                X = [f for f in features if f != feature]        
                X, y = df[X], df[feature]
                r2 = LinearRegression().fit(X, y).score(X, y)                                
                tolerance[feature] = 1 - r2
                vif[feature] = 1/(tolerance[feature])
            return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})

        vif_df = calculate_vif(self.data)

        if self.verbose:
            print(vif_df)
        self.VIF = vif_df

    def find_collinear_features (self, correlation_threshold: float):
        """Find the collinear features
        :correlation_threshold: a threshold between -1 and 1
            for any feature-pair w/ correlation greater than this threshold, keep first alphabetical
        """
        collinear_pairs = list()

        for i, row in self.corr.iterrows():
            corr_cols = (np.where(row > 0.6))[0]
            pair = set()
            for j in corr_cols:
                pair = set([i, self.corr.columns[j]])
                if self.corr.loc[i, self.corr.columns[j]] != 1 and pair not in collinear_pairs: 
                    collinear_pairs.append(pair) 

        if self.verbose:
            print(collinear_pairs)
        return collinear_pairs

    def find_multilinear_features (self, vif_threshold: float):
        """Find multilinear features
        :vif_threshold: float
        https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f
        """
        if vif_threshold > 1: 
            raise ValueError("vif_threshold must be greater than 1")
        multilin_feats = self.VIF[self.VIF['VIF'] > vif_threshold]
        return multilin_feats.index.tolist()


if __name__ == '__main__':
    None
