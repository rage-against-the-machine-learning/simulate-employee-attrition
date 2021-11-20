"""
This is a baseline model
No regularization applied
Default parameters used
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import config
import helper


class LogisticRegress:

    def __init__(self, X_data, y_data, logreg_params: dict = None, verbose: bool = False):
        """
        :X_data: predictors
        :y_data: response variable
        :logreg_params: optional argumen for any parameters specified for Logistic Regression
            see: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
            for parameters & parameter arguments
        """
        self.X = X_data
        self.y = y_data
        self.logreg_parmas = logreg_params
        self.verbose = verbose

    @property
    def scale_data(self):
        ss = StandardScaler()
        self.X_ss = ss.fit_transform(self.X)
        if self.verbose: print("Standard Scaler applied to X data...")
        
    def regress(self):
        if self.logreg_parmas is None:
            self.logreg_params = dict(
                penalty='none',
                random_state=config.SEED,
                verbose=1 if self.verbose else 0,
                n_jobs=-1
            )
        if self.verbose: print(f"\nLogistic Regression parameters: {self.logreg_params}...")

        self.scale_data
        self.lr = LogisticRegression(**self.logreg_params)
        self.lr.fit(self.X_ss, self.y)
        if self.verbose: print("\nLogistic Regression fitted on scaled X, and y...")

    def predict(self, X):
        if self.verbose: print("\nGenerating predictions...")
        self.regress()
        y_pred = self.lr.predict(X)
        y_proba = self.lr.predict_proba(X)

        if self.verbose: print(f"\ny_pred shape: {y_pred.shape}, y_proba shape: {y_proba.shape}")
        return y_pred, y_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_filepath", help="relative filepath from `logreg.py` to raw data .csv file")
    parser.add_argument("--verbose", help="True to print, False to suppress printing.")
    args = parser.parse_args()
    
    raw_data_filepath = args.raw_data_filepath
    verbose = args.verbose

    print(raw_data_filepath)
    print(verbose, "\n")

    ibm = helper.IBMData(raw_data_filepath, verbose=verbose)
    X_train, X_test, y_train, y_test = ibm.train_test_split()
    
    rgr = LogisticRegress(X_train, y_train, verbose=verbose)
    y_train_pred, y_train_proba = rgr.predict(X_train)
    y_test_pred, y_test_proba = rgr.predict(X_test)

    if verbose:
        print(f"Mean accuracy for X train, y train pair: {rgr.lr.score(X_train, y_train)}...")
        print(f"Mean accuracy for X test, y test pair: {rgr.lr.score(X_test, y_test)}...")


if __name__ == '__main__':
    main()
