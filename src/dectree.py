"""
This is a baseline Decision Tree model
No regularization applied
Default parameters used
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
import helper


class DecTree:

    def __init__(self, X_data, y_data, dectree_params: dict = None, verbose: bool = False):
        """
        :X_data: predictors
        :y_data: response variable
        :logreg_params: optional argumen for any parameters specified for Decision Tree Classifier
            see: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
            for parameters & parameter arguments
        """
        self.X = X_data
        self.y = y_data

        if dectree_params is None:
            self.dectree_params = dict(random_state=config.SEED)
        else:
            self.dectree_params = dectree_params
        self.verbose = verbose

    @property
    def scale_data(self):
        ss = StandardScaler()
        self.X_ss = ss.fit_transform(self.X)
        if self.verbose: print("Standard Scaler applied to X data...")

    def classify(self):
        if self.verbose: print(f"\Decision Tree parameters: {self.dectree_params}...")

        self.scale_data
        self.dectree = DecisionTreeClassifier(**self.dectree_params)
        self.dectree.fit(self.X_ss, self.y)
        if self.verbose: print("\nDecision Tree Classifier fitted on scaled X, and y...")

    def predict(self, X):
        if self.verbose: print("\nGenerating predictions...")
        self.classify()
        y_pred = self.dectree.predict(X)
        y_proba = self.dectree.predict_proba(X)

        if self.verbose: print(f"\ny_pred shape: {y_pred.shape}, y_proba shape: {y_proba.shape}")
        return y_pred, y_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_filepath", help="relative filepath from `dectree.py` to raw data .csv file")
    parser.add_argument("--verbose", help="True to print, False to suppress printing.")
    args = parser.parse_args()

    raw_data_filepath = args.raw_data_filepath
    verbose = args.verbose

    print(raw_data_filepath)
    print(verbose, "\n")

    ibm = helper.IBMData(raw_data_filepath, verbose=verbose)
    X_train, X_test, y_train, y_test = ibm.train_test_split()

    dect = DecTree(X_train, y_train, verbose=verbose)
    y_train_pred, y_train_proba = dect.predict(X_train)
    y_test_pred, y_test_proba = dect.predict(X_test)

    if verbose:
        print(f"Mean accuracy for X train, y train pair: {dect.dectree.score(X_train, y_train)}...")
        print(f"Mean accuracy for X test, y test pair: {dect.dectree.score(X_test, y_test)}...")


if __name__ == '__main__':
    main()
    