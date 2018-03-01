import csv
import numpy as np
from surprise import Dataset
from surprise import Reader
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


class Regression:
    def __init__(self):
        self.data = pd.read_csv('network_backup_dataset.csv')
        # print self.data['Week #']

    def random_forest(self):
        X, y = make_regression(n_features=5, n_informative=2, random_state=0, shuffle=False)
        regr = RandomForestRegressor(
            n_estimators=20,
            max_depth=4,
            bootstrap=True,
            max_features=5,
        )
        regr.fit(X, y)
        print regr.feature_importances_
        print regr.predict([[0, 0, 0, 0, 0]])