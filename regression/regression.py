import csv
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


def catagory_to_int(data):



class Regression:
    def __init__(self):
        self.data = pd.read_csv('network_backup_dataset.csv').drop('Backup Start Time - Hour of Day', 1)
        self.Y = self.data["Size of Backup (GB)"]
        self.X = self.data.drop("Size of Backup (GB)", 1)
        # print self.data
        print self.X, self.Y

    def random_forest(self):
        # X, y = make_regression(n_features=5, n_informative=2, random_state=0, shuffle=False)
        regr = RandomForestRegressor(
            n_estimators=20,
            max_depth=4,
            bootstrap=True,
            max_features=5,
        )
        regr.fit(self.X, self.Y)
        # print regr.feature_importances_
        # print regr.predict([[0, 0, 0, 0, 0]])

