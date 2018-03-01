import csv
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


def day_of_week_encoding(day):
    return {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }[day]


class Regression:
    def __init__(self):
        self.data = pd.read_csv('network_backup_dataset.csv').drop('Backup Start Time - Hour of Day', 1)
        self.Y = self.data["Size of Backup (GB)"]
        self.X = self.data.drop("Size of Backup (GB)", 1)
        # print self.data
        # print self.X, self.Y

    def scaler_encoding(self):
        newX = self.X.copy()
        newY = self.Y.copy()

        newX["Day of Week"] = newX["Day of Week"].apply(day_of_week_encoding)
        newX["Work-Flow-ID"] = newX["Work-Flow-ID"].apply(lambda id: id.split('_')[2])
        newX["File Name"] = newX["File Name"].apply(lambda name: name.split('_')[1])

        return newX, newY

    def random_forest(self):
        # X, y = make_regression(n_features=5, n_informative=2, random_state=0, shuffle=False)
        X, Y = self.scaler_encoding()
        print X, Y
        regr = RandomForestRegressor(
            n_estimators=20,
            max_depth=4,
            bootstrap=True,
            max_features=5,
        )
        regr.fit(X, Y)
        # print regr.feature_importances_
        # print regr.predict([[0, 0, 0, 0, 0]])

