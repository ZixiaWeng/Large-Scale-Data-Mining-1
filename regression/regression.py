import csv
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold


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

    def scaler_encoding(self, X):
        newX = X.copy()

        newX["Day of Week"] = newX["Day of Week"].apply(day_of_week_encoding)
        newX["Work-Flow-ID"] = newX["Work-Flow-ID"].apply(lambda id: id.split('_')[2])
        newX["File Name"] = newX["File Name"].apply(lambda name: name.split('_')[1])

        return newX

    def random_forest(self):
        # X, y = make_regression(n_features=5, n_informative=2, random_state=0, shuffle=False)
        tree = RandomForestRegressor(
            n_estimators=20,
            max_depth=4,
            bootstrap=True,
            max_features=5,
        )
        kf = KFold(n_splits=2)
        for train, test in kf.split(self.data):
            print train.shape, test.shape
            # train = self.scaler_encoding(train)
            train_Y = train["Size of Backup (GB)"]
            train_X = train.drop("Size of Backup (GB)", 1)

            # test = self.scaler_encoding(test)
            test_Y = test["Size of Backup (GB)"]
            test_X = test.drop("Size of Backup (GB)", 1)

            tree.fit(train_X, train_Y)
            print tree.predict(test_X)

        # print regr.feature_importances_
        # print tree.predict(X)

