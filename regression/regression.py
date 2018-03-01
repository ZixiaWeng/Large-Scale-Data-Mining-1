import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Reader
from surprise import Dataset

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

    def initialDraw(self, duration):
        df = self.data  #initilize data
        dic = dict()    #create a dictionary
        arr = []        #temp arr for recording the Size of Backup value for a single day for specific work-flow-id and file name
        schedule = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for index, row in df.iterrows():
            print row['Work-Flow-ID'], row['File Name']
            flowID, fileName = row['Work-Flow-ID'], row['File Name']
            week, day = row['Week #'], row['Day of Week']
            date = (int(week)-1)*7+schedule.index(day)  # Compute the date

            if date > duration:  # Terminating condition
                return dic
            print 'Current Date:', date, 'Target Date: ', duration
            if flowID not in dic:
                dic[flowID] = dict()
            if fileName not in dic[flowID]:
                dic[flowID][fileName] = [0] * duration

            newDF = df.loc[(df['Work-Flow-ID'] == flowID) & df['File Name'].isin([fileName])]  #only find the rows with this id and file name
            for index_, row_ in newDF.iterrows():
                if row_['Week #'] == week and row_['Day of Week'] == day:  # same date
                    arr.append(row_['Size of Backup (GB)'])
                    dic[flowID][fileName][date] = sum(arr)
            arr = []

        return dic

    def plot_buSize(self, duration):
        dic = self.initialDraw(duration)
        for flowID in dic:
            value = dic[flowID]
            for fileName in value:
                plt.plot(range(duration), value[fileName][: duration], label=str(fileName))
                plt.title(flowID + ', duration = ' + str(duration))
                plt.xlabel('Days')
                plt.ylabel('Size of Backup (GB)')
            plt.legend(loc=1, shadow=True, fancybox=True, prop={'size': 10})
            plt.show()

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


