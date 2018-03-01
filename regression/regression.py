import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Reader
from surprise import Dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def scaler_encoding(data):
    new_data = data.copy()

    new_data["Day of Week"] = new_data["Day of Week"].apply(day_of_week_encoding)
    new_data["Work-Flow-ID"] = new_data["Work-Flow-ID"].apply(lambda id: id.split('_')[2])
    new_data["File Name"] = new_data["File Name"].apply(lambda name: name.split('_')[1])

    return new_data

def standard_scale(data):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(data)
    return scaler.transform(data)
class Regression:
    def __init__(self):
        self.data = pd.read_csv('network_backup_dataset.csv').drop('Backup Start Time - Hour of Day', 1)
        # self.Y = self.data["Size of Backup (GB)"]
        # self.X = self.data.drop("Size of Backup (GB)", 1)
        # print self.data
        # print self.X, self.Y


    def linear_regression(self):

        kf = KFold(n_splits=10)
        # print self.data[:3]
        # result = next(kf.split(self.data), None)
        # print result
        # trainset = self.data.as_matrix()[result[0]]
        # print trainset[:3]

        newData = scaler_encoding(self.data)
        # print newData
        for train_index, test_index in kf.split(newData):
            trainset = newData.as_matrix()[train_index]
            testset = newData.as_matrix()[test_index]
            
            trainset = standard_scale(trainset)
            testset = standard_scale(testset)

            trainY = trainset[:,4]
            trainX = np.delete(trainset, 4, 1)

            testY = testset[:,4]
            testX = np.delete(testset, 4, 1)

            lr = LinearRegression()
            lr.fit(trainX, trainY)

            train_pred = lr.predict(trainX)
            train_rmse = rmse(train_pred, trainY)
            print "Training RMSE is: " + str(train_rmse)

            test_pred = lr.predict(testX)
            test_rmse = rmse(test_pred, testY)
            print "Test RMSE is: " + str(test_rmse)

            # Plot fitted results vs true values
            colors = ['blue','yellow']
            plt.scatter(testY, test_pred, color=colors)
            plt.show()

    def f_reg(self):
        newData = scaler_encoding(self.data)

        y = newData["Size of Backup (GB)"]
        X = newData.drop("Size of Backup (GB)", 1)
        f_vals, p_vals = f_regression(X, y)
        ind = np.array(f_vals).argsort()[-3:][::-1] #https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        print ind


    def initialDraw(self, duration):
        df = self.data  #initilize data
        dic = dict()    #create a dictionary
        arr = []        #temp arr for recording the Size of Backup value for a single day for specific work-flow-id and file name
        schedule = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for index, row in df.iterrows():
            # print row['Work-Flow-ID'], row['File Name']
            flowID, fileName = row['Work-Flow-ID'], row['File Name']
            week, day = row['Week #'], row['Day of Week']
            date = (int(week)-1)*7+schedule.index(day)  # Compute the date

            if date >= duration:  # Terminating condition
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


    def random_forest(self):
        # X, y = make_regression(n_features=5, n_informative=2, random_state=0, shuffle=False)
        tree = RandomForestRegressor(
            n_estimators=20,
            max_depth=4,
            bootstrap=True,
            max_features=5,
        )

        scaler_data = scaler_encoding(self.data)
        kf = KFold(n_splits=10)
        for train_idx, test_idx in kf.split(scaler_data):
            train = scaler_data.iloc[train_idx]
            test = scaler_data.iloc[test_idx]

            train_Y = train["Size of Backup (GB)"]
            train_X = train.drop("Size of Backup (GB)", 1)

            test_Y = test["Size of Backup (GB)"]
            test_X = test.drop("Size of Backup (GB)", 1)

            tree.fit(train_X, train_Y)
            print tree.predict(test_X)

        # print regr.feature_importances_
        # print tree.predict(X)


