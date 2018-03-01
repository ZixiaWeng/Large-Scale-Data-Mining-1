import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Reader
from surprise import Dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


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


def scaler_encoding(data):
    new_data = data.copy()

    new_data["Day of Week"] = new_data["Day of Week"].apply(day_of_week_encoding)
    new_data["Work-Flow-ID"] = new_data["Work-Flow-ID"].apply(lambda id: id.split('_')[2])
    new_data["File Name"] = new_data["File Name"].apply(lambda name: name.split('_')[1])

    return new_data


class Regression:
    def __init__(self):
        self.data = pd.read_csv('network_backup_dataset.csv').drop('Backup Start Time - Hour of Day', 1)
        # self.Y = self.data["Size of Backup (GB)"]
        # self.X = self.data.drop("Size of Backup (GB)", 1)
        # print self.data
        # print self.X, self.Y

    def initialDraw(self, duration):
        df = self.data  # initilize data
        dic = dict()    # create a dictionary
        arr = []        # temp arr for recording the Size of Backup value for a single day for specific work-flow-id and file name
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

            newDF = df.loc[(df['Work-Flow-ID'] == flowID) & df['File Name'].isin([fileName])]  # only find the rows with this id and file name
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

    def run_random_forest(self, tree_num=20, max_depth=4, max_features=5):
        # X, y = make_regression(n_features=5, n_informative=2, random_state=0, shuffle=False)
        tree = RandomForestRegressor(
            n_estimators=tree_num,
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=True,
            oob_score=True
        )

        scaler_data = scaler_encoding(self.data)
        kf = KFold(n_splits=10)
        all_test_mse = []
        all_train_mse = []
        all_oob_error = []
        for train_idx, test_idx in kf.split(scaler_data):
            train = scaler_data.iloc[train_idx]
            test = scaler_data.iloc[test_idx]

            train_Y = train["Size of Backup (GB)"]
            train_X = train.drop("Size of Backup (GB)", 1)

            test_Y = test["Size of Backup (GB)"]
            test_X = test.drop("Size of Backup (GB)", 1)

            tree.fit(train_X, train_Y)

            pred_train = tree.predict(train_X)
            mse_train = mean_squared_error(train_Y, pred_train)
            all_train_mse.append(mse_train)

            pred_test = tree.predict(test_X)
            mse_test = mean_squared_error(test_Y, pred_test)
            all_test_mse.append(mse_test)

            OOB_error = 1 - tree.oob_score_
            all_oob_error.append(OOB_error)

        return np.mean(all_train_mse), np.mean(all_test_mse), np.mean(all_oob_error)

    def random_forest(self):
        train_mse, test_mse, oob_error = self.run_random_forest()

        print 'average train mse: %.8f' % np.mean(train_mse)
        print 'average test mse: %.8f' % np.mean(test_mse)
        print 'average OOB error: %.5f' % np.mean(oob_error)



