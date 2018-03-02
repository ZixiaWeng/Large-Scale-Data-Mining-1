import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import tree

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
        self.data = pd.read_csv('network_backup_dataset.csv').drop('Backup Time (hour)', 1)
        self.scaler_data = scaler_encoding(self.data)
        # self.Y = self.data["Size of Backup (GB)"]
        # self.X = self.data.drop("Size of Backup (GB)", 1)
        # print self.data
        # print self.X, self.Y

    def linear_regression(self):

        kf = KFold(n_splits=10)
        newData = scaler_encoding(self.data)
        # print newData
        for train_index, test_index in kf.split(newData):
            trainset = newData.as_matrix()[train_index]
            testset = newData.as_matrix()[test_index]
            
            trainset = standard_scale(trainset)
            testset = standard_scale(testset)
            print trainset,'before'
            trainY = trainset[:,5]
            print trainY, 'after'
            trainX = np.delete(trainset, 5, 1)

            testY = testset[:,5]
            testX = np.delete(testset, 5, 1)

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
            plt.title("linear reg with standard scale for Fitted values vs. Actual values ")
            plt.show()


            plt.scatter(test_pred, test_pred-testY, color=colors)
            plt.title("linear reg with standard scale for Residuals vs. Fitted value ")
            plt.show()
            #TO DO, create a plot function to facilitate coding efficiency

    def f_reg(self):
        newData = scaler_encoding(self.data)

        y = newData["Size of Backup (GB)"]
        X = newData.drop("Size of Backup (GB)", 1)
        f_vals, p_vals = f_regression(X, y)
        ind = np.array(f_vals).argsort()[-3:][::-1]  # https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        print 'The most important variables:', ind

    # def featureEncoding(self,)

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

        kf = KFold(n_splits=10)
        all_test_mse = []
        all_train_mse = []
        all_oob_error = []
        for train_idx, test_idx in kf.split(self.scaler_data):
            train = self.scaler_data.iloc[train_idx]
            test = self.scaler_data.iloc[test_idx]

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

    def random_forest_with_para(self, para_name, para_range):
        all_oob_error = []
        all_test_mse = []
        for i in para_range:
            args = {
                'tree number': {'tree_num': i},
                'max features': {'max_features': i},
                'max depth': {'max_depth': i}
            }
            arg = args[para_name]
            train_mse, test_mse, oob_error = self.run_random_forest(**arg)
            all_oob_error.append(oob_error)
            all_test_mse.append(test_mse)

        plt.plot(para_range, all_oob_error, label='OOB error')
        plt.xlabel(para_name)
        plt.ylabel('OOB error')
        plt.title('different %s for OOB error' % para_name)
        plt.legend(loc="best")
        plt.show()

        plt.plot(para_range, all_test_mse, label='test mse')
        plt.xlabel(para_name)
        plt.ylabel('test mse')
        plt.title('different %s for test mse' % para_name)
        plt.legend(loc="best")
        plt.show()

    def random_forest(self):
        # q1
        train_mse, test_mse, oob_error = self.run_random_forest()

        print 'average train mse: %.8f' % np.mean(train_mse)
        print 'average test mse: %.8f' % np.mean(test_mse)
        print 'average OOB error: %.5f' % np.mean(oob_error)

        # q2-3
        self.random_forest_with_para('tree number', range(1, 101, 5))
        self.random_forest_with_para('max features', range(1, 6))
        self.random_forest_with_para('max depth', range(1, 10))

        # q4
        best_tree = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            max_features=5,
            bootstrap=True,
            oob_score=True
        )
        train_Y = self.scaler_data["Size of Backup (GB)"]
        train_X = self.scaler_data.drop("Size of Backup (GB)", 1)
        best_tree.fit(train_X, train_Y)
        print 'feature importance:', best_tree.feature_importances_

        # q5
        tree.export_graphviz(best_tree.estimators_[0])
        os.system('dot -Tpng tree.dot -o tree.png')
        os.system('open tree.png')




