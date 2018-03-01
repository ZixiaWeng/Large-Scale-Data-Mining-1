import csv
import numpy as np
import pandas as pd

class Regression:
    def __init__(self):
        self.data = pd.read_csv('network_backup_dataset.csv')
        print self.data["Size of Backup (GB)"][:10]

        self.Y = data["Size of Backup (GB)"]
        self.X = del data.drop("Size of Backup (GB)", 1)

    def Scaler_Encoding(self):

    	return data


   	def One_Hot_Encoding(self):

   		return 0

    def Linear_Regression():
    	return 0