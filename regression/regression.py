import csv
import numpy as np
import pandas as pd

class Regression:
    def __init__(self):
        self.data = pd.read_csv('network_backup_dataset.csv')
        #print self.data["Size of Backup (GB)"][:10]

        self.Y = self.data["Size of Backup (GB)"]
        self.X = self.data.drop("Size of Backup (GB)", 1)

    def Scaler_Encoding(self):

    	# print  type (self.X["Week #"][0])
    	# print  type (self.X["Day of Week"][0])
    	# print  type (self.X["Backup Start Time - Hour of Day"][0])
    	# print  type (self.X["Work-Flow-ID"][0])
    	# print  type (self.X["File Name"][0])
    	# print type (self.Y[0])
    	# print  type (self.X["Backup Time (hour)"][0])

    	newX = self.X.copy()
    	newY = self.Y.copy()

    	# Scaler encode day of week
    	for day_of_week in newX["Day of Week"]: 
    		if (day_of_week == "Monday") :
    			day_of_week = 1
    		elif day_of_week == "Tuesday": 
    			day_of_week = 2
    		elif day_of_week == "Wednesday": 
    			day_of_week = 3
    		elif day_of_week == "Thursday": 
    			day_of_week = 4
    		elif day_of_week == "Friday": 
    			day_of_week = 5
    		elif day_of_week == "Saturday": 
    			day_of_week = 6
    		elif day_of_week == "Sunday": 
    			day_of_week = 7

    	# Scaler encode Work-Flow-ID
    	for workID in newX["Work-Flow-ID"]: 
    		workID = int (workID.split('_')[2])

    	# Scaler encode File Name
    	for fname in newX["File Name"]:
    		fname = int (fname.split('_')[1])

    	#return newX, newY

   	def One_Hot_Encoding(self):

   		return 0

    def Linear_Regression():
    	return 0