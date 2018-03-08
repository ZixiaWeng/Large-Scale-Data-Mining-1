import numpy as np
import json


class Prediction:
    def __init__(self):
        # self.train_data_gohawks = np.loadtxt('./tweet_data/tweets_#gopatriots.txt', dtype=str)
        with open('./tweet_data/tweets_#nfl.txt') as json_file: 
            data = json.load(json_file)
            print data