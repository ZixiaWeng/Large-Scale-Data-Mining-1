#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
import pprint as pp
import datetime, time
import pytz

def read_tweet(hashtag, max_line=100):
    if hashtag not in {'gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl'}:
        raise Exception('no such data!')

    file = './tweet_data/tweets_#%s.txt' % hashtag
    tweets = []
    counter = 0
    for line in open(file, 'r'):
        data = json.loads(line)
        tweets.append(data)
        counter += 1
        if counter > max_line:
            break
    return tweets


class Prediction:
    def __init__(self):
        self.data = read_tweet('gohawks')
        # date = self.get_date()
        self.getAvrgFollower()
        self.getAvrgRetweets()
        # pp.pprint(self.data)

    def get_date(self):
        date = []
        initDate = self.data[0]['firstpost_date']
        for dta in self.data:
            temp = dta['firstpost_date']
            # print temp
            pst_tz = pytz.timezone('US/Pacific') 
            datetime.datetime.fromtimestamp(temp, pst_tz)
            print datetime,'datetime'
            date.append(datetime)

    def getAvrgFollower(self):
        tweetsLen = len(self.data)
        temp = 0
        for dta in self.data:
            temp += int(dta['author']['followers'])
        print temp/tweetsLen,'average followers of users'

    def getAvrgRetweets(self):
        tweetsLen = len(self.data)
        temp = 0
        for dta in self.data:
            temp += int(dta['metrics']['citations']['total'])
        print temp/tweetsLen,'average number of retweets'

