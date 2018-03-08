#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import pprint as pp
import datetime, time
import pytz


def read_tweet(hashtag, max_line=1000):
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


def to_date(timestamp):
    pst_tz = pytz.timezone('US/Pacific')
    return datetime.datetime.fromtimestamp(timestamp, pst_tz)


def get_hours(data):
    return data.total_seconds() / 3600.0


class Prediction:
    def __init__(self):
        self.train_data_superbowl = read_tweet('superbowl')
        self.train_data_nfl = read_tweet('nfl')

    def q1(self):
        all_tweets = read_tweet('gohawks')
        tweetsLen = len(all_tweets)

        initDate = to_date(all_tweets[0]['firstpost_date'])
        endDate = to_date(all_tweets[-1]['firstpost_date'])
        hour_diff = float(get_hours(endDate - initDate))
        print 'average tweets per hour: %.3f' % (tweetsLen / hour_diff)

        all_follower = 0.0
        all_retweets = 0.0
        for dta in all_tweets:
            all_follower += int(dta['author']['followers'])
            all_retweets += int(dta['metrics']['citations']['total'])

        print 'average followers of users: %.3f' % (all_follower / tweetsLen)
        print 'average number of retweets: %.3f' % (all_retweets / tweetsLen)

    def map_hour(self, data):
        initTime = data['firstpost_date'][0]
        data['firstpost_date'] = data['firstpost_date'].apply (lambda x : get_hours(x - initTime))

    def linear_regression(self):
        df_superbowl = pd.DataFrame(self.train_data_superbowl)
        self.map_hour(df_superbowl)
