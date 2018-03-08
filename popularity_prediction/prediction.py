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


def to_date(timestamp):
    pst_tz = pytz.timezone('US/Pacific')
    return datetime.datetime.fromtimestamp(timestamp, pst_tz)


def get_hours(data):
    return data.total_seconds() / 3600.0


class Prediction:
    def __init__(self):
        pass

    def q1(self):
        all_tweets = read_tweet('gohawks')
        initDate = to_date(all_tweets[0]['firstpost_date'])
        endDate = to_date(all_tweets[-1]['firstpost_date'])
        hour_diff = float(get_hours(endDate - initDate))
        print 'average tweets per hour: %.3f' % (len(all_tweets) / hour_diff)

        



