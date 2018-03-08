#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
import pprint as pp


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
        pp.pprint(self.data)
