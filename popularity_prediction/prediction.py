#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
import pprint as pp


class Prediction:
    def __init__(self):
        file = './tweet_data/tweets_#nfl.txt'
        tweets = []
        for line in open(file, 'r'):
            data = json.loads(line)
            tweets.append(data)
            pp.pprint(data)
