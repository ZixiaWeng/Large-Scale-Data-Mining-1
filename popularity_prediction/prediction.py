#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import pprint as pp
import datetime
import pytz
from sklearn import linear_model
from sklearn.metrics import r2_score

def to_date(timestamp):
    pst_tz = pytz.timezone('US/Pacific')
    return datetime.datetime.fromtimestamp(timestamp, pst_tz)


def get_hours(data):
    return data.total_seconds() / 3600.0


def get_hour_diff(all_tweets):
    initDate = to_date(all_tweets[0]['firstpost_date']).replace(minute=0, second=0)
    endDate = to_date(all_tweets[-1]['firstpost_date'])
    return get_hours(endDate - initDate)


def get_location(location):
    keywords_0 = {'Washington', 'WA'}
    keywords_1 = {'Massachusetts', 'MA'}
        
    for word in keywords_0:
        if word in location:
            return 0

    for word in keywords_1:
        if word in location:
            return 1

    return -1


class Prediction:
    def __init__(self):
        self.all_data = {}
        self.train_data_superbowl = self.read_tweet('superbowl')
        self.train_data_nfl = self.read_tweet('nfl')

    def read_tweet(self, hashtag, max_line=1000):
        if hashtag not in {'gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl'}:
            raise Exception('no such data!')

        if hashtag in self.all_data.keys():
            return self.all_data[hashtag]

        file = './tweet_data/tweets_#%s.txt' % hashtag
        tweets = []
        counter = 0
        for line in open(file, 'r'):
            data = json.loads(line)
            tweets.append(data)
            counter += 1
            if (max_line > 0 and counter > max_line):
                break

        self.all_data[hashtag] = tweets
        return tweets

    def plot_histogram(self, hashtag):
        all_tweets = self.read_tweet(hashtag)
        initDate = to_date(all_tweets[0]['firstpost_date']).replace(minute=0, second=0)
        hour_diff = get_hour_diff(all_tweets)
        X = range(0, int(hour_diff) + 1)
        Y = [0] * (int(hour_diff) + 1)
        for tweet in all_tweets:
            index = int(get_hours(to_date(tweet['firstpost_date']) - initDate))
            Y[index] += 1
        plt.bar(X, Y, width=1)
        plt.title('%s number of tweets per hour' % hashtag)
        plt.show()

    def q1(self):
        all_tweets = self.read_tweet('gohawks')
        tweetsLen = len(all_tweets)
        hour_diff = get_hour_diff(all_tweets)
        print 'average tweets per hour: %.3f' % (tweetsLen / hour_diff)

        all_follower = 0.0
        all_retweets = 0.0
        for dta in all_tweets:
            all_follower += int(dta['author']['followers'])
            all_retweets += int(dta['metrics']['citations']['total'])

        print 'average followers of users: %.3f' % (all_follower / tweetsLen)
        print 'average number of retweets: %.3f' % (all_retweets / tweetsLen)

        self.plot_histogram('superbowl')
        self.plot_histogram('nfl')

    def map_hour(self, data):
        initTime = data['firstpost_date'][0]
        data['firstpost_date'] = data['firstpost_date'].apply (lambda x : get_hours(to_date(x) - to_date(initTime)))


    def linear_regression(self):
        df_superbowl = pd.DataFrame(self.train_data_superbowl)
        df_new = pd.DataFrame(columns=['tweets_num','retweets_num','followers_num','followers_num_max','time_of_day'])
        initDate = to_date(self.train_data_superbowl[0]['firstpost_date']).replace(minute=0, second=0)
        hour_diff = get_hour_diff(self.train_data_superbowl)
        total_num_of_tweets = [0] * (int(hour_diff) + 1)
        total_num_of_retweets = [0] * (int(hour_diff) + 1)
        total_num_of_follower = [0] * (int(hour_diff) + 1)
        max_num_follower = [0] * (int(hour_diff) + 1)
        time_of_day = [0] * (int(hour_diff) + 1)
        for dta in self.train_data_superbowl:
            # new_entry = [1, 2, 3, 4, 5]
            # df_new.loc[len(df_new)] = new_entry
            # pass
            index = int(get_hours(to_date(dta['firstpost_date']) - initDate))
            total_num_of_tweets[index] += 1
            total_num_of_retweets[index] += dta['metrics']['citations']['total']
            total_num_of_follower[index] += dta['author']['followers']
            max_num_follower[index] = max(max_num_follower[index], dta['author']['followers'])
            # time_of_day[index] = to_date(dta['firstpost_date']).hour
        time_of_day[0] = initDate.hour
        for i in range(1, len(time_of_day)):
            time_of_day[i] = (time_of_day[i-1] + 1) % 24

        data_dict = {
            'tweets_num': total_num_of_tweets,
            'retweets_num': total_num_of_retweets,
            'followers_num': total_num_of_follower,
            'followers_num_max': max_num_follower,
            'time_of_day': time_of_day
        }

        data = pd.DataFrame(data_dict)
        target = list(data['tweets_num'])
        target.insert(0, 0)
        target = target[:-1]
        reg = linear_model.LinearRegression()
        reg.fit(data.as_matrix(), target)
        predict = reg.predict(data.as_matrix())
        train_errors = reg.score(data.as_matrix(), target)
        r2_sco = r2_score(target, predict)
        print 'Training Accuracy: ', train_errors, 'R Squared Score', r2_sco
        # print total_num_of_tweets
        # print total_num_of_follower
        # print time_of_day
        # print to_date(self.train_data_superbowl[0]['firstpost_date'])
        # print df_new, 'dsad'
        self.map_hour(df_superbowl)

    def part2(self):
        all_tweet = []
        labels = []
        for tweet in self.train_data_superbowl:
            location = get_location(tweet['tweet']['user']['location'])
            content = tweet['tweet']['text']
            if location in {0, 1}:
                all_tweet.append(content)
                labels.append(location)

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(all_tweet)

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        # print X_train_tfidf.shape

        all_models = {
            'NB': MultinomialNB(),
            'SVM': LinearSVC(),
            # 'ovo': OneVsOneClassifier(svc),
            # 'ovr': OneVsRestClassifier(svc),
            'SGD': SGDClassifier()
        }
        for name, model in all_models.items():
            self.location_predcition(X_train_tfidf, labels, model, name)

    def location_predcition(self, data, labels, model, name):
        model.fit(data, labels)
        prediction = model.predict(data)
        fpr, tpr, thresholds = roc_curve(prediction, labels)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', linewidth=2.0, label='ROC (Area is %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='yellow', linewidth=2.0)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve with %s' % name)
        plt.legend(loc='best')
        plt.show()






