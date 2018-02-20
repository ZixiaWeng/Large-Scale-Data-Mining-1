import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pprint as pp
import operator


def plot_freqency(d, msg=None):
    sorted_ = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    ids, ratings = zip(*sorted_)
    plot_bar(ratings, msg)


def plot_bar(arr, msg=None):
    plt.bar(np.arange(len(arr)), arr)
    plt.title(msg)
    plt.show()


class Recommand:
    def __init__(self):
        self.read_data()

    def read_data(self):
        max_user_id = -1
        max_movie_id = -1

        filename = 'recommand/ml-latest-small/ratings.csv'
        with open(filename, "rt") as input:
            reader = csv.reader(input, delimiter=',', quoting=csv.QUOTE_NONE)
            next(reader, None)  # skip header
            for line in reader:
                user = int(line[0])
                movie = int(line[1])
                rating = float(line[2])

                if user > max_user_id:
                    max_user_id = user

                if movie > max_movie_id:
                    max_movie_id = movie

        self.max_movie_id = max_movie_id
        self.max_user_id = max_user_id

        max_user_id += 1
        max_movie_id += 1
        rating_matrix = np.array([[None] * max_movie_id] * max_user_id)
        print rating_matrix.shape
        with open(filename, "rt") as input:
            reader = csv.reader(input, delimiter=',', quoting=csv.QUOTE_NONE)
            next(reader, None)  # skip header
            for line in reader:
                user = int(line[0])
                movie = int(line[1])
                rating = float(line[2])

                rating_matrix[user][movie] = rating

        self.ratings = rating_matrix

    def preprocessing(self):  # q1-6
        users, movies = set(), set()
        user_rating_num, movie_rating_num = defaultdict(int), defaultdict(int)
        movie_rating = defaultdict(list)

        for u in range(self.max_user_id + 1):
            for m in range(self.max_movie_id + 1):
                rating = self.ratings[u][m]
                if rating:
                    users.add(u)
                    movies.add(m)
                    user_rating_num[u] += 1
                    movie_rating_num[m] += 1
                    movie_rating[m].append(rating)

        # print len(users), len(movies)
        combinations = len(users) * len(movies)
        rating_num = 100004.0
        print "Sparisty = " + str(rating_num / combinations)    # Q1

        # Q3 Q4
        # plot_freqency(movie_rating_num, 'Movie Rating Frequency')
        # plot_freqency(user_rating_num, 'User Rating Frequency')

        # Q6
        movie_var = defaultdict(int)
        for ratings in movie_rating.values():
            movie_var[int(np.var(ratings) / 0.5)] += 1

        max_var = max(movie_var.keys())
        movie_var_list = [0] * (max_var + 1)
        for var, count in movie_var.items():
            movie_var_list[var] = count

        X = np.arange(len(movie_var_list))
        plt.bar(X, movie_var_list)
        plt.xticks(X, map(str, (np.arange(0, len(movie_var_list) / 2, 0.5))))
        plt.title('Variance of Movies')
        plt.show()




        # Q3
        # movie_count = {}
        # for i in range (len(self.ratings['rating'])):
        #   key = int (self.ratings['movie'][i])
        #   if not key in movie_count.keys():
        #     movie_count[ self.ratings['movie'][i] ] = 0
        #   else:
        #     movie_count[ self.ratings['movie'][i] ] += 1

        # print movie_count.values()


    def knn(self):  # q7-11
        pass




