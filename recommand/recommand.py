import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pprint as pp
import operator

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.prediction_algorithms import knns


def plot_freqency(d, msg=None):
    sorted_ = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    ids, ratings = zip(*sorted_)
    plot_bar(ratings, msg)


def plot_bar(arr, msg=None):
    plt.bar(np.arange(len(arr)), arr)
    plt.title(msg)
    plt.show()


def read_data():
    # Loading Ratings.csv
    ratings = {}
    sparse = {}
    ratings['user'], sparse['user'] = [], []
    ratings['movie'], sparse['movie'] = [], []
    ratings['rating'] = []
    filename = 'recommand/ml-latest-small/ratings.csv'
    with open(filename, "rt") as input:
        reader = csv.reader(input, delimiter=',', quoting=csv.QUOTE_NONE)
        next(reader, None) # skip header
        for line in reader:
            ratings['user'].append(float(line[0]))
            ratings['movie'].append(float(line[1]))
            ratings['rating'].append(float(line[2]))
            if float(line[0]) not in sparse['user']:
                sparse['user'].append(float(line[0]))  # available users
            if float(line[1]) not in sparse['movie']:
                sparse['movie'].append(float(line[1]))  # available movies
    sparisty = len(ratings['rating'])/(float((len(sparse['user'])) * len(sparse['movie'])))
    return ratings, sparisty


class Recommand:
    def __init__(self):
        self.get_data_matrix()

    def get_data_matrix(self):
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

        self.ratings_matrix = rating_matrix

    def preprocessing(self):  # q1-6
        self.ratings, self.sparisty = read_data()
        print 'sparisty: ' + str(self.sparisty)

        # Q2
        plot2_y = np.zeros(11)
        for i in range(len(self.ratings['rating'])):
            plot2_y[int(self.ratings['rating'][i] / 0.5)] += 1
        plt.bar(range(0, 11), plot2_y)
        plt.show()

        users, movies = set(), set()
        user_rating_num, movie_rating_num = defaultdict(int), defaultdict(int)
        movie_rating = defaultdict(list)

        for u in range(self.max_user_id + 1):
            for m in range(self.max_movie_id + 1):
                rating = self.ratings_matrix[u][m]
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

        # Q3, Q4
        plot_freqency(movie_rating_num, 'Movie Rating Frequency')
        plot_freqency(user_rating_num, 'User Rating Frequency')

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
        
        # Q3 Q4?
        # movie_count = {}
        # for i in range (len(self.ratings['rating'])):
        #     key = int (self.ratings['movie'][i])
        #     if not key in movie_count.keys():
        #         movie_count[ self.ratings['movie'][i] ] = 0
        #     else:
        #         movie_count[ self.ratings['movie'][i] ] += 1
        # movie_rats = [(i, movie_count[i]) for i in movie_count]
        # movie_rats = sorted(movie_rats, cmp = lambda x, y: cmp(y[1], x[1]))
        # movie_plot = [i[1] for i in movie_rats]
        # plt.plot(movie_plot)
        # plt.show()

    def knn(self):  # q7-11
        filename = 'recommand/ml-latest-small/ratings.csv'
        reader = Reader()
        # data = Dataset.load_from_file(filename, reader=reader )
        df = pd.DataFrame(self.ratings)
        data = Dataset.load_from_df(df[['user', 'movie', 'rating']], reader)
        sim_options = {
            'name': 'pearson_baseline',
            'shrinkage': 0  # no shrinkage
        }

        # Q10
        rmse_by_k = []
        mae_by_k = []
        k_values = []
        for k in range(1, 51):
            k_values.append(2*k)
            algo = knns.KNNWithMeans(k=k*2, sim_options=sim_options)
            # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
            kf = KFold(n_splits=10)
            rmse_by_fold = []
            mae_by_fold = []
            for trainset, testset in kf.split(data):
                algo.fit(trainset)
                predictions = algo.test(testset)
                
                # Compute Root Mean Squared Error
                rmse_by_fold.append(accuracy.rmse(predictions, verbose=True))
                # Compute MAE
                mae_by_fold.append(accuracy.mae(predictions, verbose=True))
            rmse_by_k.append(np.mean(rmse_by_fold))
            mae_by_k.append(np.mean(mae_by_fold))
              
        plt.plot(k_values, rmse_by_k)
        plt.plot(k_values, mae_by_k)
        plt.legend(['RMSE', 'MAE'])
        plt.show()
        
        # Q11
        # Read plot of Q10

        # Q12
        rmse_by_k = []
        mae_by_k = []
        k_values = []
        for k in range(1, 51):
            k_values.append(2*k)
            algo = knns.KNNWithMeans(k=k*2, sim_options=sim_options)
            # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
            kf = KFold(n_splits=10)
            rmse_by_fold = []
            mae_by_fold = []
            for trainset, testset in kf.split(data):
                algo.fit(trainset)
                trimmed_testset_popular = self.trimPopular(testset, 2)
                predictions = algo.test(trimmed_testset_popular)
                # Compute Root Mean Squared Error
                rmse_by_fold.append(accuracy.rmse(predictions, verbose=True))
                # Compute MAE
                mae_by_fold.append(accuracy.mae(predictions, verbose=True))
            rmse_by_k.append(np.mean(rmse_by_fold))
            mae_by_k.append(np.mean(mae_by_fold))

        plt.plot(k_values, rmse_by_k)
        plt.plot(k_values, mae_by_k)
        plt.legend(['RMSE', 'MAE'])
        plt.show()
                
        # Q13
        rmse_by_k = []
        mae_by_k = []
        k_values = []
        for k in range(1, 51):
            k_values.append(2*k)
            algo = knns.KNNWithMeans(k=k*2, sim_options=sim_options)
            # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
            kf = KFold(n_splits=10)
            rmse_by_fold = []
            mae_by_fold = []
            for trainset, testset in kf.split(data):
                algo.fit(trainset)
                trimmed_testset_unpopular = self.trimUnpopular(testset, 2)
                print len(testset)
                print len(trimmed_testset_unpopular)
                predictions = algo.test(trimmed_testset_unpopular)
                # Compute Root Mean Squared Error
                rmse_by_fold.append(accuracy.rmse(predictions, verbose=True))
                # Compute MAE
                mae_by_fold.append(accuracy.mae(predictions, verbose=True))
            rmse_by_k.append(np.mean(rmse_by_fold))
            mae_by_k.append(np.mean(mae_by_fold))
            
        plt.plot(k_values, rmse_by_k)
        plt.plot(k_values, mae_by_k)
        plt.legend(['RMSE', 'MAE'])
        plt.show()
        
        # Q14
        rmse_by_k = []
        mae_by_k = []
        k_values = []
        for k in range(1, 51):
            k_values.append(2*k)
            algo = knns.KNNWithMeans(k=k*2, sim_options=sim_options)
            # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
            kf = KFold(n_splits=10)
            rmse_by_fold = []
            mae_by_fold = []
            for trainset, testset in kf.split(data):
                algo.fit(trainset)
                trimmed_testset_unpopular = self.trimHighVariance(testset, 2)
                print len(testset)
                print len(trimmed_testset_unpopular)
                predictions = algo.test(trimmed_testset_unpopular)
                # Compute Root Mean Squared Error
                rmse_by_fold.append(accuracy.rmse(predictions, verbose=True))
                # Compute MAE
                mae_by_fold.append(accuracy.mae(predictions, verbose=True))
            rmse_by_k.append(np.mean(rmse_by_fold))
            mae_by_k.append(np.mean(mae_by_fold))
            
        plt.plot(k_values, rmse_by_k)
        plt.plot(k_values, mae_by_k)
        plt.legend(['RMSE', 'MAE'])
        plt.show()
        
    def trimPopular(self, testset, threshold):
        df_testset = pd.DataFrame(testset, columns=['userId', 'movieId', 'rating'])
        counts = df_testset['movieId'].value_counts()
        df_trimmed_testset = df_testset[df_testset['movieId'].isin(counts[counts >= threshold].index)]
        return df_trimmed_testset.values.tolist()

    def trimUnpopular(self, testset, threshold):
        df_testset = pd.DataFrame(testset, columns=['userId', 'movieId', 'rating'])
        counts = df_testset['movieId'].value_counts()
        df_trimmed_testset = df_testset[df_testset['movieId'].isin(counts[counts <= threshold].index)]
        return df_trimmed_testset.values.tolist()

    def trimHighVariance(self, testset, minVariance):
        # print testset, len(testset)
        testset = self.trimPopular(testset, 5)
        dic = {}
        for (userID, movieID, rating) in dic:
            if (movieID in dic):
                dic[movieID].append(rating)
            else:
                dic[movieID] = [rating]
        for movieID in dic:
            if numpy.var(numpy.array(dic[movieID])) < minVariance:
                testset = filter(lambda x: x[1] != movieID, testset)
        return testset

