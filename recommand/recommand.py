import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pprint as pp
import operator

from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import AlgoBase
from surprise.model_selection import KFold
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import knns
from surprise.prediction_algorithms import matrix_factorization
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from surprise.prediction_algorithms.matrix_factorization import SVD


def plot_freqency(d, msg=None):
    sorted_ = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    ids, ratings = zip(*sorted_)
    plot_bar(ratings, msg)


def plot_bar(arr, msg=None):
    plt.bar(np.arange(len(arr)), arr)
    plt.title(msg)
    plt.show()


def plot_precision_and_recall(t_values, precisions_by_t, recall_by_t, msg):
    plt.plot(t_values, precisions_by_t)
    plt.title('%s precisions' % msg)
    plt.show()

    plt.plot(t_values, recall_by_t)
    plt.title('%s recall' % msg)
    plt.show()

    plt.plot(recall_by_t, precisions_by_t)
    plt.title('%s precisions VS recall' % msg)
    plt.show()


def trimPopular(testset, threshold):
    df_testset = pd.DataFrame(testset, columns=['userId', 'movieId', 'rating'])
    counts = df_testset['movieId'].value_counts()
    df_trimmed_testset = df_testset[df_testset['movieId'].isin(counts[counts >= threshold].index)]
    return df_trimmed_testset.values.tolist()


def trimUnpopular(testset, threshold):
    df_testset = pd.DataFrame(testset, columns=['userId', 'movieId', 'rating'])
    counts = df_testset['movieId'].value_counts()
    df_trimmed_testset = df_testset[df_testset['movieId'].isin(counts[counts <= threshold].index)]
    return df_trimmed_testset.values.tolist()


def trimHighVariance(testset, minVariance):
    # print type(testset), len(testset)
    testsetTemp = trimPopular(testset, 5)
    dic = {}
    for (userID, movieID, rating) in testsetTemp:
        if (movieID in dic):
            dic[movieID].append(rating)
        else:
            dic[movieID] = [rating]
    for movieID in dic:
        if np.var(np.array(dic[movieID])) < minVariance:
            testsetTemp = filter(lambda x: x[1] != movieID, testsetTemp)  # (userID, movieID, rating), x[1] is movieID
    return testsetTemp


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
        next(reader, None)  # skip header
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


def read_csv_data(path):
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # df = pd.read_csv(path)
    return Dataset.load_from_file(path, reader)


class NaiveFiltering(AlgoBase):
    # https://github.com/jdhurwitz/UCLA-EE219/blob/master/Project3/project3.ipynb
    def __init__(self):
        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def fit(self, trainset):
        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        self.movie_rating = defaultdict(list)
        for (user, _, rating) in self.trainset.all_ratings():
            self.movie_rating[user].append(rating)

        for user in self.movie_rating:
            self.movie_rating[user] = np.mean(self.movie_rating[user])

        # print self.movie_rating
        return self

    def estimate(self, u, i):
        return self.movie_rating[u]


class Recommand:
    def __init__(self):
        self.data = read_csv_data('recommand/ml-latest-small/ratings.csv')
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

    def run_with_diff_k(self, algo, args, range_, folds=2, test_filter=None, threshold=2, msg=None, modal_name=None):
        arg_name = {
            'KNN': 'k',
            'NMF': 'n_factors',
            'SVD': 'n_factors'
        }[modal_name]

        rmse_by_k = []
        mae_by_k = []
        k_values = []
        for k in range(*range_):
            k_values.append(k)
            args.update({arg_name: k})
            modal = algo(**args)
            kf = KFold(n_splits=folds)
            rmse_by_fold = []
            mae_by_fold = []
            for trainset, testset in kf.split(self.data):
                modal.fit(trainset)
                if test_filter:
                    testset = test_filter(testset, threshold)
                predictions = modal.test(testset)
                rmse_by_fold.append(accuracy.rmse(predictions, verbose=True))
                mae_by_fold.append(accuracy.mae(predictions, verbose=True))
            rmse_by_k.append(np.mean(rmse_by_fold))
            mae_by_k.append(np.mean(mae_by_fold))
   
        plt.plot(k_values, rmse_by_k)
        plt.plot(k_values, mae_by_k)
        plt.legend(['RMSE', 'MAE'])
        plt.title(msg)
        plt.show()

    def run_and_test_model(self, algo, algo_args, model, range_, modal_name=None):
        # self.run_with_diff_k(algo, algo_args, range_, test_filter=None, msg='not trimed %s' % modal_name, modal_name=modal_name)
        # self.run_with_diff_k(algo, algo_args, range_, test_filter=trimPopular, msg='trimPopular %s' % modal_name, modal_name=modal_name)
        # self.run_with_diff_k(algo, algo_args, range_, test_filter=trimUnpopular, msg='trimUnpopular %s' % modal_name, modal_name=modal_name)
        # self.run_with_diff_k(algo, algo_args, range_, test_filter=trimHighVariance, msg='trimHighVariance %s' % modal_name, modal_name=modal_name)

        trainset, testset = train_test_split(self.data, test_size=0.1)
        threshold = [2.5, 3.0, 3.5, 4.0]
        model.fit(trainset)
        pred = model.test(testset)
        trueValue, scoreValue = [], []
        for x in pred:
            trueValue.append(x[2])  # r_ui
            scoreValue.append(x[3])  # est

        all_roc_auc = []
        for th in threshold:
            realValue = map(lambda x: 0 if x < th else 1, trueValue)
            fpr, tpr, thresholds = roc_curve(realValue, scoreValue)
            roc_auc = auc(fpr, tpr)
            all_roc_auc.append((fpr, tpr))
            plt.plot(fpr, tpr, color='blue', linewidth=2.0, label='ROC (Area is %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='yellow', linewidth=2.0)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC with threshold = ' + str(th))
            plt.legend(loc="lower right")
            plt.show()

        return all_roc_auc

    def run_naive_filter(self, folds=5, test_filter=None, threshold=2, msg=None):
        model = NaiveFiltering()
        kf = KFold(n_splits=folds)
        rmse_by_fold = []
        mae_by_fold = []
        for trainset, testset in kf.split(self.data):
            model.fit(trainset)
            if test_filter:
                testset = test_filter(testset, threshold)
            predictions = model.test(testset)
            print predictions
            rmse_by_fold.append(accuracy.rmse(predictions))
            mae_by_fold.append(accuracy.mae(predictions))

        print '%s naive filter: ' % msg
        print 'rmse: %.3f' % np.mean(rmse_by_fold)
        print 'mae: %.3f' % np.mean(mae_by_fold)

    def run_and_test_all_models(self):
        step_size = 2

        # KNN
        sim_options = {
            'name': 'pearson_baseline',
            'shrinkage': 0  # no shrinkage
        }
        algo = knns.KNNWithMeans
        args = {'sim_options': sim_options}
        best_model = knns.KNNWithMeans(k=20, sim_options=sim_options)
        roc_auc_KNN = self.run_and_test_model(algo, args, best_model, (2, 101, step_size), 'KNN')

        # # NMF
        algo = matrix_factorization.NMF
        args = {'biased': False}
        best_model = matrix_factorization.NMF(n_factors=20, biased=False)
        roc_auc_NMF = self.run_and_test_model(algo, args, best_model, (2, 51, step_size), 'NMF')

        # SVD
        algo = matrix_factorization.SVD
        args = {}
        best_model = SVD(20)
        roc_auc_SVD = self.run_and_test_model(algo, args, best_model, (2, 51, step_size), 'SVD')

        # all
        for i in range(len(roc_auc_KNN)):
            plt.plot(roc_auc_KNN[i][0], roc_auc_KNN[i][1], color='blue', linewidth=2.0, label='KNN')
            plt.plot(roc_auc_NMF[i][0], roc_auc_NMF[i][1], color='blue', linewidth=2.0, label='NMF')
            plt.plot(roc_auc_SVD[i][0], roc_auc_SVD[i][1], color='blue', linewidth=2.0, label='SVD')
            plt.plot([0, 1], [0, 1], color='yellow', linewidth=2.0)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend(loc="lower right")
            plt.show()

        # NaiveFilter
        self.run_naive_filter(msg='normal')
        self.run_naive_filter(test_filter=trimPopular, msg='trimPopular')
        self.run_naive_filter(test_filter=trimUnpopular, msg='trimUnpopular')
        self.run_naive_filter(test_filter=trimHighVariance, msg='trimHighVariance')

    def non_negative_matrix_factorization(self):
        algo = matrix_factorization.NMF(20)
        model = algo.fit(self.data.build_full_trainset())
        U = model.pu
        V = model.qi
        print U.shape
        print V.shape

        top_ten = V[:, 1].argsort()[::-1][:10]

        movies = {}
        genre = []
        f = open('recommand/ml-latest-small/movies.csv')
        movies_reader = csv.reader(f)
        for movie_entry in movies_reader:
            movies[movie_entry[0]] = movie_entry[2]
        f.close()
        for i in top_ten:
            for j in movies:
                if j != "movieId" and int(i) == int(j):
                    genre.append([i, movies[j]])
        print (genre)

    def evaluate_pred(self, predictions, t):
        predictions_by_user = {}
        for prediction in predictions:
            uid = prediction[0]
            movieid = prediction[1]
            rating = prediction[2]
            estimated_rating = prediction[3]
            if uid not in predictions_by_user.keys():
                predictions_by_user[uid] = []
            predictions_by_user[uid].append([movieid, rating, estimated_rating])

        precisons_by_user = []
        recall_by_user = []
        for ratings_by_user in predictions_by_user.values():
            if len(ratings_by_user) < t:
                continue
            ratings_by_user.sort(key=lambda x: x[2])
            s_count = t
            g_count = 0
            s_intersect_g_count = 0
            for i in range(len(ratings_by_user)):
                if ratings_by_user[i][1] > 2.5:
                    g_count += 1
                    if i < t:
                        s_intersect_g_count += 1
            if not g_count == 0:
                precisons_by_user.append(float(s_intersect_g_count) / s_count)
                recall_by_user.append(float(s_intersect_g_count) / g_count)
        return np.mean(precisons_by_user), np.mean(recall_by_user)

    def test_with_t_and_k(self, best_model, folds=3, msg=None):
        precisions_by_t = []
        recall_by_t = []
        t_values = []
        for t in range(1, 26, 2):
            print t
            t_values.append(t)
            kf = KFold(n_splits=folds)
            precisions_by_fold = []
            recall_by_fold = []
            for trainset, testset in kf.split(self.data):
                best_model.fit(trainset)
                predictions = best_model.test(testset)
                precision, recall = self.evaluate_pred(predictions, t)
                precisions_by_fold.append(precision)
                recall_by_fold.append(recall)
            precisions_by_t.append(np.mean(precisions_by_fold))
            recall_by_t.append(np.mean(recall_by_fold))

        plot_precision_and_recall(t_values, precisions_by_t, recall_by_t, msg)
        return t_values, precisions_by_t, recall_by_t

    def recommand(self):
        sim_options = {
            'name': 'pearson_baseline',
            'shrinkage': 0  # no shrinkage
        }
        best_model = knns.KNNWithMeans(k=20, sim_options=sim_options)
        t_values, precisions_knn, recall_knn = self.test_with_t_and_k(best_model, msg='KNN')

        best_model = matrix_factorization.NMF(n_factors=20, biased=False)
        t_values, precisions_nmf, recall_nmf = self.test_with_t_and_k(best_model, msg='NMF')

        best_model = SVD(20)
        t_values, precisions_svd, recall_svd = self.test_with_t_and_k(best_model, msg='SVD')

        plt.plot(t_values, precisions_knn, label='precisions_knn')
        plt.plot(t_values, precisions_nmf, label='precisions_nmf')
        plt.plot(t_values, precisions_svd, label='precisions_svd')
        plt.plot(t_values, recall_knn, label='recall_knn')
        plt.plot(t_values, recall_nmf, label='recall_nmf')
        plt.plot(t_values, recall_svd, label='recall_svd')
        plt.xlabel('t_value')
        plt.ylabel('percent')
        plt.legend(loc="best")
        plt.show()
