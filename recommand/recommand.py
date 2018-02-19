import csv
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.prediction_algorithms import knns

def read_data():
    # Loading Ratings.csv
    ratings = {}
    sparse = {}
    ratings['user'], sparse['user'] = [],[]
    ratings['movie'], sparse['movie'] = [],[]
    ratings['rating'] = []
    filename = 'recommand/ml-latest-small/ratings.csv'
    with open (filename , "rt") as input:
      	reader = csv.reader(input, delimiter=',', quoting=csv.QUOTE_NONE)
      	next(reader, None) # skip header
      	for line in reader:
      		ratings['user'].append( float(line[0]))
      		ratings['movie'].append( float(line[1]))
        	ratings['rating'].append( float(line[2]))
        	if float(line[0]) not in sparse['user']:
        		sparse['user'].append( float(line[0])) #available users
        	if float(line[1]) not in sparse['movie']:
        		sparse['movie'].append( float(line[1])) #available movies
    sparisty = len(ratings['rating'])/(float((len(sparse['user'])) * len(sparse['movie'])))
    return ratings, sparisty


class Recommand:
    def __init__(self):
        self.ratings, self.sparisty = read_data()

    def preprocessing(self):  # q1-6
    	  # Q1
        print "Sparisty = " + str(self.sparisty)

        # Q2
        plot2_y = numpy.zeros(11)
        for i in range (len(self.ratings['rating'])):
            plot2_y [  int (self.ratings['rating'][i] / 0.5) ] += 1
        plt.bar( range(0,11), plot2_y)
        plt.show()

        # Q3
        movie_count = {}
        for i in range (len(self.ratings['rating'])):
            key = int (self.ratings['movie'][i])
            if not key in movie_count.keys():
                movie_count[ self.ratings['movie'][i] ] = 0
            else:
                movie_count[ self.ratings['movie'][i] ] += 1
        movie_rats = [(i, movie_count[i]) for i in movie_count]
        movie_rats = sorted(movie_rats, cmp = lambda x, y: cmp(y[1], x[1]))
        movie_plot = [i[1] for i in movie_rats]
        plt.plot(movie_plot)
        plt.show()


    def knn(self):  # q7-11
        filename = 'recommand/ml-latest-small/ratings.csv'
        reader = Reader()
        #data = Dataset.load_from_file(filename, reader=reader )
        df = pd.DataFrame(self.ratings)
        data = Dataset.load_from_df(df[['user', 'movie', 'rating']], reader)
        sim_options = {'name': 'pearson_baseline',
          'shrinkage': 0  # no shrinkage
        }
        algo = knns.KNNWithMeans(k = 40, sim_options=sim_options)
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
        

