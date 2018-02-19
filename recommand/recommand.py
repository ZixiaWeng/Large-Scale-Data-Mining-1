import csv
import numpy
import matplotlib.pyplot as plt


def read_data():
    # Loading Ratings.csv
    ratings = {}
    ratings['user'] = []
    ratings['movie'] = []
    ratings['rating'] = []

    filename = 'recommand/ml-latest-small/ratings.csv'
    with open (filename , "rt") as input:
      reader = csv.reader(input, delimiter=',', quoting=csv.QUOTE_NONE)
      next(reader, None) # skip header
      for line in reader:
        #print line
        ratings['user'].append( float(line[0]))
        ratings['movie'].append( float(line[1]))
        ratings['rating'].append( float(line[2]))

    return ratings


class Recommand:
    def __init__(self):
        self.ratings = read_data()

    def preprocessing(self):  # q1-6
        # 100004 is the number of available ratings
        # 671 is the number of users
        # 9125 is the number of movies
        print "Sparisty = " + str(100004 / 671*9125)

        # Q2
        plot2_y = numpy.zeros(11)
        for i in range (len(self.ratings['rating'])):
          plot2_y [  int (self.ratings['rating'][i] / 0.5) ] += 1
        plt.bar( range(0,11), plot2_y)
        plt.show()
        
        '''
        # Q3
        movie_count = {}
        for i in range (len(self.ratings['rating'])):
          key = int (self.ratings['movie'][i])
          if not key in movie_count.keys():
            movie_count[ self.ratings['movie'][i] ] = 1
          else:
            movie_count[ self.ratings['movie'][i] ] += 1
        plt.bar( range(len(movie_count)), movie_count.values())
        plt.xticks(range(len(movie_count)), movie_count.keys())
        plt.show()
        
        # Q4
        user_count = {}
        for i in range (len(self.ratings['rating'])):
          key = int (self.ratings['user'][i])
          if not key in movie_count.keys():
            user_count[ self.ratings['user'][i] ] = 1
          else:
            user_count[ self.ratings['user'][i] ] += 1
        plt.bar( range(len(user_count)), user_count.values())
        plt.xticks(range(len(user_count)), user_count.keys())
        plt.show()
        '''
        
        '''
        # Q6
        rating_by_moive = {}
        for i in range (len(self.ratings['rating'])):
            movie_key = (self.ratings['movie'][i])
            if not movie_key in rating_by_moive.keys():
                rating_by_moive[ movie_key ] = []
                rating_by_moive[ movie_key ].append( self.ratings['rating'][i] )
            else:
                rating_by_moive[ movie_key ].append( self.ratings['rating'][i] )
        variances = []
        for ratings in rating_by_moive.values():
            variances.append ( numpy.var(ratings))
        count_by_binned_variance = {}
        for variance in variances:
            bin_key = int (variance / 0.5)
            if not bin_key in count_by_binned_variance.keys():
                count_by_binned_variance[ bin_key] = 1
            else:
                count_by_binned_variance[ bin_key] += 1
        plt.bar( range(len(count_by_binned_variance)), count_by_binned_variance.values())
        plt.xticks( range(len(count_by_binned_variance)), count_by_binned_variance.keys())
        plt.show()
        '''
          
    def knn(self):  # q7-11
        pass




