Project 2 ReadMe

Completed by Zixia Weng(305029822) and Shunji Zhan(405030387)

Our code is organized in a class named Clustering, which contains all the importing libraries and relating methods for all the models we used in this project. It also contains some variables declared in the beginning such as tfidf matrix, training data, training data labels, etc for the easy use later in each method. In the utils.py, it contains some helper functions. In the main.py, we just run the functions in the Clustering under the project instructions from question 1 to 5. For question 5, we put all the output in our report to see how purely will 20 groups be clustered in this context.

Type

Make

to run all the code. You will see the details of the output in the stdout. Note that this will only run the questions from 1-4, for question 5, you should change the "categories" variable in line67: self.train_data = fetch_data(categories, 'train') in Clustering.py to "allCat" and change the  "n_clusters=2" in every km_svd = KMeans(n_clusters=2, max_iter=100, n_init=3) statement to "n_clusters=20", for your convenience, we run the code in advance and put the output in our report. Enjoy!


Thank you so much!

Sincerely,
Zixia and Shunji
