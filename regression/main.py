from regression import Regression
import numpy as np


if __name__ == '__main__':
    r = Regression()
    # Q1a
    r.plot_buSize(20)
    # Q1b
    r.plot_buSize(105)
    # Q2a i&ii now is standardized, should comment 2 lines of standardization of code to see normal regression model
    r.linear_regression_whole_Data_Points()
    r.linear_regression()
    # Q2a iii
    # r.f_reg()
    # r.random_forest()

    # Q2a iv.
    r.featureCombinationEncoding()
    r.rsmeUnderCombineEncoding()
    r.linear_regression_whole_Data_Points_best_combination()
    # Q2a v
    r.ridgeRegularizer()
    r.runRidgeAndLasso()
    r.getCoefficient()

    # r.f_reg()
    # r.nn()
    # r.knn()
    # r.mutual_info_reg()
    # r.predictBUSizeLinear()
    # r.predictBUSizePolynomial()
