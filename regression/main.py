from regression import Regression
import numpy as np


if __name__ == '__main__':
    r = Regression()
    # Q1a
    r.plot_buSize(20)
    # Q1b
    r.plot_buSize(105)
    r.linear_regression_whole_Data_Points()
    r.linear_regression()
    # r.f_reg()
    # r.random_forest()

    # Q2a iv.
    # r.featureCombinationEncoding()
    # r.rsmeUnderCombineEncoding()
    # r.linear_regression_whole_Data_Points_best_combination()
    # r.ridgeRegularizer()
    # r.runRidgeAndLasso()
    r.getCoefficient()
    # r.f_reg()
    # r.nn()
    # r.knn()
    # r.mutual_info_reg()
    # r.predictBUSizeLinear()
    # r.predictBUSizePolynomial()
