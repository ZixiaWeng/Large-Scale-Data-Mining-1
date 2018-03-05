from regression import Regression
import numpy as np


if __name__ == '__main__':
    r = Regression()
    # Q1Test
    # r.plot_buSize(5)
    # Q1a
    # r.plot_buSize(20)
    # Q1b
    # r.plot_buSize(105)
    # r.linear_regression()
    # r.f_reg()
    r.random_forest()
    # r.rsmeUnderCombineEncoding()
    # r.ridgeRegularizer()
    # r.runRidgeAndLasso()
    # r.f_reg()
    r.nn()
    r.knn()
    # r.mutual_info_reg()
    # r.predictBUSizeLinear()
    # r.predictBUSizePolynomial()
