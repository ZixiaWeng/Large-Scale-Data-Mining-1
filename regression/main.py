from regression import Regression


if __name__ == '__main__':
    r = Regression()
    # Q1a    
    r.plot_buSize(5)

    r.plot_buSize(20)
    # Q1b
    # r.plot_buSize(105)
    r.random_forest()
