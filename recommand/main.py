from recommand import Recommand


if __name__ == '__main__':
    r = Recommand()
    # r.preprocessing()
    # r.knn()
    # r.NMF()
    r.run_and_test_all_models()
