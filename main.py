import pprint as pp
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def get_graphic(category):
    return fetch_20newsgroups(
        subset='train',
        categories=category,
        shuffle=True,
        random_state=42
    )['data']


def get_graphic_len(category):
    return len(get_graphic(category))


def question_a():
    categories = [
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey'
    ]

    res = {}
    for c in categories:
        res[c] = get_graphic_len([c])

    plt.bar(res.keys(), res.values(), 0.5, color=['g', 'r', 'b', 'y', 'r', 'y', 'r', 'y'])
    plt.show()


if __name__ == '__main__':
    
    text = get_graphic(['comp.graphics'])[0]
    print text
    # pp.pprint(Counter())


