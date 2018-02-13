import random
import re
import string

from nltk.stem.snowball import SnowballStemmer
from sklearn.datasets import fetch_20newsgroups


def rand_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return '#%02X%02X%02X' % (r, g, b)


def rand_color_arr(length):
    res = []
    for i in range(length):
        res.append(rand_color())
    return res


def stemTokenizer(para):
    stemmer = SnowballStemmer("english")
    temp = "".join([i if ord(i) < 128 else ' ' for i in para])  # remove non-ascii
    temp = re.sub('[,.-:/()?><{}*$#&]', '', temp)  # remove some special punc
    temp = "".join(c for c in temp if c not in string.punctuation)
    return [stemmer.stem(item) for item in temp.split()]


def tokenizer(para):
    temp = "".join([i if ord(i) < 128 else ' ' for i in para])  # remove non-ascii
    temp = re.sub('[,.-:/()?><{}*$#&]', '', temp)  # remove some special punc
    temp = "".join(c for c in temp if c not in string.punctuation)
    return temp.split()


def fetch_data(category, subset):
    assert(subset in {'train', 'test'})
    return fetch_20newsgroups(
        subset=subset,
        categories=category,
        shuffle=True,
        random_state=42
    )


def new_line(length):
    print '=' * length


def build_labels(data):
    labels = []
    for i in data.target:
        labels.append(1) if i >= 4 else labels.append(0)
    return labels


def print_question(x):
    print ' '
    print '=' * 50
    print '=' * 18 + '  Question %s  ' % str(x) + '=' * 18
    print '=' * 50



