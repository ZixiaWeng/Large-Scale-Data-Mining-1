import pprint as pp
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import string,re

#This File contains Question a&b
def stemTokenizer(text):
    stemmer = SnowballStemmer("english")
    temp = "".join([i if ord(i) < 128 else ' ' for i in text])#remove non-ascii
    temp = re.sub('[,.-:/()?><{}*$#&]','', temp) #remove some special punc
    tem = "".join(c for c in temp if c not in string.punctuation)
    return [stemmer.stem(item) for item in temp.split()]


def get_graphic(category):
    return fetch_20newsgroups(
        subset='train',
        categories=category,
        shuffle=True,
        random_state=42
    )


def get_graphic_len(category):
    return len(get_graphic(category).data)


def question_a():
    res = {}
    for c in categories:
        res[c] = get_graphic_len([c])

    # Plot the histogram
    plt.bar(res.keys(), res.values(), 0.5, color=['g', 'r', 'b', 'y', 'r', 'y', 'r', 'y'])
    plt.show()


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
stop_words = text.ENGLISH_STOP_WORDS

doc = get_graphic(categories)
ps = PorterStemmer()

#When Min_df = 5
vectorizer = text.TfidfVectorizer(
    stop_words=stop_words,
    encoding='unicode',
    analyzer='word',
    min_df=5,
    tokenizer=stemTokenizer
)
vectors = vectorizer.fit_transform(doc.data)
vectorizer.fit(doc.data)
print(vectors.shape, len(vectorizer.vocabulary_))
print(vectorizer.idf_)



#When Min_df = 2
vectorizer_df2 = text.TfidfVectorizer(
    stop_words=stop_words,
    encoding='unicode',
    analyzer='word',
    min_df=2,
    tokenizer=stemTokenizer
)
vectors_df2 = vectorizer_df2.fit_transform(doc.data)
vectorizer_df2.fit(doc.data)
print(vectors_df2.shape, len(vectorizer_df2.vocabulary_))
print(vectorizer_df2.idf_)
