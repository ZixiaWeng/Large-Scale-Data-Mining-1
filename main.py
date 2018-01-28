import pprint as pp
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

#This File contains Question a&b

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

if __name__ == '__main__':
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
    vectorizer = text.TfidfVectorizer(
        stop_words=stop_words,
        encoding='unicode',
        analyzer='word',
        min_df=5,
        # tokenizer=ps.stem
    )
    vectors = vectorizer.fit_transform(doc.data)
    vectorizer.fit(doc.data)
    # print vectors.shape
    print(vectorizer.vocabulary_,len(vectorizer.vocabulary_))
    print(vectorizer.idf_)
    # print text.strip('\n').rstrip('\n').split(' ')
    # pp.pprint(Counter())
    

