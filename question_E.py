import numpy as np
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
import string,re
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

from sklearn import svm
import main

def stemTokenizer(text):
    stemmer = SnowballStemmer("english")
    temp = "".join([i if ord(i) < 128 else ' ' for i in text])#remove non-ascii
    temp = re.sub('[,.-:/()?><{}*$#&]','', temp) #remove some special punc
    tem = "".join(c for c in temp if c not in string.punctuation)
    return [stemmer.stem(item) for item in temp.split()]


def get_graphic_train(category):
    return fetch_20newsgroups(
        subset='train',
        categories=category,
        shuffle=True,
        random_state=42
    )

def get_graphic_test(category):
    return fetch_20newsgroups(
        subset='test',
        categories=category,
        shuffle=True,
        random_state=42
    )

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

vectorizer = text.TfidfVectorizer(
    stop_words=stop_words,
    encoding='unicode',
    analyzer='word',
    min_df=5,
    tokenizer=stemTokenizer
)

# Main
trainingPoints = []
testingPoints = []
test = get_graphic_test(categories)
test_counts = count_vect_final.fit_transform(test.data)
tfidf_test = tfidf_transformer.fit_transform(test_counts)
transformed_test_tfidf = SVD.fit_transform(tfidf_test)

# Hard Classifier
h_c = svm.LinearSVC(C = 1000, dual = False, random_state = 42)

for i in get_graphic_train(categories).target:
	if i<4:	
		trainingPoints.append('x')
	else:
		trainingPoints.append('y')
print trainingPoints
h_c.fit(question_D.transformed_tfidf, trainingPoints)



# print doc.target, len(doc.target)




