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
import main

#This File contains Question D

#pply LSI to the TFxIDF matrix corresponding to the 8 classes. and pick k=50; so each document is mapped to a 50-dimensional vector. Alternatively, reduce dimensionality through Non-Negative Matrix Factorization (NMF) and compare the results of the parts e-i using both methods.
tfidf_train = TfidfTransformer().fit_transform(main.vectors)
SVD = TruncatedSVD(n_components=50, random_state=42)
transformed_tfidf = SVD.fit_transform(tfidf_train)
print transformed_tfidf.shape

