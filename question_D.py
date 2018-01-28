import numpy as np
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
import string,re

#NEW Import
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF



def stemTokenizer(text):
	stemmer = SnowballStemmer("english")
	temp = ""
	for char in text:
		if char in string.punctuation:
			pass
		else:
			temp.join(char)
	temp = re.sub('[,.-:/()?><{}*$#&]','', text) #remove some special punc
	temp = "".join([i if ord(i) < 128 else ' ' for i in text])#remove non-ascii
	temp = re.sub('[^A-Za-z0-9]+', '', text)
	return [stemmer.stem(item) for item in temp.split()]


def get_graphic(category):
    return fetch_20newsgroups(
        subset='train',
        categories=category,
        shuffle=True,
        random_state=42
    )

if __name__ == '__main__':

	tfidf_transformer = TfidfTransformer()
	stop_words = text.ENGLISH_STOP_WORDS

	allCat = [ 'comp.sys.ibm.pc.hardware',
	       			'comp.sys.mac.hardware',
	                'misc.forsale',
	                'soc.religion.christian',
					'comp.graphics',
        			'comp.os.ms-windows.misc',
	                'comp.windows.x',
	                'rec.autos',
	                'rec.motorcycles',
	                'rec.sport.baseball',
	                'rec.sport.hockey',
	                'alt.atheism',
	                'sci.crypt',
	                'sci.electronics',
	                'sci.med',
	                'sci.space',
	                'talk.politics.guns',
	                'talk.politics.mideast',
	                'talk.politics.misc',
	                'talk.religion.misc'
	                ]

	allDoc = []
	for cat in allCat:
	    category_data = get_graphic([cat]).data
	    poke = ""
	    for doc in category_data:
	        poke = poke + "" + doc
	    allDoc.append(poke)
	    
	vectorizer = CountVectorizer(analyzer='word',stop_words=stop_words, min_df=2, tokenizer=stemTokenizer)
	vectors = vectorizer.fit_transform(allDoc)
	tfidf_train = TfidfTransformer().fit_transform(vectorizer)

	#pply LSI to the TFxIDF matrix corresponding to the 8 classes. and pick k=50; so each document is mapped to a 50-dimensional vector. Alternatively, reduce dimensionality through Non-Negative Matrix Factorization (NMF) and compare the results of the parts e-i using both methods.
	SVD = TruncatedSVD(n_components=50, random_state=42)
	transformed_tfidf = SVD.fit_transform(tfidf_train)
	print transformed_tfidf.shape

