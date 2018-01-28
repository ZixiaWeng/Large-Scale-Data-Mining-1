import numpy as np
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
import string,re

#This File contains Question C

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
	print(vectors.shape)
	tficf_train = tfidf_transformer.fit_transform(vectors)

	tficf_train_copy = tficf_train.copy()
	features = vectorizer.get_feature_names()
	for i in range(4):
	    temp = []
	    for j in range(10):
	        temp.append(features[np.argmax(tficf_train_copy[i])])
	        tficf_train_copy[i, np.argmax(tficf_train_copy[i])] = 0
	    print(allCat[i],temp)

