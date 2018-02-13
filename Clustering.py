import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from utils import print_question, stemTokenizer, fetch_data, build_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction import text


stop_words = text.ENGLISH_STOP_WORDS
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
allCat = [
    'comp.sys.ibm.pc.hardware',
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


class Clustering:
    def __init__(self):
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words, min_df=3, tokenizer=stemTokenizer)
        self.svd = TruncatedSVD(n_components=1000, random_state=0)
        self.nmf = NMF(n_components=50, random_state=42)
        self.nmfp2 = NMF(n_components=1000, random_state=0)

        # build training data
        self.train_data = fetch_data(categories, 'train')
        self.train_labels = build_labels(self.train_data)
        self.vectors = self.to_vec(self.train_data.data)
        self.tfidf = self.to_tfidf(self.vectors)
        # self.tfidf_SVD = self.to_SVD(self.tfidf)
        # self.tfidf_NMF = self.to_NMF(self.tfidf)

    def _transform(self, data, tool):
        return tool.fit_transform(data)

    def to_vec(self, data):
        return self._transform(data, self.vectorizer)

    def to_tfidf(self, data):
        return self._transform(data, self.tfidf_transformer)

    def to_SVD(self, data):
        return self._transform(data, self.svd)

    def to_NMF(self, data):
        return self._transform(data, self.nmf)

    def q1(self):
        print_question('1')
        print "dimensions: ", self.tfidf.shape
        return self.tfidf

    def q2(self):
        print_question('2')
        km = KMeans(n_clusters=2, n_init=100, max_iter=1000)  # k = 2
        km.fit(self.tfidf)
        print(self.train_labels, km.labels_)
        homo_score = metrics.homogeneity_score(self.train_labels, km.labels_)
        complete_score = metrics.completeness_score(self.train_labels, km.labels_)
        v_score = metrics.v_measure_score(self.train_labels, km.labels_)
        rand_score = metrics.adjusted_rand_score(self.train_labels, km.labels_)
        mutual_info = metrics.adjusted_mutual_info_score(self.train_labels, km.labels_)
        print("Homogeneity Score: %0.3f" % homo_score)
        print("Completeness Score: %0.3f" % complete_score)
        print("V-measure: %0.3f" % v_score)
        print("Adjusted Rand Score: %0.3f" % rand_score)
        print("Adjusted Mutual Info Score: %0.3f\n" % mutual_info)

    def q3(self):
        svd = self.svd
        svd.fit(self.tfidf)
        plt.plot(range(1, 1001), svd.explained_variance_ratio_.cumsum().tolist(), label="SVD")
        plt.xlabel('r value')
        plt.ylabel('Percent of variance retained')
        plt.title('Variance retained vs r value for Truncated SVD')
        plt.show()

        r = [1, 2, 3, 5, 10, 20, 50, 100, 300]