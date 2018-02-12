import numpy as np
import matplotlib.pyplot as plt

from utils import rand_color_arr, stemTokenizer, fetch_data, build_labels, new_line, print_question
from collections import defaultdict
from sklearn import svm
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score

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
cat_4 = [
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'misc.forsale',
    'soc.religion.christian'
]


class TextAnalyzer:
    def __init__(self):
        new_line(50)
        print 'started to build all training and testing data...'
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words, min_df=5, tokenizer=stemTokenizer)
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.svdp2 = TruncatedSVD(n_components=1000, random_state=0)
        self.nmf = NMF(n_components=50, random_state=42)
        self.nmfp2 = NMF(n_components=1000, random_state=0)
        self.mm = MinMaxScaler()
        self.r = [1, 2, 3, 5, 10, 20, 50, 100, 300]

        # build training data
        self.train_data = fetch_data(categories, 'train')
        self.train_labels = build_labels(self.train_data)
        self.vectors = self.to_vec(self.train_data.data)
        self.tfidf = self.to_tfidf(self.vectors)
        self.tfidf_SVD = self.to_SVD(self.tfidf)
        self.tfidf_NMF = self.to_NMF(self.tfidf)
        self.tfidf_mm = self.mm.fit_transform(self.tfidf_SVD)

        # build testing data
        self.test_data = fetch_data(categories, 'test')
        self.test_labels = build_labels(self.test_data)
        self.test_vectors = self.vectorizer.transform(self.test_data.data)
        self.test_tfidf = self.tfidf_transformer.transform(self.test_vectors)
        self.test_tfidf_SVD = self.svd.transform(self.test_tfidf)
        self.test_tfidf_NMF = self.nmf.transform(self.test_tfidf)
        self.test_tfidf_mm = self.mm.fit_transform(self.test_tfidf_SVD)
        print 'finished building all training and testing data...'
        new_line(50)
        print ' '

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

    def plot_ROC(self, score):
        fpr, tpr, _ = roc_curve(self.test_labels, score)
        plt.plot([0, 1], [0, 1])
        plt.plot(fpr, tpr)
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.title('ROC Curve')
        plt.axis([0, 1, 0, 1])
        plt.show()

    def show_result(self, score, acc, report, matrix, plot=True):
        print "Accuracy: %.2f" % acc
        print "Classification Report:"
        print report
        print "Confusion Matrix:"
        print matrix
        if plot:
            self.plot_ROC(score)

    def svm_classify_SVD(self, classifier, name):
        print '================== %s with SVD ==================' % name
        # train model
        classifier.fit(self.tfidf_SVD, self.train_labels)

        # make prediction
        prediction = classifier.predict(self.test_tfidf_SVD)
        score = classifier.decision_function(self.test_tfidf_SVD)
        acc = np.mean(prediction == self.test_labels)
        report = classification_report(self.test_labels, prediction, target_names=['Computer technology', 'Recreational activity'])
        matrix = confusion_matrix(self.test_labels, prediction)

        return score, acc, report, matrix

    def svm_classify_NMF(self, classifier, name):
        print '================== %s with NMF ==================' % name
        # train model
        classifier.fit(self.tfidf_NMF, self.train_labels)

        # make prediction
        prediction = classifier.predict(self.test_tfidf_NMF)
        score = classifier.decision_function(self.test_tfidf_NMF)
        acc = np.mean(prediction == self.test_labels)
        report = classification_report(self.test_labels, prediction, target_names=['Computer technology', 'Recreational activity'])
        matrix = confusion_matrix(self.test_labels, prediction)

        return score, acc, report, matrix

    def prob_classify_SVD(self, classifier, name):
        print '================== %s with SVD ==================' % name
        # train model
        classifier.fit(self.tfidf_mm, self.train_labels)

        # make prediction
        prediction = classifier.predict(self.test_tfidf_mm)
        prob = classifier.predict_proba(self.test_tfidf_mm[:])[:, 1]
        acc = np.mean(prediction == self.test_labels)
        report = classification_report(self.test_labels, prediction, target_names=['Computer technology', 'Recreational activity'])
        matrix = confusion_matrix(self.test_labels, prediction)

        return prob, acc, report, matrix

    def prob_classify_NMF(self, classifier, name):
        print '================== %s with NMF ==================' % name
        # train model
        classifier.fit(self.tfidf_NMF, self.train_labels)

        # make prediction
        prediction = classifier.predict(self.test_tfidf_NMF)
        prob = classifier.predict_proba(self.test_tfidf_NMF[:])[:, 1]
        acc = np.mean(prediction == self.test_labels)
        report = classification_report(self.test_labels, prediction, target_names=['Computer technology', 'Recreational activity'])
        matrix = confusion_matrix(self.test_labels, prediction)

        return prob, acc, report, matrix

    def multi_classify(self, classifier, nmf_train, nmf_test, train_labels, test_labels, name):
        print '================== %s ==================' % name
        classifier.fit(nmf_train, train_labels)

        prediction = classifier.predict(nmf_test)
        acc = np.mean(prediction == test_labels)
        report = classification_report(test_labels, prediction, target_names=cat_4)
        matrix = confusion_matrix(test_labels, prediction)
        self.show_result(-1, acc, report, matrix, plot=False)

    def run_all(self):
            self.a()
            self.b()
            self.c()
            self.d()
            self.e()
            self.f()
            self.g()
            self.h()
            self.i()
            self.j()
    def p2q1(self):
        print_question('1')
        vectorizer_3 = TfidfVectorizer(analyzer='word', stop_words=stop_words, min_df=3, tokenizer=stemTokenizer)
        tfidf = vectorizer_3.fit_transform(self.train_data.data)
        print "dimensions: ", tfidf.shape
        return tfidf

    def p2q2(self):
        tfidf = self.p2q1()
        print_question('2')
        km = KMeans(n_clusters = 2, n_init = 100, max_iter = 1000) #k = 2
        km.fit(tfidf)
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

    def p2q3(self):
        tfidf = self.p2q1()
        svd = self.svdp2
        svd.fit(tfidf)
        plt.plot(range(1,1001), svd.explained_variance_ratio_.cumsum().tolist(), label="SVD")
        plt.xlabel('r value')
        plt.ylabel('Percent of variance retained')
        plt.title('Variance retained vs r value for Truncated SVD')
        plt.show()

        r = self.r



    def a(self):
        # count the documents in each category
        print_question('a')
        count = defaultdict(int)
        for d in self.train_data.target:
            count[d] += 1

        colors = rand_color_arr(8)
        plt.barh(self.train_data.target_names, count.values(), alpha=0.8, color=colors)
        plt.xlabel('Class')
        plt.ylabel('Number')
        plt.title('the Number of Documents per Class')
        plt.show()

    def b(self):
        print_question('b')
        vectorizer_2 = CountVectorizer(analyzer='word', stop_words=stop_words, min_df=2, tokenizer=stemTokenizer)
        vectors_2 = vectorizer_2.fit_transform(self.train_data.data)
        print "terms num when mid_df = 2: %d" % vectors_2.shape[1]
        print "terms num when mid_df = 5: %d" % self.vectors.shape[1]

    def c(self):
        print_question('c')
        allDoc = []
        for cat in allCat:
            data = fetch_data([cat], 'train').data
            poke = ""
            for doc in data:
                poke = poke + " " + doc
            allDoc.append(poke)

        vectors_full = self.to_vec(allDoc)
        tficf_train = self.to_tfidf(vectors_full)
        tficf_train_copy = tficf_train.copy()
        features = self.vectorizer.get_feature_names()
        for i in range(4):
            words = []
            for j in range(10):
                doc = tficf_train_copy[i]
                max_index = np.argmax(doc)
                words.append(features[max_index])
                tficf_train_copy[i, max_index] = 0
            print allCat[i], words

    def d(self):
        print_question('d')
        print 'SVD shape: %s' % str(self.tfidf_SVD.shape)
        print 'NMF shape: %s' % str(self.tfidf_NMF.shape)

    def e(self):
        print_question('e')
        hard_classifier = svm.LinearSVC(C=1000, random_state=42)
        soft_classifier = svm.LinearSVC(C=0.001, random_state=42)
        self.show_result(*self.svm_classify_SVD(hard_classifier, 'Hard Margin SVM'))
        self.show_result(*self.svm_classify_SVD(soft_classifier, 'Soft Margin SVM'))      
        self.show_result(*self.svm_classify_NMF(hard_classifier, 'Hard Margin SVM'))
        self.show_result(*self.svm_classify_NMF(soft_classifier, 'Soft Margin SVM'))

    def f(self):
        print_question('f')
        best_score = 0
        best_gamma = 0
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            classifier = svm.LinearSVC(C=gamma, random_state=42)
            classifier.fit(self.tfidf_SVD, self.train_labels)
            scores = (cross_val_score(classifier, self.tfidf_SVD, self.train_labels, cv=5))
            score = scores.mean()
            if score > best_score:
                best_score = score
                best_gamma = gamma

            print "Accuracy: %.5f | gamma: " % score, gamma

        print "Best Accuracy: %.5f | gamma: %d" % (best_score, best_gamma)

        classifier = svm.LinearSVC(C=best_gamma, random_state=42)
        self.show_result(*self.svm_classify_SVD(classifier, 'Best SVM'))

        best_score = 0
        best_gamma = 0
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            classifier = svm.LinearSVC(C=gamma, random_state=42)
            classifier.fit(self.tfidf_NMF, self.train_labels)
            scores = (cross_val_score(classifier, self.tfidf_NMF, self.train_labels, cv=5))
            score = scores.mean()
            if score > best_score:
                best_score = score
                best_gamma = gamma

            print "Accuracy: %.5f | gamma: " % score, gamma

        print "Best Accuracy: %.5f | gamma: %d" % (best_score, best_gamma)

        classifier = svm.LinearSVC(C=best_gamma, random_state=42)
        self.show_result(*self.svm_classify_NMF(classifier, 'Best SVM'))

    def g(self):
        print_question('g')
        nb = MultinomialNB()
        self.show_result(*self.prob_classify_NMF(nb, 'Naive Beyes Classifier'))
        self.show_result(*self.prob_classify_SVD(nb, 'Naive Beyes Classifier'))

    def h(self):
        print_question('h')
        lg = LogisticRegression()
        self.show_result(*self.prob_classify_NMF(lg, 'Logistic Regression Classifier'))
        self.show_result(*self.prob_classify_SVD(lg, 'Logistic Regression Classifier'))

    def i(self):
        print_question('i')
        params = [0.001, 0.1, 1, 10, 1000]
        penalties = ['l1', 'l2']

        for p in penalties:
            for c in params:
                lg = LogisticRegression(C=c, penalty=p)
                msg = 'Logistic Regression Classifier with c=%s, penalty=%s' % (str(c), p)
                self.show_result(*self.prob_classify_NMF(lg, msg))
                self.show_result(*self.prob_classify_SVD(lg, msg))

    def j(self):
        print_question('j')
        # build training data
        train = fetch_data(cat_4, 'train')
        vectors = self.to_vec(train.data)
        tfidf = self.to_tfidf(vectors)
        nmf = self.to_NMF(tfidf)

        # build testing data
        test = fetch_data(cat_4, 'test')
        vectors_test = self.vectorizer.transform(test.data)
        tfidf_test = self.tfidf_transformer.transform(vectors_test)
        nmf_test = self.nmf.transform(tfidf_test)

        # build classifiers
        svc = svm.LinearSVC(C=1, random_state=42)
        nb = MultinomialNB()
        ovo = OneVsOneClassifier(svc)
        ovr = OneVsRestClassifier(svc)

        # train and test
        self.multi_classify(nb, nmf, nmf_test, train.target, test.target, 'naive bayes')
        self.multi_classify(ovo, nmf, nmf_test, train.target, test.target, 'one vs one')      
        self.multi_classify(ovr, nmf, nmf_test, train.target, test.target, 'one vs rest')








