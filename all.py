# A
import random
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
import matplotlib.pyplot as plt
# B
import re
import string
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
# C
import numpy as np
# D
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
# E
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
# F
from sklearn.model_selection import cross_val_score
# G
from sklearn.linear_model import LogisticRegression


# ------------------------------------------- #
# -------------------- A -------------------- #
# ------------------------------------------- #
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


def fetch_data(category, subset):
    assert(subset in {'train', 'test'})
    return fetch_20newsgroups(
        subset=subset,
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
train_data = fetch_data(categories, 'train')

'''
# count the documents in each category
count = defaultdict(int)
for d in train_data.target:
    count[d] += 1

# plot the result
# *****
fig = plt.figure()
ax = fig.add_subplot(111)
colors = rand_color_arr(8)
plt.barh(train_data.target_names, count.values(), alpha=0.8, color=colors)
plt.xlabel('Class')
plt.ylabel('Number')
plt.title('the Number of Documents per Class')
plt.show()
'''

# ------------------------------------------- #
# -------------------- B -------------------- #
# ------------------------------------------- #
def stemTokenizer(para):
    stemmer = SnowballStemmer("english")
    temp = "".join([i if ord(i) < 128 else ' ' for i in para])  # remove non-ascii
    temp = re.sub('[,.-:/()?><{}*$#&]', '', temp)  # remove some special punc
    temp = "".join(c for c in temp if c not in string.punctuation)
    return [stemmer.stem(item) for item in temp.split()]


stop_words = text.ENGLISH_STOP_WORDS

# vectorizer_2 = CountVectorizer(analyzer='word', stop_words=stop_words, min_df=2, tokenizer=stemTokenizer)
vectorizer_5 = CountVectorizer(analyzer='word', stop_words=stop_words, min_df=5, tokenizer=stemTokenizer)
# vectors_2 = vectorizer_2.fit_transform(train_data.data)
vectors_5 = vectorizer_5.fit_transform(train_data.data)
# print "terms num when mid_df = 2: %d" % vectors_2.shape[1]
# print "terms num when mid_df = 5: %d" % vectors_5.shape[1]

tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(vectors_5)
# print tfid.shape

vectorizer = vectorizer_5
vectors = vectors_5

# ------------------------------------------- #
# -------------------- C -------------------- #
# ------------------------------------------- #
# allCat = [
#     'comp.sys.ibm.pc.hardware',
#     'comp.sys.mac.hardware',
#     'misc.forsale',
#     'soc.religion.christian',
#     'comp.graphics',
#     'comp.os.ms-windows.misc',
#     'comp.windows.x',
#     'rec.autos',
#     'rec.motorcycles',
#     'rec.sport.baseball',
#     'rec.sport.hockey',
#     'alt.atheism',
#     'sci.crypt',
#     'sci.electronics',
#     'sci.med',
#     'sci.space',
#     'talk.politics.guns',
#     'talk.politics.mideast',
#     'talk.politics.misc',
#     'talk.religion.misc'
# ]
# allDoc = []
# for cat in allCat:
#     data = fetch_data([cat], 'train').data
#     poke = ""
#     for doc in data:
#         poke = poke + " " + doc
#     allDoc.append(poke)

# vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words, min_df=2, tokenizer=stemTokenizer)
# vectors_full = vectorizer.fit_transform(allDoc)
# print vectors_full.shape

# tficf_train = tfidf_transformer.fit_transform(vectors_full)
# tficf_train_copy = tficf_train.copy()
# features = vectorizer.get_feature_names()
# for i in range(4):
#     words = []
#     for j in range(10):
#         doc = tficf_train_copy[i]
#         max_index = np.argmax(doc)
#         words.append(features[max_index])
#         tficf_train_copy[i, max_index] = 0
#     print allCat[i], words


# ------------------------------------------- #
# -------------------- D -------------------- #
# ------------------------------------------- #
SVD = TruncatedSVD(n_components=50, random_state=42)
tfidf_SVD = SVD.fit_transform(tfidf)
# print tfidf_SVD.shape

NMF = NMF(n_components=50, random_state=42)
tfidf_NMF = NMF.fit_transform(tfidf)
# print tfidf_NMF.shape


# ------------------------------------------- #
# -------------------- E -------------------- #
# ------------------------------------------- #
def build_labels(data):
    labels = []
    for i in data.target:
        labels.append(1) if i >= 4 else labels.append(0)
    return labels


def show_result(score, acc, report, matrix):
    print "Accuracy: %.2f" % acc
    print "Classification Report:"
    print report
    print "Confusion Matrix:"
    print matrix

    fpr, tpr, _ = roc_curve(test_labels, score)
    plt.plot([0, 1], [0, 1])
    plt.plot(fpr, tpr)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.title('ROC Curve')
    plt.axis([0, 1, 0, 1])
    plt.show()


def svm_classify(classifier, tfidf_SVD, tfidf_SVD_test, name):
    print '============= %s =============' % name

    # train model
    classifier.fit(tfidf_SVD, train_labels)

    # make prediction
    prediction = classifier.predict(tfidf_SVD_test)
    score = classifier.decision_function(tfidf_SVD_test)
    acc = np.mean(prediction == test_labels)
    report = classification_report(test_labels, prediction, target_names=['Computer technology', 'Recreational activity'])
    matrix = confusion_matrix(test_labels, prediction)

    return score, acc, report, matrix


# fetch training and testing labels
train_data = fetch_data(categories, 'train')
test_data = fetch_data(categories, 'test')

# build training and testing labels
train_labels = build_labels(train_data)
test_labels = build_labels(test_data)

# build testing data
test_vectors = vectorizer.fit_transform(test_data.data)
test_tfidf = tfidf_transformer.fit_transform(test_vectors)
tfidf_SVD_test = SVD.fit_transform(test_tfidf)
tfidf_NMF_test = NMF.fit_transform(test_tfidf)

# build classifiers
# hard_classifier = svm.LinearSVC(C=1000, random_state=42)
# soft_classifier = svm.LinearSVC(C=0.001, random_state=42)
# show_result(*svm_classify(hard_classifier, tfidf_SVD, tfidf_SVD_test, 'Hard Margin SVM'))
# show_result(*svm_classify(soft_classifier, tfidf_SVD, tfidf_SVD_test, 'Soft Margin SVM'))

# ------------------------------------------- #
# -------------------- F -------------------- #
# ------------------------------------------- #
# best_score = 0
# best_gamma = 0
# for gamma in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
#     classifier = svm.LinearSVC(C=gamma, random_state=42)
#     classifier.fit(tfidf_SVD, train_labels)
#     scores = (cross_val_score(classifier, tfidf_SVD, train_labels, cv=5))
#     score = scores.mean()
#     if score > best_score:
#         best_score = score
#         best_gamma = gamma

#     print "Accuracy: %.5f | gamma: " % score, gamma

# print "Best Accuracy: %.5f | gamma: %d" % (best_score, best_gamma)

# classifier = svm.LinearSVC(C=best_gamma, random_state=42)
# show_result(*svm_classify(classifier, tfidf_SVD, tfidf_SVD_test, 'Best SVM'))

# ------------------------------------------- #
# -------------------- G -------------------- #
# ------------------------------------------- #
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

def prob_classify(classifier, tfidf_SVD, tfidf_SVD_test, name):
    print '============= %s =============' % name

    # train model
    classifier.fit(tfidf_SVD, train_labels)

    # make prediction
    prediction = classifier.predict(tfidf_SVD_test)
    prob = classifier.predict_proba(tfidf_SVD_test[:])[:, 1]
    acc = np.mean(prediction == test_labels)
    report = classification_report(test_labels, prediction, target_names=['Computer technology', 'Recreational activity'])
    matrix = confusion_matrix(test_labels, prediction)

    return prob, acc, report, matrix


min_max_scaler = MinMaxScaler()
scaled_tfidf = min_max_scaler.fit_transform(tfidf_SVD)
scaled_test_tfidf = min_max_scaler.fit_transform(tfidf_SVD_test)

# nb.fit(scaled_tfidf, train_labels)
# prediction = nb.predict(scaled_test_tfidf)
# acc = np.mean(prediction == test_labels)
# prob = nb.predict_proba(scaled_test_tfidf[:])[:, 1]

# nb = MultinomialNB()
# show_result(*prob_classify(nb, scaled_tfidf, scaled_test_tfidf, 'Naive Beyes Classifier'))

# ------------------------------------------- #
# -------------------- H -------------------- #
# ------------------------------------------- #
# lg = LogisticRegression()
# show_result(*prob_classify(lg, scaled_tfidf, scaled_test_tfidf, 'Logistic Regression Classifier'))

# ------------------------------------------- #
# -------------------- I -------------------- #
# ------------------------------------------- #
params = [0.001, 0.1, 1, 10, 1000]
penalties = ['l1', 'l2']

for p in penalties:
    for c in params:
        lg = LogisticRegression(C=c, penalty=p)
        msg = 'Logistic Regression Classifier with c=%s, penalty=%s' % (str(c), p)
        show_result(*prob_classify(lg, scaled_tfidf, scaled_test_tfidf, msg))








