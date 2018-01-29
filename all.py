import random
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
import matplotlib.pyplot as plt


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
import re
import string
from sklearn.feature_extraction import text
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer


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
tfid = tfidf_transformer.fit_transform(vectors_5)
# print tfid.shape

vectorizer = vectorizer_5
vectors = vectors_5

# ------------------------------------------- #
# -------------------- C -------------------- #
# ------------------------------------------- #
import numpy as np


# def argmax_N(arr, n):
#     return np.argpartition(arr, -n)[-n:]


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
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

# #pply LSI to the TFxIDF matrix corresponding to the 8 classes. and pick k=50; so each document is mapped to a 50-dimensional vector. Alternatively, reduce dimensionality through Non-Negative Matrix Factorization (NMF) and compare the results of the parts e-i using both methods.
SVD = TruncatedSVD(n_components=50, random_state=42)
tfidf_SVD = SVD.fit_transform(tfid)
# print tfidf_SVD.shape

# trainNMF = NMF(n_components=50, init='random', random_state=42)
# tfid_NMF = trainNMF.fit_transform(vectors)
# print tfid_NMF.shape


# ------------------------------------------- #
# -------------------- E -------------------- #
# ------------------------------------------- #
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve


def svm_classify(classifier, name):
    print '============= %s =============' % name
    train_data = fetch_data(categories, 'train')
    test_data = fetch_data(categories, 'test')

    # build training and testing lables
    train_labels = []
    for i in train_data.target:
        if i >= 4:
            train_labels.append(1)
        else:
            train_labels.append(0)

    test_labels = []
    for i in test_data.target:
        if i >= 4:
            test_labels.append(1)
        else:
            test_labels.append(0)

    # build testing data
    test_vectors = vectorizer.fit_transform(test_data.data)
    test_tfidf = tfidf_transformer.fit_transform(test_vectors)
    test_tfidf_SVD = SVD.fit_transform(test_tfidf)

    classifier.fit(tfidf_SVD, train_labels)

    # make prediction
    prediction = classifier.predict(test_tfidf_SVD)
    score = classifier.decision_function(test_tfidf_SVD)
    acc = np.mean(prediction == test_labels)

    # print results
    print "Accuracy: %.2f" % acc
    print "Classification Report:"
    print classification_report(test_labels, prediction, target_names=['Computer technology', 'Recreational activity'])
    print "Confusion Matrix:"
    print confusion_matrix(test_labels, prediction)

    fpr, tpr, _ = roc_curve(test_labels, score)
    # plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1])
    plt.plot(fpr, tpr)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.title('ROC Curve')
    plt.axis([0, 1, 0, 1])
    plt.show()


hard_classifier = svm.LinearSVC(C=1000, dual=False, random_state=42)
soft_classifier = svm.LinearSVC(C=0.01, dual=False, random_state=42)
svm_classify(hard_classifier, 'Hard Margin SVM')
svm_classify(soft_classifier, 'Soft Margin SVM')


