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
# vectorizer_5 = CountVectorizer(analyzer='word', stop_words=stop_words, min_df=5, tokenizer=stemTokenizer)
# vectors_2 = vectorizer_2.fit_transform(train_data.data)
# vectors_5 = vectorizer_5.fit_transform(train_data.data)
# print ("terms num when mid_df = 2: %d" % vectors_2.shape[1])
# print ("terms num when mid_df = 5: %d" % vectors_5.shape[1])

tfidf_transformer = TfidfTransformer()
# tfid = tfidf_transformer.fit_transform(vectors_5)
# print tfid.shape

# ------------------------------------------- #
# -------------------- C -------------------- #
# ------------------------------------------- #
import numpy as np


def argmax_N(arr, n):
    return np.argpartition(arr, -n)[-n:]


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
allDoc = []
for cat in allCat:
    data = fetch_data([cat], 'train').data
    poke = ""
    for doc in data:
        poke = poke + " " + doc
    allDoc.append(poke)

vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words, min_df=2, tokenizer=stemTokenizer)
vectors = vectorizer.fit_transform(allDoc)
print(vectors.shape)

tficf_train = tfidf_transformer.fit_transform(vectors)
tficf_train_copy = tficf_train.copy()
features = vectorizer.get_feature_names()
for i in range(4):
    doc = tficf_train_copy[i]
    max_indexs = argmax_N(doc, 10)
    # print doc
    words = []
    for j in range(10):
        words.append(features[np.argmax(doc)])
        tficf_train_copy[i, np.argmax(doc)] = 0
    print(allCat[i], words)





# from nltk.stem import PorterStemmer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import NMF
# from sklearn import svm
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve
# tfidf_train = TfidfTransformer().fit_transform(main.vectors)

# #pply LSI to the TFxIDF matrix corresponding to the 8 classes. and pick k=50; so each document is mapped to a 50-dimensional vector. Alternatively, reduce dimensionality through Non-Negative Matrix Factorization (NMF) and compare the results of the parts e-i using both methods.
# SVD = TruncatedSVD(n_components=50, random_state=42)
# transformed_tfidf = SVD.fit_transform(tfidf_train)
# print transformed_tfidf.shape


# # Main
# trainingPoints = []
# testingPoints = []
# test = get_graphic(categories, 'test')
# test_counts = main.vectorizer.fit_transform(test.data)
# tfidf_test = TfidfTransformer().fit_transform(test_counts)
# transformed_test_tfidf = SVD.fit_transform(tfidf_test)

# # Hard Classifier
# h_c = svm.LinearSVC(C = 1000, dual = False, random_state = 42)

# for i in get_graphic(categories, 'train').target:
#     if i<4: 
#         trainingPoints.append(0)
#     else:
#         trainingPoints.append(1)
# # print trainingPoints
# h_c.fit(transformed_tfidf, trainingPoints)

# for i in get_graphic(categories, 'test').target:
#     if i<4: 
#         testingPoints.append(0)
#     else:
#         testingPoints.append(1)
# # print testingPoints

# predicted = h_c.predict(transformed_test_tfidf)
# score = h_c.decision_function(transformed_test_tfidf)
# accuracy = np.mean(predicted == testingPoints)
# # Report results
# print("Accuracy of Hard Margin SVM: " + str(accuracy))
# print("-"*60)
# print("Classification report: ")
# print(classification_report(testingPoints, predicted, target_names=['Computer technology', 'Recreational activity']))
# print("-"*60)
# print("Confusion Matrix: ")
# print(confusion_matrix(testingPoints, predicted))
# print("-"*60)

# fpr, tpr, threshold = roc_curve(testingPoints, score)
# line = [0, 1]
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr)
# plt.plot([0,1],[0,1])
# plt.ylabel('True Positive Rate', fontsize = 20)
# plt.xlabel('False Positive Rate', fontsize = 20)
# plt.title('ROC-Curve of Hard Margin SVM Classification', fontsize = 20)
# plt.axis([-0.004, 1, 0, 1.006])
# plt.show()
# # print doc.target, len(doc.target)




