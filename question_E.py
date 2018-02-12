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
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import main

#This File contains Question E

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



tfidf_train = TfidfTransformer().fit_transform(main.vectors)

#pply LSI to the TFxIDF matrix corresponding to the 8 classes. and pick k=50; so each document is mapped to a 50-dimensional vector. Alternatively, reduce dimensionality through Non-Negative Matrix Factorization (NMF) and compare the results of the parts e-i using both methods.
SVD = TruncatedSVD(n_components=50, random_state=42)
transformed_tfidf = SVD.fit_transform(tfidf_train)
print transformed_tfidf.shape


# Main
trainingPoints = []
testingPoints = []
test = get_graphic_test(categories)
test_counts = main.vectorizer.fit_transform(test.data)
tfidf_test = TfidfTransformer().fit_transform(test_counts)
transformed_test_tfidf = SVD.fit_transform(tfidf_test)

# Hard Classifier
h_c = svm.LinearSVC(C = 1000, dual = False, random_state = 42)

for i in get_graphic_train(categories).target:
	if i<4:	
		trainingPoints.append(0)
	else:
		trainingPoints.append(1)
# print trainingPoints
h_c.fit(transformed_tfidf, trainingPoints)

for i in get_graphic_test(categories).target:
	if i<4:	
		testingPoints.append(0)
	else:
		testingPoints.append(1)
# print testingPoints

predicted = h_c.predict(transformed_test_tfidf)
score = h_c.decision_function(transformed_test_tfidf)
accuracy = np.mean(predicted == testingPoints)
# Report results
print("Accuracy of Hard Margin SVM: " + str(accuracy))
print("-"*60)
print("Classification report: ")
print(classification_report(testingPoints, predicted, target_names=['Computer technology', 'Recreational activity']))
print("-"*60)
print("Confusion Matrix: ")
print(confusion_matrix(testingPoints, predicted))
print("-"*60)

fpr, tpr, threshold = roc_curve(testingPoints, score)
line = [0, 1]
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.title('ROC-Curve of Hard Margin SVM Classification', fontsize = 20)
plt.axis([-0.004, 1, 0, 1.006])
plt.show()
# print doc.target, len(doc.target)




