# gensim modules
import itertools
import sklearn
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec

# numpy
import numpy

# random
from random import shuffle

# classifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

sources = {'data/AdTest5000.txt':'TEST_AD', 'data/AwarenessTest5000.txt':'TEST_AWARENESS', 'data/NewsTest5000.txt':'TEST_NEWS',
           'data/OthersTest5000.txt':'TEST_OTHER','data/ResearchTest5000.txt':'TEST_RESEARCH','data/SelfTest5000.txt':'TEST_SELF',
           'data/AdTrain5000.txt':'TRAIN_AD', 'data/AwarenessTrain5000.txt':'TRAIN_AWARENESS', 'data/NewsTrain5000.txt':'TRAIN_NEWS',
           'data/OthersTrain5000.txt':'TRAIN_OTHER','data/ResearchTrain5000.txt':'TRAIN_RESEARCH','data/SelfTrain5000.txt':'TRAIN_SELF'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(dm = 0,min_count=10, window=10, size=100, sample=1e-4, negative=5, workers=8, dbow_words=1)

model.build_vocab(sentences.to_array())

for epoch in range(50):
    model.train(sentences.sentences_perm())

model.save('./imdb.d2v')
adtestsize=sum(1 for line in open('data/AdTest5000.txt'))
awarenesstestsize=sum(1 for line in open('data/AwarenessTest5000.txt'))
newstestsize=sum(1 for line in open('data/NewsTest5000.txt'))
otherstestsize=sum(1 for line in open('data/OthersTest5000.txt'))
researchtestsize=sum(1 for line in open('data/ResearchTest5000.txt'))
selftestsize=sum(1 for line in open('data/SelfTest5000.txt'))
adtrainsize=sum(1 for line in open('data/AdTrain5000.txt'))
awarenesstrainsize=sum(1 for line in open('data/AwarenessTrain5000.txt'))
newstrainsize=sum(1 for line in open('data/NewsTrain5000.txt'))
otherstrainsize=sum(1 for line in open('data/OthersTrain5000.txt'))
researchtrainsize=sum(1 for line in open('data/ResearchTrain5000.txt'))
selftrainsize=sum(1 for line in open('data/SelfTrain5000.txt'))
totaltrainsize= adtrainsize+awarenesstrainsize+newstrainsize+otherstrainsize+researchtrainsize+selftrainsize
train_array = numpy.zeros((totaltrainsize,100))
train_labels = numpy.zeros(totaltrainsize)

for i in range(adtrainsize):
    print str(i)
    prefix_train_ad = 'TRAIN_AD_' + str(i)
    print prefix_train_ad
    train_array[i] = model.docvecs[prefix_train_ad]
    train_labels[i] = 0


for i in range(awarenesstrainsize):
    print str(i)
    prefix_train_awareness = 'TRAIN_AWARENESS_' + str(i)
    print prefix_train_awareness
    train_array[adtrainsize+i] = model.docvecs[prefix_train_awareness]
    train_labels[adtrainsize+i] = 1

for i in range(newstrainsize):
    print str(i)
    prefix_train_news = 'TRAIN_NEWS_' + str(i)
    train_array[adtrainsize+awarenesstrainsize+i] = model.docvecs[prefix_train_news]
    train_labels[adtrainsize+awarenesstrainsize+i] = 2

for i in range(otherstrainsize):
    print str(i)
    prefix_train_other = 'TRAIN_OTHER_' + str(i)
    train_array[adtrainsize+awarenesstrainsize+newstrainsize+i] = model.docvecs[prefix_train_other]
    train_labels[adtrainsize+awarenesstrainsize+newstrainsize+i] = 3

for i in range(researchtrainsize):
    print str(i)
    prefix_train_research = 'TRAIN_RESEARCH_' + str(i)
    train_array[adtrainsize+awarenesstrainsize+newstrainsize+otherstrainsize+i] = model.docvecs[prefix_train_research]
    train_labels[adtrainsize+awarenesstrainsize+newstrainsize+otherstrainsize+i] = 4

for i in range(selftrainsize):
    print str(i)
    prefix_train_self = 'TRAIN_SELF_' + str(i)
    train_array[adtrainsize+awarenesstrainsize+newstrainsize+otherstrainsize+researchtrainsize+i] = model.docvecs[prefix_train_self]
    train_labels[adtrainsize+awarenesstrainsize+newstrainsize+otherstrainsize+researchtrainsize+i] = 5

print train_array
totaltestsize=adtestsize+awarenesstestsize+newstestsize+otherstestsize+researchtestsize+selftestsize
test_array = numpy.zeros((totaltestsize, 100))
test_labels = numpy.zeros(totaltestsize)

for i in range(adtestsize):
    print "entered 1"
    prefix_test_ad = 'TEST_AD_' + str(i)
    test_array[i] = model.docvecs[prefix_test_ad]
    test_labels[i] = 0

for i in range(awarenesstestsize):
    print "entered 2"
    prefix_test_awareness = 'TEST_AWARENESS_' + str(i)
    test_array[adtestsize + i] = model.docvecs[prefix_test_awareness]
    test_labels[adtestsize + i] = 1

for i in range(newstestsize):
    print "entered 3"
    prefix_test_news = 'TEST_NEWS_' + str(i)
    test_array[adtestsize + awarenesstestsize + i] = model.docvecs[prefix_test_news]
    test_labels[adtestsize + awarenesstestsize + i] = 2

for i in range(otherstestsize):
    print "entered 4"
    prefix_test_other = 'TEST_OTHER_' + str(i)
    test_array[adtestsize + awarenesstestsize + newstestsize + i] = model.docvecs[prefix_test_other]
    test_labels[adtestsize + awarenesstestsize + newstestsize + i] = 3

for i in range(researchtestsize):
    print "entered 5"
    prefix_test_research = 'TEST_RESEARCH_' + str(i)
    test_array[adtestsize + awarenesstestsize + newstestsize + otherstestsize + i] = model.docvecs[prefix_test_research]
    test_labels[adtestsize + awarenesstestsize + newstestsize + otherstestsize + i] = 4

for i in range(selftestsize):
    print "entered 6"
    prefix_test_self = 'TEST_SELF_' + str(i)
    test_array[adtestsize + awarenesstestsize + newstestsize + otherstestsize + researchtestsize + i] = model.docvecs[prefix_test_self]
    test_labels[adtestsize + awarenesstestsize + newstestsize + otherstestsize + researchtestsize + i] = 5

print test_array

classifier = SVC()
classifier.fit(train_array, train_labels)
predictions = classifier.predict(test_array)
classifier.score(test_array, test_labels)

print classifier.score(test_array, test_labels)
print classifier.score(train_array,train_labels)




def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred = clf.predict(X)
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(sklearn.metrics.accuracy_score(y, y_pred)), "\n")

    if show_classification_report:
        print ("Classification report")
        print (sklearn.metrics.classification_report(y, y_pred))

    if show_confusion_matrix:
        print ("Confusion matrix")
        print (sklearn.metrics.confusion_matrix(y, y_pred), "\n")

measure_performance(test_array,test_labels,classifier)


import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap='coolwarm')
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


conf = sklearn.metrics.confusion_matrix(test_labels, predictions)
# measure_performance(testDataVecs,test["sentiment"],lr)
# conf = sklearn.metrics.confusion_matrix(test["sentiment"], lrres)
plt.figure()
plot_confusion_matrix(conf, classes=['Ad', 'Awareness', 'Research', 'News', 'Others', 'Self'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()