import itertools
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.learning_curve import learning_curve
import preprocessor as p
import re
import sys
import nltk

reload(sys)
sys.setdefaultencoding("utf-8")
# messages = [line.rstrip() for line in open('td.txt')]
# print(len(messages))
dataframe = pandas.read_csv('data/CombinedBookCategory.txt', sep='\t', quoting=csv.QUOTE_NONE,encoding='utf8',
                            names=["label","message"])
# dataframe = dataframe.drop(dataframe[dataframe.label == 'others'].index)
# dataframe = dataframe.drop(dataframe[dataframe.label == 'news'].index)
# dataframe = dataframe.drop(dataframe[dataframe.label == 'Ad'].index)
# dataframe = dataframe.drop(dataframe[dataframe.label == 'Research'].index)
# dataframe['label']=dataframe['label'].str.replace('Ad','Awareness')
# dataframe['label']=dataframe['label'].str.replace('news','self')
# dataframe['label']=dataframe['label'].str.replace('Research','Awareness')
# dataframe['label']=dataframe['label'].str.replace('others','Awareness')
# dataframe('label').replace({'news':'self'}, regex =True)
print(dataframe.groupby('label').describe())
# dataframe = dataframe[(dataframe.label == 'Awareness')|(dataframe.label == 'self')|(dataframe.label == 'others')|(dataframe.label == 'Research')]
print len(dataframe.index)
# print messages
# print(messages.groupby('label').describe())
#//////////////////////////////////////Add this for POS tagging/////////////////////////////////////////////////////////////////
# def preprocess_text(message):
#     for i, row in dataframe.iterrows():
#
#         # print row['message']
#         # clean_tweet = re.match('(.*?)http.*?\s?(.*?)', message)
#
#
#             # message = p.tokenize(message)
#             print message
#             pattern = re.compile("[^\w? ]")
#             message = re.sub(ur"[^\w\d?\s]+",'',message)
#             print message
#
#             message = ' '.join(word for word in message.split() if len(word) < 15)
#             # tagged_texts = pos_tag(message.split())
#             # words, tags = zip(*tagged_texts)
#             # pos =  ' '.join(tags)
#             # message = ' '.join([message,pos])
#             blob = TextBlob(message)
#             s = " "
#             for word,pos in blob.tags:
#              s += word + " "
#              s += pos + ' '
#
#             # dataframe['POS'] = tags
#             # message = message.replace('$PIC$', '$PIC$ ')
#             # message = message.replace('$NUMBER$', '$NUMBER$ ')
#             # message = message.replace('$URL$', '$URL$ ')
#             # if message.__contains__('$PIC$')|message.__contains__('$URL'):
#             #    message = message.rsplit('$',1)[0]+"$"
#             # message = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', message)
#
#             # print message
#             print s
#             return s
# dataframe['message'] = dataframe['message'].apply(preprocess_text)
# print dataframe
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def preprocess_text(message):
    for i, row in dataframe.iterrows():

        # print row['message']
        # clean_tweet = re.match('(.*?)http.*?\s?(.*?)', message)


            message = p.tokenize(message)
            # pattern = re.compile("[^\w$ ]")
            # message = pattern.sub('', message)
            # message = re.sub('[0-9]+', '', message)
            message = message.replace('$PIC$', '$PIC$ ')
            message = message.replace('$NUMBER$', '$NUMBER$ ')
            message = message.replace('$URL$', '$URL$ ')
            if message.__contains__('$PIC$')|message.__contains__('$URL'):
               message = message.rsplit('$',1)[0]+"$"
            # message = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', message)
            print message
            return message


dataframe['message'] = dataframe['message'].apply(preprocess_text)
print dataframe

def split_into_tokens(message):
    message = message.decode().encode('utf-8')  # convert bytes into proper unicode
    return TextBlob(message).words

def split_into_lemmas(message):
    message = message.encode("utf-8").lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

def pos(message):
    text = nltk.word_tokenize(message)
    return nltk.pos_tag(text)


msg_train, msg_test, label_train, label_test = \
    train_test_split(dataframe['message'], dataframe['label'], test_size=0.2)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

pipeline_svm = Pipeline([
    # ('bow', CountVectorizer(ngram_range=(1, 4), token_pattern=r'\b\w+\b', min_df=1)),
    ('bow', TfidfVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])


scores = cross_val_score(pipeline_svm,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         )
print scores

print scores.mean(), scores.std()


plt = plot_learning_curve(pipeline_svm, "accuracy vs. training set size", msg_train, label_train, cv=10)
plt.show()
# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
print svm_detector.grid_scores_

print confusion_matrix(label_test, svm_detector.predict(msg_test))
print classification_report(label_test, svm_detector.predict(msg_test))
conf1 = sklearn.metrics.confusion_matrix(label_test, svm_detector.predict(msg_test))
print('Best C:',svm_detector.best_estimator_.C)
print('Best Kernel:',svm_detector.best_estimator_.kernel)
print('Best Gamma:',svm_detector.best_estimator_.gamma)

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
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap='coolwarm')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
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

measure_performance(msg_test,label_test,svm_detector)
conf = sklearn.metrics.confusion_matrix(label_test, svm_detector.predict(msg_test))
plt.figure()
plot_confusion_matrix(conf, classes=['Ad', 'Awareness', 'Research', 'News', 'Others', 'Self'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()