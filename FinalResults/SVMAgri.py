# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import csv

import itertools
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt


from nltk.corpus import stopwords

#\\\\\\\\\\\\\\\\\\\\\\\\\for balanced dataset use trainagri5000 for unbalanced use trainagri5000full/////////////
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from textblob import TextBlob

train = pd.read_csv('data/trainagri5000full.txt', sep='\t', quoting=csv.QUOTE_NONE,
                            names=["Category", "Tweet"])
print train
#//////////////////////////////////////Add this for POS tagging/////////////////////////////////////////////////////////////////
# def preprocess_text(message):
#     for i, row in train.iterrows():
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
# train['Tweet'] = train['Tweet'].apply(preprocess_text)
# print train
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def review_to_words(raw_text):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw ""), and
    # the output is a single string (a preprocessed "")
    #
    # 1. Remove HTML
    # review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    print stopwords.words("english")
    words = [w for w in words if not w in stopwords.words("english")]
    print words
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

# Get the number of reviews based on the dataframe column size
num_reviews = train["Tweet"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the "" list
for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["Tweet"][i] ) )

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
# vectorizer = CountVectorizer(ngram_range=(1, 3),
#                                    token_pattern=r'\b\w+\b', min_df=1)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag

print "Training the random forest..."

param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

pipeline_svm = Pipeline([

    ('classifier', SVC()),  # <== change here
])
grid_svm = GridSearchCV(
    pipeline_svm,
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(train["Category"], n_folds=5),  # what type of cross validation to use
)

svm_detector = grid_svm.fit(train_data_features, train["Category"])


#\\\\\\\\\\\\\\\\\\\\\change file name here too as said before \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
test = pd.read_csv('data/testagri5000full.txt', sep='\t', quoting=csv.QUOTE_NONE,
                            names=["Category", "Tweet"])
# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["Tweet"])
clean_test_reviews = []

print "Cleaning and parsing the test set ""s...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["Tweet"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions

nbresult = svm_detector.predict(test_data_features)
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

measure_performance(test_data_features,test["Category"],svm_detector)

conf = sklearn.metrics.confusion_matrix(test["Category"], nbresult)
plot_confusion_matrix(conf, classes=['NonAgri Tweet','Agri Tweet'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

output = pd.DataFrame( data={"Tweet":test["Tweet"],"Category":nbresult} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False )