import itertools
import pandas as pd
import csv
import preprocessor as p
# Read data from files
import sklearn
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import Imputer
import numpy as np
from textblob import TextBlob


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


train = pd.read_csv('data/Train5000.txt', sep='\t', quoting=csv.QUOTE_NONE,
                            names=["sentiment", "review"])

test = pd.read_csv('data/Test5000.txt', sep='\t', quoting=csv.QUOTE_NONE,
                            names=["sentiment", "review"])
train = train.dropna()
test = test.dropna()

import re
from nltk.corpus import stopwords
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
# train['review'] = train['review'].apply(preprocess_text)
# print train
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    # review_text = p.clean(review)
    # print review_text
    # review.replace('$MENTION$','')
    # review.replace('$URL$', '')
    # review.replace('$PIC$', '')
    # review.replace('$NUMBER$', '')
    # review.replace('$SMILEY$', '')
    # sw = ['$MENTION$','$URL$','$PIC$','$NUMBER$','$SMILEY$','$RESERVED$']
    # review = ' '.join(filter(lambda x: x not in sw, review.split()))

    wnl = PorterStemmer()
    review = " ".join([wnl.stem(i) for i in review.split()])

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Download the punkt tokenizer for sentence splitting
import nltk.data
# nltk.download()

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"

for review in train["review"]:
    print review
    if review != 'nan':
        sentences += review_to_sentences(review, tokenizer)

# print "Parsing sentences from unlabeled set"
# for review in unlabeled_train["review"]:
#     print review
#     if review != 'nan':
#      sentences += review_to_sentences(review, tokenizer)

# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 100    # Word vector dimensionality
min_word_count = 5   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
sg =1                 # Architecture: 0- CBOW sg=1 - skip-gram  predict the context given a word
hs =0                 # 1, hierarchical softmax , 0 (default), negative sampling

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, sg = sg, hs = hs,workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

import numpy as np  # Make sure that numpy is imported

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(penalty='l2')

print "Fitting a random forest to labeled training data..."
trainDataVecs = Imputer().fit_transform(trainDataVecs)
forest = forest.fit(trainDataVecs, train["sentiment"] )
# lr = lr.fit(trainDataVecs, train['sentiment'])
# Test & extract results
testDataVecs = Imputer().fit_transform(testDataVecs)
result = forest.predict( testDataVecs )
# lrres=lr.predict(testDataVecs)
measure_performance(testDataVecs,test["sentiment"],forest)
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

measure_performance(testDataVecs,test["sentiment"],forest)
conf = sklearn.metrics.confusion_matrix(test["sentiment"], result)
# measure_performance(testDataVecs,test["sentiment"],lr)
# conf = sklearn.metrics.confusion_matrix(test["sentiment"], lrres)
plt.figure()
plot_confusion_matrix(conf, classes=['Ad', 'Awareness', 'Research', 'News', 'Others', 'Self'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# Write the test results
output = pd.DataFrame( data={"Tweet":test["sentiment"],"Category":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.wv.index2word, idx ))

# For the first 10 clusters
for cluster in xrange(0,10):
    #
    # Print the cluster number
    print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    print words

def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["review"].size, num_clusters), \
    dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros(( test["review"].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)
measure_performance(test_centroids,test["sentiment"],forest)


conf = sklearn.metrics.confusion_matrix(test["sentiment"], result)
plt.figure()
plot_confusion_matrix(conf, classes=['Ad', 'Awareness', 'Research', 'News', 'Others', 'Self'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# Write the test results
output = pd.DataFrame(data={"Tweet":test["review"],"Category":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )




