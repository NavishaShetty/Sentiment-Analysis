# -*- coding: utf-8 -*-

"""
Environment : python 2 (Spyder)
Libraries : numpy, pandas, sklearn, matplotlib, nltk, subprocess

"""

%matplotlib inline

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import matplotlib.pyplot as plt

from subprocess import check_output

dataset = pd.read_csv('Sentiment.csv')

#Keeping only the unneccessary columns
dataset = dataset[['text','sentiment']]

# Splitting the dataset into train and test set
training_set, testing_set = train_test_split(dataset,test_size = 0.2)
# Removing neutral sentiments
training_set = training_set[training_set.sentiment != "Neutral"]

train_positive = training_set[ training_set['sentiment'] == 'Positive']
train_positive = train_positive['text']
train_negative = training_set[ training_set['sentiment'] == 'Negative']
train_negative = train_negative['text']

tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in training_set.iterrows():
    wordsfiltered = [e.lower() for e in row.text.split() if len(e) >= 3]
    wordscleaned = [word for word in wordsfiltered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in wordscleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))

test_positive = testing_set[ testing_set['sentiment'] == 'Positive']
test_positive = test_positive['text']
test_negative = testing_set[ testing_set['sentiment'] == 'Negative']
test_negative = test_negative['text']

# Extracting word features
    def get_words_in_tweets(tweets):
        all = []
        for (words, sentiment) in tweets:
            all.extend(words)
            return all

    def get_word_features(wordlist):
        wordlist = nltk.FreqDist(wordlist)
        features = wordlist.keys()
        return features
wfeatures = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# Training the Naive Bayes classifier
training_set = nltk.classify.apply_features(extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

negative_word_count = 0
positive_word_count = 0
for obj in test_negative: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        negative_word_count = negative_word_count + 1
for obj in test_positive: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        positive_word_count = positive_word_count + 1
        
print('[Negative word count]: %s/%s ' % (len(test_negative),negative_word_count))        
print('[Positive word count]: %s/%s ' % (len(test_positive),positive_word_count))    