# -*- coding: utf-8 -*-
#!/usr/bin/python
import codecs
import json
import nltk
import numpy as np
import random
import re

"""-------------------------"""
""" importing external data """
"""-------------------------"""

"""
function which reads data from the file path 'path'
and returns a list of dictionaries with data extracted from .json
"""
def read_dataset(path):
    with codecs.open(path, 'r', 'utf-8') as myFile:
        json_content = myFile.read()
    dataset = json.loads(json_content)
    return dataset

"""
opens the file in 'path', reads until EOF using readline() 
and returns a list containing the words from narratives
"""
def read_narratives(path):
    with open(path,"r") as f:
        narratives=f.readlines()
    return narratives

# calling above functions in __main__
if __name__ == '__main__':
    path = './pizza_request_dataset.json'
    dataset = read_dataset(path)
    keywords=[]
    path ="./narratives/job.txt"
    keywords+=read_narratives(path)
    path ="./narratives/desire.txt"
    keywords+=read_narratives(path)
    path ="./narratives/money.txt"
    keywords+=read_narratives(path)
    path ="./narratives/family.txt"
    keywords+=read_narratives(path)
    path ="./narratives/student.txt"
    keywords+=read_narratives(path)
    # we need to eliminate last symbol '\n' in every word
    keywords[:] = [word.rstrip('\n') for word in keywords]

keywords = set(keywords)


"""
returns separeate words containing A-Z, a-z from text
"""
def get_words(text):
    return re.compile('[A-Za-z]+').findall(text)
    

# takes a list of all the attributes and sort them by keys
all_attributes = sorted(dataset[0].keys())

# since not all of the attributes from original file may be needed
# a new list is declared to hold the ones we'll use
used_attributes=list(all_attributes)

# for readability, all of the attributes we want eliminated are listed here
# 'comment out' ones we want to remain, rest are eliminated
eliminated_attributes = ["giver_username_if_known",
                         "in_test_set",
#                         "number_of_downvotes_of_request_at_retrieval",
#                         "number_of_upvotes_of_request_at_retrieval",
                         "post_was_edited",
                         "request_id",
#                         "request_number_of_comments_at_retrieval",
                         "request_text",
                         "request_text_edit_aware",
                         "request_title",
#                         "requester_account_age_in_days_at_request",
                         "requester_account_age_in_days_at_retrieval",
#                         "requester_days_since_first_post_on_raop_at_request",
                         "requester_days_since_first_post_on_raop_at_retrieval",
#                         "requester_number_of_comments_at_request",
                         "requester_number_of_comments_at_retrieval",
#                         "requester_number_of_comments_in_raop_at_request",
                         "requester_number_of_comments_in_raop_at_retrieval",
#                        "requester_number_of_posts_at_request",
                         "requester_number_of_posts_at_retrieval",
#                         "requester_number_of_posts_on_raop_at_request",
                         "requester_number_of_posts_on_raop_at_retrieval",
                         "requester_subreddits_at_request",
                         "requester_received_pizza",
#                         "requester_upvotes_minus_downvotes_at_request",
                         "requester_upvotes_minus_downvotes_at_retrieval",
#                         "requester_upvotes_plus_downvotes_at_request",
                         "requester_upvotes_plus_downvotes_at_retrieval",
                         "requester_user_flair",
                         "requester_username",
                         "unix_timestamp_of_request",
                         "unix_timestamp_of_request_utc"
                         ]

# 'eliminination' of above mentioned attributes
for att in eliminated_attributes:
    used_attributes.remove(att)


"""--------------------"""
""" feature extraction """
"""--------------------"""

"""
this is a feature extractor function
it returns a dictionary containing relevant info about a given request
based on given attributes
"""
def request_features(request, attributes):
    all_words=set(get_words(request["request_text"]) + get_words(request["request_title"]))
    features={}
    if attributes:
        for att in attributes:
            features[att] = json.dumps(request[att], sort_keys=True, indent=2)
    topic_features=list(keywords)
    for word in topic_features:
        features["contains({})".format(word)]=(word in all_words)
    return features


"""
generates featuresets for given list of requests and attributes 
using above function on each request
"""
def featuresets_generator(requests, attributes):
    return ([(request_features(d,attributes),outcome) for (d,outcome) in requests])


"""--------------------------"""
""" creating train/test sets """
"""--------------------------"""

"""                        
2 lists of all requests split by their Boolean outcome 
list element is a tuple (dictionary, Boolean) of request and its outcome
we need this to have a representative sample
"""
positive_requests = []
negative_requests = []
for d in dataset:
    if(d["requester_received_pizza"]):
        positive_requests.append((d,True))
    elif(not d["requester_received_pizza"]):
        negative_requests.append((d,False))

random.shuffle(positive_requests)
random.shuffle(negative_requests)


"""
two representative sets are created by keeping same ratio of 
positive vs negative requests in both of them
factor of 0.7 is used to split data into train and test sets
"""
test_ratio = 0.7

positive_index = int(test_ratio * len(positive_requests))
negative_index = int(test_ratio * len(negative_requests))

representative_train = positive_requests[:positive_index] + negative_requests[:negative_index]
representative_test = positive_requests[positive_index:] + negative_requests[negative_index:]

split_index = int(len(representative_train))

random.shuffle(representative_train)
random.shuffle(representative_test)

labeled_requests = representative_train + representative_test


"""
categorizing numerical values based on their deviation from average in distribution
distribution is calculated for all values of each attribute separately
"""
for att in used_attributes:
    for request in labeled_requests:
        values = []
        values.append(request[0][att])
    avg = np.average(values)
    std = np.std(values)
    for request in labeled_requests:
        if request[0][att] > (avg + 0.5*std):
            request[0][att] = 'high'
        elif request[0][att] < (avg - 0.5*std):
            request[0][att] = 'low'
        else: 
            request[0][att] = 'average'
      

"""
generating featuresets, train and test sets based on 3 selections of attributes
    1)all attributes
    2)attributes based on paper research
    3)only request words as attributes
"""
#1
featuresets_all= featuresets_generator(labeled_requests,all_attributes)
train_set_all, test_set_all = featuresets_all[:split_index], featuresets_all[split_index:]
#2
featuresets_chosen = featuresets_generator(labeled_requests, used_attributes)
train_set_chosen, test_set_chosen = featuresets_chosen[:split_index], featuresets_chosen[split_index:]
#3
no_attributes=[]
featuresets_topic=featuresets_generator(labeled_requests,no_attributes)
train_set_topic, test_set_topic = featuresets_topic[:split_index], featuresets_topic[split_index:]    


"""------------------"""
""" machine learning """
"""------------------"""

"""
declaring classifier as Naive Bayes from nltk library
method show_most_informative_features(n) may be used to print n most informative
features of a classifier, as used in documentation examples
"""
#1
classifier = nltk.NaiveBayesClassifier.train(train_set_all)
print("NLTK NaiveBayesAllAttributes: ",nltk.classify.accuracy(classifier,test_set_all))

#2
classifier = nltk.NaiveBayesClassifier.train(train_set_chosen)
print("NLTK NaiveBayesSomeAttributes: ",nltk.classify.accuracy(classifier,test_set_chosen))

#3
classifier = nltk.NaiveBayesClassifier.train(train_set_topic)
print("NLTK NaiveBayesTextAnalysis: ",nltk.classify.accuracy(classifier,test_set_topic))

