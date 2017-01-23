import codecs
import json
import numpy as np
import random
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

"""-------------------------------------------------"""
""" Importing data from external files into program """
"""-------------------------------------------------"""

"""
function which reads data from the file path 'path'
and returns a list of dictionaries with data extracted from .json
list od dictionaries is saved in the variable called dataset
"""
def read_dataset(path):
    # 'with' makes sure the file we're accessing is closed when the block ends
    with codecs.open(path, 'r', 'utf-8') as myFile:
        # content stores whole file as a single string with raw .json data
        json_content = myFile.read()
    # we save data from .json to list of dictionaries
    dataset = json.loads(json_content)
    return dataset
    
# calling the above function in __main__
if __name__ == "__main__":
    path = "./pizza_request_dataset.json"
    dataset = read_dataset(path)

##with open("./new_requests.txt","r") as f:
##    data_new=f.readlines()
##f.close()
##print (data_new)

# control print 
print ("The dataset contains",len(dataset), "samples.\n" )

"""--------------------------------------------------"""
""" Exctracting data to convenient variables that'll """
""" be easy to work with later                       """
"""--------------------------------------------------"""


"""
from dataset we extract all words of all requests in all_words variable
set is used as variable type so the same words wouldn't appear twice
we also extract all requests as request text and their outcomes in two list
variables, all_request and all_outcome accordingly 
"""
all_words=set()
all_requests=[]
all_outcomes=[]
for d in dataset:
    #request_words is a string of all words in request text
    #all_words is a set of all words in all requests so far
    request_words=d["request_text"].split()
    request_words=set(request_words)
    all_words=all_words.union(request_words)
    
    #extraction od request text and outcomes
    all_requests.append(d["request_text"])
    all_outcomes.append(d["requester_received_pizza"])
    
# we shuffle requests so the order doesn't influence our training
zipped=list(zip(all_requests,all_outcomes))
random.shuffle(zipped)
all_requests,all_outcomes=zip(*zipped)

"""--------------------------------------------------"""
""" Tokenizing text and preparing it for training,   """
""" using scikit-learn                               """
"""--------------------------------------------------"""

"""
CountVectorizer() is sklearn class from sklearn.feature_extraction.text module, used for extracting text features.
It is used to convert a collection of text documents to a matrix of token counts.
It's method fit_transform(raw_documents[,y]) learns the vocabulary dictionary and returns term-document matrix
"""
count_vect = CountVectorizer()
train_set_counts = count_vect.fit_transform(all_requests[:3970])
#control print of dimension of matrix of token counts
print(train_set_counts.shape)

"""
TfidfTrasnforemer() is another sklearn class from sklearn.feature_extraction.text module
It is used to transform a count matrix to a normalized tf or tf-idf representation
Tf stand for term-frequency, tf-idf stands for term-frequency times inverse document-frequency
Value of a word in count matrix is instead of occurrences changed for frequencies in tf-idf
representation and then it's weights are downscaled in tf-idf representation
We do this because words that occur in many documents in corpus (words as 'and', 'if', 'or', etc.)
are usually less informative than those occuring in smaller portions
"""
tfidf_transformer = TfidfTransformer()
train_set_tfidf = tfidf_transformer.fit_transform(train_set_counts)

"""--------------------------------------------------------------------------"""
""" Training classifier using multinomial variant of naive Bayes classfifier """
"""--------------------------------------------------------------------------"""

classifier = MultinomialNB().fit(train_set_tfidf, all_outcomes[:3970])

"""------------------------------------------------------------------------------"""
""" Setting a pipeline. The purpose of the pipeline is to assemble several steps """ 
""" that can be cross-validated together while setting different parameters.     """
"""------------------------------------------------------------------------------"""
Pipeline
text_classifier = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(),)
])
text_classifier = text_classifier.fit(all_requests[:3970], all_outcomes[:3970])

"""--------------------------------------------------------------------------"""
""" Testing our predictions                                                  """
"""--------------------------------------------------------------------------"""
test_set = all_requests[3970:]
predicted = text_classifier.predict(test_set)
print("NBayes success: ",np.mean(predicted == all_outcomes[3970:]))
##
##X_new_counts=count_vect.transform(data_new)
##X_new_tfidf=tfidf_transformer.transform(X_new_counts)
##predicted=classifier.predict(X_new_tfidf)
##for doc, category in zip(data_new, predicted):
##    if(doc!='\n'):
##        print('%r => %s' % (doc, category))




