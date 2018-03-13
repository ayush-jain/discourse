# -*- coding: utf-8 -*-

from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk.classify.util, nltk.metrics
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from random import choice
from numpy import array, dot, random
import re
import collections, itertools
import numpy as np
import time
import math
import sys
import argparse
import unidecode as ud
import pickle

import math
from sklearn import svm
import time
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
from nltk import ngrams


url_dict = {}
dict_url = {}
post_body = {}
res_dict = {}



          

unit_step_function = lambda x: 0 if x < 0 else 1


def remove_stopwords(l_words, lang='english'):
	l_stopwords = stopwords.words(lang)
        #l_stopwords.remove('not')
	content = [w for w in l_words if w.lower() not in l_stopwords]
	return content

def tokenize(str):
	'''Tokenizes into sentences, then strips punctuation/abbr, converts to lowercase and tokenizes words'''
	return 	[word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
			for t in sent_tokenize(str.replace("'", ""))]
				
#Stem all words with stemmer of type
def stemming(words_l, type="PorterStemmer", lang="english", encoding="utf8"):
	supported_stemmers = ["PorterStemmer", "WordNetLemmatizer"]
	if type is False or type not in supported_stemmers:
		return words_l
	else:
		l = []
		if type == "PorterStemmer":
			stemmer = PorterStemmer()
			for word in words_l:
				l.append(stemmer.stem(word).encode(encoding))
		if type == "WordNetLemmatizer": 
			wnl = WordNetLemmatizer()
			for word in words_l:
				l.append(wnl.lemmatize(word).encode(encoding))
		return l
		
def preprocess_pipeline(str, lang="english", stemmer_type="PorterStemmer", return_as_list=False, do_remove_stopwords=False):
	l = []
	words = []
	
	sentences = tokenize(str)
	for sentence in sentences:
		#sentence = sentence.decode('ascii', 'ignore')
		if do_remove_stopwords:
			words = remove_stopwords(sentence, lang)
		else:
			words = sentence
		words = stemming(words, stemmer_type)
		#if return_as_list:
		#	l.append(" ".join(words))
		#else:
		l.append(words)
	if return_as_list:
	        new = []
	        map(new.extend, l)
	        return new 
	else:
	        return l


def extract_best_bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=500, freq_filter = 3):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(words, window_size=3)
    
    if(freq_filter > 0):
        bigram_finder.apply_freq_filter(freq_filter)
    
    bigrams = bigram_finder.nbest(score_fn, n)
    #print(bigrams)
    #print("-------")
    #return bigram_finder.score_ngrams(bigram_measures.pmi)
    #return bigrams
    
    return ["_".join(bigram) for bigram in bigrams]


def compute_feature_vector(arr, features):
    size = len(features)
    count = 0
    vec = np.zeros(size)
    for each in arr:
        each = each.lower()
        if each in features:
            inx = features.index(each)
            #if(inx<800):
                #print("hehy........\n")
            #print("inx---------")
            #print(inx)
            vec[inx] += 1
            count = count+1
    #print("count---------")
    #print(float(count*(1.0))/len(arr))
    return vec
    
#Converts rows of words per (after Lemmatization) to vector per sentence     
def prepare_training_data(data, labels, features):
    train_pos = []
    train_neg = []
    
    """
    pdata = np.array(data[:4265])
    pdata_labels = np.array(labels[:4265])

    ndata = np.array(data[-4265:])
    ndata_labels = np.array(labels[-4265:])


    shuffle_indices = np.random.permutation(np.arange(len(pdata_labels)))


    #print(len(train_data))
    #print(train_data)
    pdata = pdata[shuffle_indices]
    pdata_labels = pdata_labels[shuffle_indices]

    shuffle_indices = np.random.permutation(np.arange(len(ndata_labels)))
    ndata = ndata[shuffle_indices]
    ndata_labels = ndata_labels[shuffle_indices]

    data = list(pdata) + list(ndata)
    labels = list(pdata_labels) + list(ndata_labels)
    """

    pdata = data[:3412]
    ndata = data[4265:7677]

    for each in pdata:
        each = each.decode('ascii', 'ignore')
        #print(each)
        res = preprocess_pipeline(each, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)
        #print(res)
        train_pos.append(res)
        
    for each in ndata:
        each = each.decode('ascii', 'ignore')
        #print(each)
        res = preprocess_pipeline(each, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)
        #print(res)
        train_neg.append(res)
        
    train_data = train_pos + train_neg
    
    #print(train_pos[:10])
    #print(len(train_pos))
    #print(len(train_neg))
    #print(len(train_data))
    #print("---")
    
    train_data = np.array(train_data)
    tr_labels = labels[:3412] + labels[4265:7677]
    tr_labels = np.array(tr_labels)


    shuffle_indices = np.random.permutation(np.arange(len(tr_labels)))
    #print(len(train_data))
    #print(train_data)
    train_data = train_data[shuffle_indices]
    tr_labels = tr_labels[shuffle_indices]

    tr_data = []
    for each in train_data:
        bigrams = extract_best_bigrams(each, freq_filter=0)
        each = each + bigrams
        #print(each)
        vec = compute_feature_vector(each, features)
        tr_data.append(vec)
    tr_data = np.array(tr_data)

    test_pos = []
    test_neg = []
    for each in data[3412:4265]:
        each = " ".join(each)
        each = each.decode('ascii', 'ignore')
        test_pos.append(preprocess_pipeline(each, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False))

    for each in data[-853:]:
        each = " ".join(each)
        each = each.decode('ascii', 'ignore')
        test_neg.append(preprocess_pipeline(each, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False))

    tst_data = test_pos + test_neg
    test_labels = labels[3412:4265] + labels[-853:]
    test_labels = np.array(test_labels) 

    test_data = []
    for each in tst_data:
        bigrams = extract_best_bigrams(each, freq_filter =0)
        each = each + bigrams
        vec = compute_feature_vector(each, features)
        test_data.append(vec)

    test_data = np.array(test_data)
    
    return tr_data, tr_labels, test_data, test_labels
    
#Converts each sentence to vector form based on features    
def prepare_any_data(data, features):
    processed_data = []
    for each in data:
        each = each.decode('ascii', 'ignore')
        #print(each)
        res = preprocess_pipeline(each, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)
        #print(res)
        processed_data.append(res)
        
    vec_data = []
    for each in processed_data:
        bigrams = extract_best_bigrams(each, freq_filter=0)
        each = each + bigrams
        #print(each)
        vec = compute_feature_vector(each, features)
        vec_data.append(vec)
    vec_data = np.array(vec_data)
    return vec_data

def feature_extractor(new_pos_data, new_neg_data, pos_labels, neg_labels):

    new_pos_data = " ".join(new_pos_data)
    new_pos_data = new_pos_data.decode('ascii', 'ignore')
    new_neg_data = " ".join(new_neg_data)
    new_neg_data = new_neg_data.decode('ascii', 'ignore')

    #print(len(new_data))

    res_pos = preprocess_pipeline(new_pos_data, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)
    res_neg = preprocess_pipeline(new_neg_data, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)

    words = res_pos + res_neg

    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
     
    for word in res_pos:
        word_fd[(word.lower())] += 1
        label_word_fd['pos'][(word.lower())] += 1
     
    for word in res_neg:
        word_fd[(word.lower())] += 1
        label_word_fd['neg'][(word.lower())] += 1
        
    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    print("total")
    print(total_word_count)
     
    word_scores = {}
    #print(len(word_fd))
    #print(len(label_word_fd))
     
    for word, freq in word_fd.iteritems():
        #print(word)
        #print(freq)
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
            (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
     
    best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:3000]
    
    bestwords = set([w for w, s in best])

    bigrams = extract_best_bigrams(words)
    #print(list(bestwords))
    #print(bigrams)
 
    features = list(bestwords) + bigrams
    print features

    return features
	        
def evaluate(preds, golds):
    tp, pp, cp = 0.0, 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    try:
        precision = tp / pp
        recall = tp / cp
    except ZeroDivisionError:
        return (-1, -1, -1)
    
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

'''
class Classifier(object):
    def __init__(obj):
        pass

    def train():
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def inference():
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")

class MLP(Classifier):

    def __init__(self, tr_data, tr_labels, vec_size, validation_data=[], validation_labels=[]):    
        super(MLP, self).__init__()
        self.tr_data = tr_data
        self.tr_labels = tr_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.vec_size = vec_size
        self.w = np.random.rand(self.vec_size)
        self.eta = 0.025
        self.regularization_constant = 0.005
        self.input_size = 1000
        self.hidden_size = 10
        self.output_size = 2
        self.myMLPImpl = mlp.MLPImpl(self.input_size, self.hidden_size, self.output_size)
        self.myBackProp = mlp.Backprop(self.myMLPImpl, self.eta, self.regularization_constant)


    def train(self):
        self.n=500
        self.evaluate_every = 1        
        
        self.batch_size = 15
        self.tr_size = len(self.tr_labels)
        if(self.tr_size%self.batch_size == 0):
	        steps = self.tr_size/(self.batch_size)
        else:
	        steps = (self.tr_size/(self.batch_size))+1

        for i in xrange(self.n):
            shuffle_indices = np.random.permutation(np.arange(len(self.tr_labels)))
            #print(shuffle_indices)
            tr_data = self.tr_data[shuffle_indices]
            tr_labels = self.tr_labels[shuffle_indices]
            for j in xrange(steps):
                start = self.batch_size*j
                if(j!=steps):
                    batch = tr_data[start:(start+self.batch_size)]
                    batch_labels = tr_labels[start:(start+self.batch_size)]
                else:
                    batch = tr_data[start:]
                    batch_labels = tr_labels[start:]
                labels = []
                for each in batch_labels:
                    temp = np.zeros(2)
                    temp[each]=1
                    labels.append(list(temp))

                #print("Start----------")
                self.myBackProp.iterate(batch, labels)	
                #print("End-----------\n")
		        
            
            preds = []
            if ((i%self.evaluate_every)== 0 and i!=0):
                for each in self.validation_data:
                    res = self.myMLPImpl.compute(each)
                    preds.append(np.argmax(res))
                #print("Epoch issssssssssssss " + str(i))
                
                #print(preds)
                #print(self.validation_data)
                precision, recall, f1 = evaluate(preds, self.validation_labels)
                print "MLP results in order of Precision, Recall and F1", precision, recall, f1
                #np.savetxt("W0.csv", self.myMLPImpl.WList[0], delimiter=",")
                #np.savetxt("W1.csv", self.myMLPImpl.WList[1], delimiter=",")                
                #print("----------")
                #print(self.myMLPImpl.WList[0])
                #print(self.myMLPImpl.WList[1])
                
                
    def inference(self, test_data, w0=[], w1=[]):
        preds = []
        for each in test_data:
            if(len(w1) == 0) or (len(w1) == 0):
                res = self.myMLPImpl.compute(each)
                preds.append(np.argmax(res))
                continue        
            res = self.myMLPImpl.compute_with_inputs(each, w0, w1)
            preds.append(np.argmax(res))
        return np.array(preds)  


class Perceptron(Classifier):
    """
    Implement your Perceptron here

    """
    def __init__(self, tr_data, tr_labels, vec_size, validation_data=[], validation_labels=[]):
        super(Perceptron, self).__init__()
        self.tr_data = tr_data
        self.tr_labels = tr_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.vec_size = vec_size
        self.w = np.random.rand(self.vec_size)
      
    def train(self):        
        self.evaluate_every = 3 
        self.eta = 0.025
        self.n = 500
        
        errors = []
        #bias = 1

        for i in xrange(self.n):
            shuffle_indices = np.random.permutation(np.arange(len(self.tr_labels)))
            #print(shuffle_indices)
            tr_data = self.tr_data[shuffle_indices]
            tr_labels = self.tr_labels[shuffle_indices]
            for j in xrange(len(tr_labels)):
                result = dot(self.w, tr_data[j])
                error = tr_labels[j] - unit_step_function(result)
                errors.append(error)
                self.w += self.eta * error * tr_data[j]
        
            preds = []
            if (i%self.evaluate_every)== 0 and len(self.validation_labels) != 0:
                for each in self.validation_data:
                    result = dot(each, self.w)
                    #print("{} -> {}".format(result, unit_step_function(result)))
                    preds.append(unit_step_function(result))
                print(i)
            
                #print(preds)
                #print(test_labels)
                precision, recall, f1 = evaluate(preds, self.validation_labels)
                
                print "Perceptron results in order of Precision, Recall and F1", precision, recall, f1
                #np.savetxt("PW.csv", self.w, delimiter=",")
                #print(self.w)
                #print("----------")
                
        #print(self.w)
                
    def inference(self, test_data, w=[]):
        preds = []
        if(len(w) == 0):
            w=self.w
        for each in test_data:
            result = dot(each, w)
            #print("{} -> {}".format(result, unit_step_function(result)))
            preds.append(unit_step_function(result))
            #print(i)
        return np.array(preds)        

'''

def markerstovec(text):
    f = open('markers.txt', 'r')
    markers = f.readlines()
    #print(markers)
    #print(len(markers))
    features = []
    for each in markers:
        each = each.strip()
        if(len(each) == 0):
            continue
        lis = re.split('[;. |, |\*|\n]',each) 
        res = preprocess_pipeline(" ".join(lis), lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)
        elem = "_".join(res)
        features.append(elem)
    
    #print(features)
    
    size = len(features)
    vec = np.zeros(size)
    
    n = 0
    count = 0
    tex = re.split('[;. |, |\*|\n]',text) 
    #print(text)     
    for n in xrange(1,6):
        grams = ngrams(tex, n)
        for gram in grams:
            #print(gram)       
            res = preprocess_pipeline(" ".join(gram), lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)
            elem = "_".join(res)
            #print(elem)
            if elem in features:
                inx = features.index(elem)
                vec[inx] += 1
                count = count+1
                #print(elem) 
    
    #print(count)
    #print(len(vec))
    return vec
              
def main():

    #a = markerstovec("New Delhi has in contrast to this been selected as one of the hundred Indian cities to be developed as a smart city under Prime Minister of India Narendra Modi's flagship Smart Cities Mission.")
    #print(a)
    #return

    with open('urls_784.pickle', 'rb') as handle:
        url_dict = pickle.load(handle)
        dict_url = {v: k for k, v in url_dict.iteritems()}
        
    with open('url_784_body.pickle', 'rb') as handle:
        post_body = pickle.load(handle)
        
    print(len(url_dict))
    print(len(dict_url))
    print(len(post_body))  
    
    print(post_body)  
    
    #with open('url_784_title_post.pickle', 'rb') as handle:
    #    post_title = pickle.load(handle)

    with open("final_results.txt") as f:
        res = f.readlines()
        print(len(res))
        for each in res:
            #print(res)
            each = each.strip()    
            each = each.split(",")
            #print(each)
            inx = int(((each[0]).split("-"))[1])    
            val = each[1]
            if(int(val) == -1):
                continue
            else:
                val = int(val)
            res_dict[inx] = val
    
    inx_list = res_dict.keys()
    print(len(res_dict))
    pos_data = []
    neg_data = []
    pos_labels = []
    neg_labels = []
    #print(dict_url)
    for key,val in post_body.iteritems():
        #print("hey")
        #print(key)
        if url_dict[key] not in inx_list:
            continue
        #print("hey")    
        text = val.values()
        new_text = []
        for each in text:
            new_text.append(each.strip().encode('utf-8'))
        text = " ".join(new_text)
        if(res_dict[url_dict[key]] == 1):
            pos_data.append(text)  
            pos_labels.append(res_dict[url_dict[key]])  
        else:
            neg_data.append(text)  
            neg_labels.append(res_dict[url_dict[key]])    
        
    #print(pos_data[0])
    #print(neg_data[0])
    features = feature_extractor(pos_data[0:70] + pos_data[140:], neg_data[:112] + neg_data[224:], pos_labels[0:70] + pos_labels[140:], neg_labels[:112] + neg_labels[224:])
    
    #features = feature_extractor(pos_data[100:], neg_data[90:], pos_labels[100:], neg_labels[:90])
    
    vec_size = len(features)
    pos_vec_train_x = prepare_any_data(pos_data[0:70] + pos_data[140:], features)
    pos_vec_test_x = prepare_any_data(pos_data[70:140], features)
    
    
    neg_vec_train_x = prepare_any_data(neg_data[:112] + neg_data[224:], features)
    neg_vec_test_x = prepare_any_data(neg_data[112:224], features)
    
    #pos_vec_train_x = prepare_any_data(pos_data[100:], features)
    #pos_vec_test_x = prepare_any_data(pos_data[:100], features)
    
    
    #neg_vec_train_x = prepare_any_data(neg_data[90:], features)
    #neg_vec_test_x = prepare_any_data(neg_data[:90], features)
    
    
    #print(vec_test_x[0])                    
    #print(vec_size)
    
    
    train_data = np.append(pos_vec_train_x, neg_vec_train_x, axis=0) 
    test_data = np.append(pos_vec_test_x, neg_vec_test_x, axis=0)
    
    train_labels = np.array(pos_labels[0:70] + pos_labels[140:] + neg_labels[:112] + neg_labels[224:])
    test_labels = pos_labels[70:140] + neg_labels[112:224]
    
    #train_labels = np.array(pos_labels[100:] + neg_labels[90:])
    #test_labels = pos_labels[:100] + neg_labels[:90]
    
    
    shuffle_indices = np.random.permutation(np.arange(len(train_data)))
    
    train_data = train_data[shuffle_indices]
    
    train_labels = train_labels[shuffle_indices]
    
    print(len(train_data))
    print(len(train_labels))
    print(len(test_data))
    print(len(test_labels))
    
    
    svc = svm.SVC(kernel='linear', C=1)

    #svc = svm.SVC(C=8192, kernel='rbf', degree=2, gamma=0.00048828125, coef0=0.0, shrinking=True, probability=True,tol=0.001, cache_size=200,       class_weight=None, verbose=False, max_iter=-1, random_state=None)

    
    
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)    

    clf=svc	
    
    clf.fit(train_data, train_labels)

    #print clf.named_steps['feature_selection'].get_support()	#prints the support vectors


    #clf.probability=True

    #preds = clf.predict_proba(X_1)[:,1]                    
    preds = clf.predict(test_data)
    preds = preds.tolist()
    print(preds)
    
    precision, recall, f1 = evaluate(preds, test_labels)
    
    print "Final Results", precision, recall, f1
    
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(train_data, train_labels)

    #print clf.named_steps['feature_selection'].get_support()	#prints the support vectors


    #clf.probability=True

    #preds = clf.predict_proba(X_1)[:,1]                    
    preds = clf.predict(test_data)
    preds = preds.tolist()
    print(preds)
    
    precision, recall, f1 = evaluate(preds, test_labels)
    
    print "Final Results", precision, recall, f1 
  
'''
def main():

    argparser = argparse.ArgumentParser()
    with open("sentences.txt") as f:
        data = f.readlines()
    with open("labels.txt") as g:
        labels = [int(label) for label in g.read()[:-1].split("\n")]

    features = feature_extractor(data, labels)
    vec_size = len(features)
    
    tr_data, tr_labels, validation_data, validation_labels = prepare_training_data(data, labels, features)
    
    
    
    
    myperceptron = Perceptron(tr_data, tr_labels, vec_size, validation_data, validation_labels)
    
    #myperceptron.train()

    myMLP = MLP(tr_data, tr_labels, vec_size, validation_data, validation_labels)
    
    #myMLP.train()
    
        
    #Testing on unseen testing data in grading
    
    argparser.add_argument("--test_data", type=str, default="../test_sentences.txt", help="The real testing data in grading")
    argparser.add_argument("--test_labels", type=str, default="../test_labels.txt", help="The labels for the real testing data in grading")

    parsed_args = argparser.parse_args(sys.argv[1:])
    real_test_sentences = parsed_args.test_data
    real_test_labels = parsed_args.test_labels
    with open(real_test_sentences) as f:
        real_test_x = f.readlines()
    with open(real_test_labels) as g:
        real_test_y = [int(label) for label in g.read()[:-1].split("\n")]
        
    
    #predicted_y = mymlp.inference()
    #precision, recall, f1 = evaluate(predicted_y, test_y)
    #print "MLP results", precision, recall, f1
    vec_test_x = prepare_any_data(real_test_x, features)
    
    PW= np.genfromtxt ('PW.csv', delimiter=",")
    #Inference through perceptron
    predicted_y = myperceptron.inference(vec_test_x, PW)
    
    
    W0= np.genfromtxt ('W0.csv', delimiter=",")
    W1= np.genfromtxt ('W1.csv', delimiter=",")
    
    #Uncomment this to get inference through MLP
    #predicted_y = myMLP.inference(vec_test_x, W0, W1)
    
    precision, recall, f1 = evaluate(predicted_y, real_test_y)
    #print "Final Perceptron results in order of Precision, Recall and F1", precision, recall, f1
    print "Final MLP results in order of Precision, Recall and F1", precision, recall, f1
    
'''

        

if __name__ == '__main__':
    main()
