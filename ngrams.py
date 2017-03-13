### Ngram model for identifying question results type
### Hector Salvador Lopez

'''
These functions generate word indices, later used to find 
quickly word appearance on documents. Indices are stored as json
files in a folder called indices/.
'''
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import random
import os

# Define some global variables for word preprocessing and ngrams
START_CHAR = '<s>'
END_CHAR = '<e>'
UNK = '<u>'
PUNCTUATION = '".,;:+*`´¨!?¡¿&%()/' + "'"
N = 3

def preprocess(sentence, n):
    '''
    Remove whitespaces and punctuation, turn into lowercase, add necessary starting and ending
    dummy words. 
    Takes:
        - sentence, a strings
        - n, an int indicating the n-grams
    
    Returns:
        - words, a list of words stripped of punctuation and with additional characters
        to build ngram model
    '''
    words = sentence.lower().strip().split()
    words = [START_CHAR]*(n-1) + words + [END_CHAR]
    words = [word.strip(PUNCTUATION) for word in words]
    words = [word for word in words if len(word) != 0]

    return words

def init_dict():
    return {'_wordcount': 0}

def add_words(d, words, n):
    '''
    Modifies d, a dictionary with counts of ngrams, on site.
    Takes:
        - d, a dictionary where words will be added
        - words, a list of strings representing words
        - n, size of ngrams
    '''
    for i in range(len(words) - n):
        prefix = ' '.join(words[i:i + n])
        value  = words[i + n]
        d['_wordcount'] += 1
        if prefix not in d:
            d[prefix] = init_dict()
        temp = d[prefix]
        temp['_wordcount'] += 1
        if value not in temp:
            temp[value] = 0
        temp[value] += 1
        
def calculate_probability(words, corpus, n):
    '''
    Takes:
        words, a preprocessed list of words, according to n
        corpus, a dictionary with counts corresponding to a set of sentences
        n, an int indicating size of ngrams
        
    Returns:
        probability (float) of a word in the model
    '''
    logp = 0
    for i in range(len(words) - n + 1):
        
        if n == 1:
            if words[i] not in corpus['']: 
                temp_p = 1.0 / corpus['']['_wordcount'] 
            else:
                temp_p = 1.0 * corpus[''][words[i]] / corpus['']['_wordcount'] 
                
        else:
            prefix = ' '.join(words[i:i + n - 1])
            if (prefix not in corpus) or (words[i + n - 1] not in corpus[prefix]): 
                temp_p = 1.0 / corpus['_wordcount'] 
            else:
                temp_p = 1.0 * corpus[prefix][words[i + n - 1]] / corpus[prefix]['_wordcount']
        logp += math.log(temp_p)
    
    return math.exp(logp)

def linear_smoothing(lambdas, words_list, model_list, n):
    '''
    Takes:
        - lambdas, a list of float values to weight probabilities
        - words_list, a list of lists of words with START and END characters 
        - model_list, a list of models with ngram counts
        - n, an int indicating ngram size
    
    Returns:
        rv, a lambda-weighted probability of finding a word in the model_list
    '''
    rv = 0
    for l, words, model, i in zip(lambdas, words_list, model_list, range(n)):
        rv += l*calculate_probability(words, model, i+1)

    return rv


class NGramClassifier(object):
    
    def __init__(self, lambda1, lambda2, lambda3):
        #assert lambda1 + lambda2 + lambda3 == 1
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3
        self._unigram, self._bigram, self._trigram = {}, {}, {}
        self.label = set()
    
    @property
    def unigrams(self):
        return self._unigram
    
    @property
    def bigrams(self):
        return self._bigram
    
    @property
    def trigrams(self):
        return self._trigram
    
    @property
    def lambdas(self):
        return [self.lambda1, self.lambda2, self.lambda3]
    
    def modify_lambdas(self, l1, l2, l3):
        self.lambda1, self.lambda2, self.lambda3 = l1, l2, l3
    
    def build_ngram_model(self, n, list_of_sentences):
        '''
        Takes:
            - n, an int indicating number in n-grams
            - list_of_sentences, a list of strings with sentences
            - label, a string indicating the label
            
        Returns:
            - ngram, 
        '''
        model = init_dict()
        for i, sentence in list_of_sentences.iterrows():
            words = preprocess(sentence.text, n)
            add_words(model, words, n-1)
            
        return model
              
    def train(self, list_of_sentences, label):
        '''
        Takes:
            - list_of_sentences, a list of lists of strings [[...],[...],...,[...]]
            - label, a string indicating the label of interest
        '''
        self.label.add(label)
        self._unigram[label] = self.build_ngram_model(1, list_of_sentences)
        self._bigram[label] = self.build_ngram_model(2, list_of_sentences)
        self._trigram[label] = self.build_ngram_model(3, list_of_sentences)
    
    def call_model_list_by_label(self, label):
        return [self.unigrams[label], self.bigrams[label], self.trigrams[label]]
        
    def predict_one_sentence(self, sentence):
        '''
        Takes:
            - sentence, a list of strings (words)
            
        Returns:
            - results, a dictionary where the key, value pairs are label, likelihood of
            text being of such label
        '''
        lambdas = self.lambdas
        words_list = [preprocess(sentence, n+1) for n in range(N)]
        results = {}
        for label in self.label:
            model_list = self.call_model_list_by_label(label)
            results[label] = linear_smoothing(lambdas, words_list, model_list, N)
        return results
    
    def predict(self, list_of_sentences, prob=True):
        '''
        Takes:
            - list_of_sentences, a pandas df
        
        Returns:
            - rv, a list of predicted label (strings) 
        '''
        rv = []
        for i, sentence in list_of_sentences.iterrows():
            results = self.predict_one_sentence(sentence.text)
            if prob:
                if results[0] == 0:
                    if results[1] != 0:
                        score = 1
                    else:
                        score = 0.5
                else:
                    score = results[1]-results[0]
            else:
                max_cat = 0
                max_val = 0
                for key, value in results.items():
                    if value > max_val:
                        max_val = value
                        max_cat = key
                score = max_cat
            rv.append(score)
            
        return rv
    
    # results table
    def get_labeled_predictions(self, data):
        '''
        Takes:
            - data, a dictionary with train, test, and holdout data
            
        Returns:
            - rv, predicted labels for test data
        '''
        rv = pd.DataFrame(columns=['prediction', 'label'])
        for label in self.label:
            test = data.data[label]['test']
            results = self.predict(test)
            labels = [label]*len(results)
            temp = pd.DataFrame({'prediction': results, 'label': labels})
            rv = rv.append(temp)

        return rv

    #get rates from results table 
    def get_metrics(self, predictions):
        '''
        Takes:
            - predictions, a list of labels
        
        Returns:
            - tp, fn, fp, fn: true positive, false negative, false positive, and false negative
            rates
        '''
        tp, fn, fp, tn = {}, {}, {}, {}
        for label in self.label:
            g = predictions[predictions['label'] == label]
            tp[label] = g[g['prediction'] == label].label.count()
            fn[label] = g[g['prediction'] != label].label.count()
            for other_label in self.label:
                if other_label != label:
                    og = predictions[predictions['label'] == other_label]
                    fp[label] = og[og['prediction'] == label].label.count()
                    tn[label] = og[og['prediction'] != label].label.count()

        return tp, fn, fp, fn

    def micro_avg_precision(self, tp, fp):
        tp_sum = 0
        fp_sum = 0
        for label in self.label:
            tp_sum += tp[label]
            fp_sum += fp[label]

        return 1.0 * tp_sum / (tp_sum + fp_sum)

    def micro_avg_recall(self, tp, fn):
        tp_sum = 0
        fn_sum = 0
        for label in self.label:
            tp_sum += tp[label]
            fn_sum += fn[label]

        return 1.0 * tp_sum / (tp_sum + fn_sum)

    def test(self, data, l1 = 0, l2=0.5, l3=0.5):
        '''
        Takes:
            - data, a dictionary with train, test, and holdout data
            - l1, l2, l3, lambdas (floats) to weight ngram models
        '''
        self.modify_lambdas(l1, l2, l3)
        labeled_preds = self.get_labeled_predictions(data)
        #print(labeled_preds)
        tp, fn, fp, fn = self.get_metrics(labeled_preds)
        p = self.micro_avg_precision(tp, fp)
        r = self.micro_avg_recall(tp, fn)
        
        return 2*p*r/(p+r)

class Data(object):
    def __init__(self):
        self.data = {}
        self.populate_df()
    
    def replace_unk(self, train):
        '''
        Takes:
            - train, a dataframe
        '''
        words = []
        for i, row in train.iterrows():
            #print("working with", row, i)
            text = row.text.split()
            text = [x for x in text if len(x.strip()) > 0]
            for j in range(len(text)):
                if text[j] not in words:
                    words.append(text[j])
                    text[j] = UNK
            train['text'].ix[i] = ' '.join(text)
        return train 

    def split(self, data):
        '''
        Splits data into train, test, and holdout sets. Also modifies
        the training data to include <UNK> character to correct for unknown words.
        '''
        split = train_test_split(data)
        working, holdout = split[0], split[1]
        train, test = train_test_split(working)
        #train = self.replace_unk(train)
        return train, test, holdout

    def populate_df(self):  
        path_pfx = os.getcwd()
        for label in [0, 1]:
            if label == 1:
                filename = path_pfx + '/multas.txt'
            elif label == 0:
                filename = path_pfx + '/nomultas.txt'
            text = open(filename).read().split('\n')
            text = [i for i in text if len(i.strip()) > 0]
            rv = pd.DataFrame({'text': text, 'labels': [label]*len(text)})
            train, test, holdout = self.split(rv)
            self.data[label] = {'train': train, 'test': test, 'holdout': holdout}  
            
def train_test():
    data = Data()
    clf = NGramClassifier(0.3, 0.2, 0.5) 
    clf.train(data.data[1]['train'], 1)
    clf.train(data.data[0]['train'], 0)

    max_val = 0
    max_weights = []
    for j in range(100):
        for i in range(100):
            l1, l2 = 0.01 * j, 0.01 * i
            l3 = 1 - l1 - l2
            if i%100 == 0 : print('Trying {}, {}, {}... '.format(l1,l2,l3))
            f1 = clf.test(data, l1, l2, l3)
            #print(f1)
            if f1 > max_val:
                max_val = f1
                max_weights = [l1, l2, l3]
                print(max_val, max_weights)

    print("Best model has an f1 of {} and lambdas of {}".format(max_val, max_weights))

def retrieve_model():
    '''Called by passage retrieval function.
    '''
    data = Data()
    clf = NGramClassifier(0, 0, 1) 
    clf.train(data.data[1]['train'], 1)
    clf.train(data.data[0]['train'], 0)
    return clf
